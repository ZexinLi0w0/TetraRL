"""Week 7 Task 5: FFmpeg co-runner interference harness.

Provides:

* :class:`LatencyRecorder` — lightweight per-step timestamp logger using
  ``time.perf_counter_ns`` (monotonic, nanosecond resolution).
* :class:`FFmpegInterference` — context manager that spawns an ``ffmpeg``
  subprocess to introduce a controlled CPU/GPU co-runner workload, then
  reaps it cleanly on exit.
* :func:`run_workload` — drives a TetraRL framework + Gym env loop while a
  recorder samples per-step wall time.
* :func:`summarize` — produces a Markdown table of per-condition tail
  percentiles and the slowdown ratio relative to a baseline (``none``).
* :func:`ffmpeg_available` — best-effort PATH probe for the ``ffmpeg``
  binary, used by tests to skip cleanly when it's missing.

The protocol mirrors the one used by Li et al. (RTSS '23, Fig. 15): run a
training workload alone, then alongside ``ffmpeg`` decoding a 720p, 1080p,
and 2K H.264 stream, and report 99th-percentile training-step latency.
This module's defaults use ``-f lavfi -i testsrc=...`` so the harness has
no external video file dependency.
"""
from __future__ import annotations

import json
import statistics  # noqa: F401  -- kept for downstream extensions
import subprocess
import time
from typing import Iterable, Literal, Optional, Sequence

# stdlib only per spec; numpy is intentionally avoided.

ResolutionLiteral = Literal["720p", "1080p", "2K", "none"]

_RESOLUTION_TO_WH: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "2K": (2560, 1440),
}

# Codec names we treat as "hardware H.264 decode is available" on Orin.
_HW_DECODE_HINTS: tuple[str, ...] = ("nvv4l2dec", "nvdec", "cuda")


def ffmpeg_available() -> bool:
    """Return True iff the ``ffmpeg`` binary is on PATH and runs.

    Used by tests and the CLI driver to gracefully skip work when the
    binary is missing (e.g., minimal CI containers).
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


class LatencyRecorder:
    """Lightweight per-step timestamp logger using ``perf_counter_ns``.

    Usage::

        rec = LatencyRecorder()
        rec.start()
        for _ in range(N):
            do_work()
            rec.mark()      # appends (now - last) ms to ``samples_ms``
        pcts = rec.percentiles([50.0, 90.0, 99.0, 99.9])
    """

    def __init__(self) -> None:
        self._samples_ms: list[float] = []
        self._last_ns: int | None = None

    def start(self) -> None:
        """Record the baseline timestamp (next ``mark()`` is relative to this)."""
        self._last_ns = time.perf_counter_ns()

    def mark(self) -> None:
        """Append the elapsed milliseconds since the previous ``start``/``mark``."""
        now_ns = time.perf_counter_ns()
        if self._last_ns is None:
            # If start() wasn't called, treat this mark as the baseline.
            self._last_ns = now_ns
            return
        delta_ms = (now_ns - self._last_ns) / 1e6
        # perf_counter_ns is monotonic; clamp tiny negatives that could
        # arise from clock resolution to 0 for defensive safety.
        if delta_ms < 0.0:
            delta_ms = 0.0
        self._samples_ms.append(delta_ms)
        self._last_ns = now_ns

    @property
    def samples_ms(self) -> list[float]:
        """Raw per-step samples in milliseconds (read-only view via copy)."""
        return list(self._samples_ms)

    def percentiles(self, p: Sequence[float]) -> dict[float, float]:
        """Linear-interpolated percentiles over the recorded samples.

        Empty samples raise ``ValueError`` to surface mis-use in the
        driver script (a 0-step run has no meaningful percentile).
        """
        if not self._samples_ms:
            raise ValueError("no samples recorded; call mark() at least once")
        sorted_samples = sorted(self._samples_ms)
        n = len(sorted_samples)
        out: dict[float, float] = {}
        for pct in p:
            if not (0.0 <= pct <= 100.0):
                raise ValueError(f"percentile {pct} outside [0, 100]")
            if n == 1:
                out[float(pct)] = sorted_samples[0]
                continue
            # Type-7 linear interpolation (R default; matches numpy default).
            rank = (pct / 100.0) * (n - 1)
            lo = int(rank)  # floor
            hi = min(lo + 1, n - 1)
            frac = rank - lo
            value = sorted_samples[lo] + frac * (sorted_samples[hi] - sorted_samples[lo])
            out[float(pct)] = float(value)
        return out

    def to_jsonl(self, path: str) -> None:
        """Write one JSON object per sample: ``{"sample_ms": float, "idx": int}``."""
        with open(path, "w", encoding="utf-8") as f:
            for idx, sample in enumerate(self._samples_ms):
                f.write(json.dumps({"sample_ms": float(sample), "idx": int(idx)}) + "\n")


class FFmpegInterference:
    """Context manager spawning an ``ffmpeg`` co-runner subprocess.

    The lifecycle is::

        with FFmpegInterference(resolution="720p") as ff:
            ...   # workload runs here, ff.process is alive
        # __exit__ has SIGTERMed and reaped the subprocess.

    With ``resolution="none"`` no subprocess is spawned (for the baseline
    condition). The ``__exit__`` path uses ``terminate -> wait(5) -> kill``
    so it is safe under ``KeyboardInterrupt`` and exception unwinding —
    Python's ``with`` statement guarantees ``__exit__`` runs in both cases.
    """

    def __init__(
        self,
        resolution: ResolutionLiteral,
        video_path: Optional[str] = None,
        hw_decode: bool = False,
    ) -> None:
        self.resolution = resolution
        self.video_path = video_path
        self.hw_decode = bool(hw_decode)
        self._process: Optional[subprocess.Popen] = None
        # Cached argv for diagnostic / test inspection.
        self._argv: Optional[list[str]] = None

    @staticmethod
    def _resolution_to_wh(res: str) -> tuple[int, int]:
        if res not in _RESOLUTION_TO_WH:
            raise ValueError(
                f"unknown resolution {res!r}; expected one of {list(_RESOLUTION_TO_WH)}"
            )
        return _RESOLUTION_TO_WH[res]

    @staticmethod
    def _hw_decode_available() -> bool:
        """Probe ``ffmpeg -hwaccels`` for an Orin-class H.264 hw decoder.

        On Mac this returns False (only ``videotoolbox`` is listed). On
        Orin with a JetPack ffmpeg build, ``nvv4l2dec`` (or ``nvdec`` /
        ``cuda``) appears in the output.
        """
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        if proc.returncode != 0:
            return False
        text = proc.stdout.decode("utf-8", errors="replace").lower()
        return any(hint in text for hint in _HW_DECODE_HINTS)

    @staticmethod
    def build_argv(
        resolution: str,
        video_path: Optional[str],
        hw_decode: bool,
    ) -> list[str]:
        """Return the ffmpeg argv list for the given condition.

        Exposed as a static method so tests can assert on the argv
        without spawning a real subprocess.
        """
        if resolution == "none":
            raise ValueError("build_argv should not be called with resolution='none'")
        w, h = FFmpegInterference._resolution_to_wh(resolution)

        argv: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        # Only the real-video path can use the hw decoder; lavfi testsrc
        # generates raw frames, so an h264 decoder slot is meaningless.
        use_hw = (
            hw_decode
            and video_path is not None
            and FFmpegInterference._hw_decode_available()
        )
        if use_hw:
            argv += ["-c:v", "h264_nvv4l2dec"]

        if video_path is not None:
            # Loop the input forever so the co-runner outlives the workload.
            argv += ["-stream_loop", "-1", "-i", str(video_path)]
        else:
            # Synthetic source — no external file dependency.
            argv += [
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size={w}x{h}:rate=30",
            ]

        # Discard decoded output; we only care about the CPU/GPU load.
        argv += ["-f", "null", "-"]
        return argv

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._process

    @property
    def argv(self) -> Optional[list[str]]:
        return list(self._argv) if self._argv is not None else None

    def __enter__(self) -> "FFmpegInterference":
        if self.resolution == "none":
            # No-op baseline.
            self._process = None
            self._argv = None
            return self
        self._argv = self.build_argv(
            resolution=self.resolution,
            video_path=self.video_path,
            hw_decode=self.hw_decode,
        )
        # stdout/stderr to DEVNULL so the co-runner can't pollute the host
        # terminal or saturate a pipe buffer.
        self._process = subprocess.Popen(
            self._argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    # Final wait so we don't leave a zombie behind.
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        # Best effort — log and move on rather than hang.
                        pass
        finally:
            self._process = None


def run_workload(
    framework,
    env,
    n_steps: int,
    recorder: "LatencyRecorder",
) -> None:
    """Drive ``framework.step`` + ``env.step`` for ``n_steps`` and record latency.

    Each loop iteration records one sample = wall time between successive
    ``recorder.mark()`` calls, i.e. the round-trip cost of one framework
    decision plus one environment step. Episodes are auto-reset.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    obs, _info = env.reset()
    recorder.start()
    for _ in range(int(n_steps)):
        rec = framework.step(obs)
        action = int(rec["action"])
        step_out = env.step(action)
        # Gymnasium returns (obs, reward, terminated, truncated, info).
        obs = step_out[0]
        reward = float(step_out[1])
        terminated = bool(step_out[2])
        truncated = bool(step_out[3])
        framework.observe_reward(reward)
        recorder.mark()
        if terminated or truncated:
            obs, _info = env.reset()


def summarize(results: dict[str, "LatencyRecorder"]) -> str:
    """Return a Markdown table comparing per-condition tail latencies.

    Columns:

      condition | n | p50_ms | p90_ms | p99_ms | p99.9_ms | slowdown_p99

    where ``slowdown_p99 = condition_p99 / baseline_p99``. The baseline is
    the recorder keyed by ``'none'`` if present; otherwise the slowdown
    column reads ``N/A``.
    """
    header = (
        "| condition | n | p50_ms | p90_ms | p99_ms | p99.9_ms | slowdown_p99 |"
    )
    sep = (
        "|-----------|---|--------|--------|--------|----------|--------------|"
    )

    baseline_p99: Optional[float] = None
    baseline_rec = results.get("none")
    if baseline_rec is not None and baseline_rec.samples_ms:
        baseline_p99 = baseline_rec.percentiles([99.0])[99.0]

    rows: list[str] = []
    for cond, rec in results.items():
        n = len(rec.samples_ms)
        if n == 0:
            rows.append(
                f"| {cond} | 0 | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        pcts = rec.percentiles([50.0, 90.0, 99.0, 99.9])
        if baseline_p99 is not None and baseline_p99 > 0:
            slowdown = pcts[99.0] / baseline_p99
            slow_str = f"{slowdown:.2f}x"
        else:
            slow_str = "N/A"
        rows.append(
            f"| {cond} | {n} | {pcts[50.0]:.3f} | {pcts[90.0]:.3f} | "
            f"{pcts[99.0]:.3f} | {pcts[99.9]:.3f} | {slow_str} |"
        )

    return "\n".join([header, sep, *rows]) + "\n"
