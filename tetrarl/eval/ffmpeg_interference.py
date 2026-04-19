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
import time
from typing import Iterable, Sequence

# stdlib only per spec; numpy is intentionally avoided.


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
