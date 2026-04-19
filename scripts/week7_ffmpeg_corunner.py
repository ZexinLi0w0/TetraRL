"""Week 7 Task 5 deliverable: FFmpeg co-runner interference harness.

Drives the TetraRLFramework on CartPole-v1 under several FFmpeg
co-runner conditions (none / 720p / 1080p / 2K) and writes per-step
latency JSONL plus a Markdown summary table comparing tail percentiles.

Mirrors Li et al. (RTSS '23, Fig. 15) protocol: training alone vs
training + ffmpeg decode of an H.264 stream at increasing resolution.
On Mac the synthetic ``testsrc=`` lavfi source is used (no external
video file dependency); on Orin point ``--video`` at a real H.264
sample for an apples-to-apples comparison.

Example::

    python3 scripts/week7_ffmpeg_corunner.py \\
        --n-steps 5000 \\
        --conditions none,720p,1080p,2K \\
        --out-dir runs/week7_ffmpeg_<unix_ts> \\
        [--video PATH] [--hw-decode]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Make the repository root importable so we can pull in week6_e2e_smoke.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.week6_e2e_smoke import make_framework  # noqa: E402
from tetrarl.eval.ffmpeg_interference import (  # noqa: E402
    FFmpegInterference,
    LatencyRecorder,
    ffmpeg_available,
    run_workload,
    summarize,
)


_NEEDS_FFMPEG = {"720p", "1080p", "2K"}


def _parse_conditions(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    valid = {"none", "720p", "1080p", "2K"}
    bad = [p for p in parts if p not in valid]
    if bad:
        raise SystemExit(
            f"unknown conditions {bad!r}; valid choices are {sorted(valid)}"
        )
    return parts


def _run_one_condition(
    condition: str,
    n_steps: int,
    video_path: Optional[str],
    hw_decode: bool,
    out_dir: Path,
    seed: int,
) -> LatencyRecorder:
    """Run a single condition end-to-end, returning the recorder."""
    import gymnasium as gym  # lazy import keeps test collection fast

    rec = LatencyRecorder()
    with FFmpegInterference(
        resolution=condition,
        video_path=video_path,
        hw_decode=hw_decode,
    ):
        env = gym.make("CartPole-v1")
        try:
            fw, _telemetry, _override = make_framework(
                n_actions=int(env.action_space.n), seed=seed
            )
            run_workload(fw, env, n_steps=n_steps, recorder=rec)
        finally:
            env.close()

    out_path = out_dir / f"{condition}.jsonl"
    rec.to_jsonl(str(out_path))
    return rec


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument(
        "--conditions",
        type=str,
        default="none,720p,1080p,2K",
        help="Comma-separated list of conditions.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory; defaults to runs/week7_ffmpeg_<unix_ts>.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to a real H.264 video file (Orin runs).",
    )
    parser.add_argument(
        "--hw-decode",
        action="store_true",
        help="Request h264_nvv4l2dec when available (Orin).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cooldown-s",
        type=float,
        default=2.0,
        help="Seconds to sleep between conditions.",
    )
    args = parser.parse_args(argv)

    conditions = _parse_conditions(args.conditions)

    if args.out_dir is None:
        ts = int(time.time())
        out_dir = _REPO_ROOT / "runs" / f"week7_ffmpeg_{ts}"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    have_ffmpeg = ffmpeg_available()
    if not have_ffmpeg:
        ffmpeg_conds = [c for c in conditions if c in _NEEDS_FFMPEG]
        if ffmpeg_conds and "none" not in conditions:
            print(
                f"ffmpeg not on PATH; cannot run conditions {ffmpeg_conds}. "
                "Add 'none' to --conditions to record a baseline only."
            )
            return 0
        if ffmpeg_conds:
            print(
                f"ffmpeg not on PATH; skipping conditions {ffmpeg_conds}, "
                f"running 'none' baseline only."
            )
            conditions = [c for c in conditions if c not in _NEEDS_FFMPEG]

    print(f"Output directory: {out_dir}")
    print(f"Conditions      : {conditions}")
    print(f"n_steps         : {args.n_steps}")
    print(f"Video path      : {args.video!r}")
    print(f"hw_decode       : {args.hw_decode}")

    results: dict[str, LatencyRecorder] = {}
    for i, cond in enumerate(conditions):
        print(f"\n[{i + 1}/{len(conditions)}] Running condition: {cond}")
        t0 = time.perf_counter()
        rec = _run_one_condition(
            condition=cond,
            n_steps=args.n_steps,
            video_path=args.video,
            hw_decode=args.hw_decode,
            out_dir=out_dir,
            seed=args.seed,
        )
        dt = time.perf_counter() - t0
        if rec.samples_ms:
            pcts = rec.percentiles([50.0, 99.0])
            print(
                f"  {cond}: {len(rec.samples_ms)} samples in {dt:.2f}s "
                f"(p50={pcts[50.0]:.3f} ms, p99={pcts[99.0]:.3f} ms)"
            )
        else:
            print(f"  {cond}: 0 samples in {dt:.2f}s")
        results[cond] = rec

        if i < len(conditions) - 1 and args.cooldown_s > 0:
            print(f"  cooldown {args.cooldown_s:.1f}s ...")
            time.sleep(args.cooldown_s)

    summary_md = summarize(results)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    print(f"\nSummary written to {summary_path}")
    print()
    print(summary_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
