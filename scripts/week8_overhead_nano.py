"""Week 8 Task 3: Per-component overhead measurement on Jetson Nano.

Drives ``TetraRLFramework`` on CartPole-v1 twice:

  (a) **Bare-RL baseline** — ``env.step`` + ``arbiter.act`` only,
      no framework wiring, no profiler. Yields a per-step wall-clock
      reference (``mean_bare_step_ms``).
  (b) **Framework pass** — full ``framework.step()`` with an
      :class:`OverheadProfiler` attached. The framework auto-times its
      6 in-step components; the loop here additionally wraps a
      ``ReplayBuffer.add()`` call (the 7th component
      ``replay_buffer_add``) and, when ``--with-lag-feature`` is on,
      a ``LAGFeatureExtractor.extract`` call (separate
      ``lag_feature_extract`` row).

Outputs (under ``--out-dir``):

    overhead_table.md        Paper Table 5 candidate (per-component
                             mean / p50 / p99 / mem / rss).
    overhead_breakdown.csv   Per-sample CSV from OverheadProfiler.
    lag_feature_overhead.md  Only when ``--with-lag-feature`` is set;
                             carries the W8 < 0.5 ms p99 check.
    summary.json             Headline numbers for ``result.md``.

Acceptance (printed banner):

    PASS iff
        framework_overhead_pct < 5.0  AND
        (--with-lag-feature  =>  lag p99_ms < 0.5)

Mac dev note: ``--no-real-tegrastats`` and ``--no-real-dvfs`` should be
passed when running off-Nano so the script falls back to
``PsutilTelemetrySource`` / stub DVFS and never touches sysfs.

Example::

    python3 scripts/week8_overhead_nano.py --n-steps 5000 \\
        --out-dir runs/w8_overhead_nano/ \\
        --with-lag-feature
"""
from __future__ import annotations

import argparse
import json
import os
import platform as _pyplatform
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Make the repository root importable so we can pull in tetrarl.* and the
# week7 driver helpers even when invoked as
# ``python scripts/week8_overhead_nano.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from scripts.week7_nano_cartpole import (  # noqa: E402
    _RandomArbiter,
    _telemetry_to_hw,
    make_nano_framework,
)
from tetrarl.eval.overhead import OverheadProfiler  # noqa: E402
from tetrarl.morl.native.lag_feature import LAGFeatureExtractor  # noqa: E402
from tetrarl.sys.buffer import ReplayBuffer  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device_uname() -> str:
    """Return a short ``uname -a``-style identifier (no shell-out required)."""
    u = _pyplatform.uname()
    return f"{u.system} {u.node} {u.release} {u.machine}"


def _bare_rl_pass(n_steps: int, seed: int) -> tuple[float, int]:
    """Run a bare-RL baseline (env.step + arbiter.act) and return (mean_ms, total_steps).

    Uses ``time.perf_counter_ns()`` directly per step; no profiler involvement
    so the bare path stays as cheap as possible.
    """
    import gymnasium as gym  # lazy import keeps test collection cheap

    env = gym.make("CartPole-v1")
    n_actions = int(env.action_space.n)
    arbiter = _RandomArbiter(n_actions=n_actions, seed=seed)
    omega = np.array([0.5, 0.5], dtype=np.float32)

    step_times_ns: list[int] = []
    obs, _ = env.reset(seed=seed)
    try:
        for _ in range(n_steps):
            t0 = time.perf_counter_ns()
            action = arbiter.act(obs, omega)
            obs, _r, terminated, truncated, _ = env.step(action)
            elapsed_ns = time.perf_counter_ns() - t0
            step_times_ns.append(elapsed_ns)
            if bool(terminated or truncated):
                obs, _ = env.reset(seed=seed)
    finally:
        env.close()

    if not step_times_ns:
        return 0.0, 0
    mean_ms = float(np.mean(step_times_ns)) / 1_000_000.0
    return mean_ms, len(step_times_ns)


def _framework_pass(
    n_steps: int,
    seed: int,
    platform: str,
    use_real_dvfs: bool,
    use_real_tegrastats: bool,
    track_memory: bool,
    with_lag_feature: bool,
) -> tuple[OverheadProfiler, float, int, Optional[str]]:
    """Run the full framework pass under the profiler.

    Returns ``(profiler, mean_framework_step_ms, total_steps, deferred_dvfs_reason)``.
    """
    import gymnasium as gym  # lazy import

    env = gym.make("CartPole-v1")
    n_actions = int(env.action_space.n)

    # Reuse the week7 nano builder so we share telemetry + DVFS plumbing.
    fw, telemetry, _override, deferred = make_nano_framework(
        n_actions=n_actions,
        seed=seed,
        platform=platform,
        # Pick a permissive override threshold; we are timing overhead,
        # not validating override semantics here.
        max_memory_util=0.99,
        with_override=False,
        use_real_dvfs=use_real_dvfs,
        use_real_tegrastats=use_real_tegrastats,
    )

    prof = OverheadProfiler(track_memory=track_memory)
    # The framework stores ``self.profiler`` in ``__init__``; assigning it
    # here picks up the auto-timing of the 6 in-step components on the very
    # next ``fw.step()`` call.
    fw.profiler = prof

    # 7th component: ReplayBuffer.add (timed manually in the loop).
    obs_dim = int(np.prod(env.observation_space.shape))
    buf = ReplayBuffer(
        capacity=1024,
        obs_shape=(obs_dim,),
        act_shape=(),
    )

    extractor: Optional[LAGFeatureExtractor] = None
    if with_lag_feature:
        extractor = LAGFeatureExtractor(soft_latency_ms=50.0, n_corunners=1)

    energy_remaining = 1000.0
    framework_step_times_ns: list[int] = []

    obs, _ = env.reset(seed=seed)
    try:
        for _ in range(n_steps):
            # Optional LAG feature extraction. We sample telemetry directly
            # so the LAG row is captured even though framework.step() also
            # samples telemetry internally (under tegra_daemon_sample).
            if extractor is not None:
                hw = _telemetry_to_hw(telemetry.latest())
                with prof.time("lag_feature_extract"):
                    _ = extractor.extract(hw)
                # Materialise the appended-state version to mirror the real
                # training-time wiring (``state = extractor.append_to_state(state, hw)``).
                _ = extractor.append_to_state(obs, hw)

            t_fw0 = time.perf_counter_ns()
            record = fw.step(obs)
            t_fw1 = time.perf_counter_ns()
            framework_step_times_ns.append(t_fw1 - t_fw0)
            action = int(record["action"])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            with prof.time("replay_buffer_add"):
                buf.add(obs, action, float(reward), next_obs, done)

            fw.observe_reward(float(reward))
            energy_remaining = max(0.0, energy_remaining - 1e-3)
            telemetry.update(latency_ms=0.0, energy_remaining_j=energy_remaining)

            if done:
                obs, _ = env.reset(seed=seed)
            else:
                obs = next_obs
    finally:
        env.close()
        if hasattr(telemetry, "stop"):
            try:
                telemetry.stop()
            except Exception:
                pass

    mean_ms = (
        float(np.mean(framework_step_times_ns)) / 1_000_000.0
        if framework_step_times_ns
        else 0.0
    )
    return prof, mean_ms, len(framework_step_times_ns), deferred


def _write_overhead_table_md(
    path: Path,
    prof: OverheadProfiler,
    *,
    device: str,
    platform: str,
    n_steps: int,
    mean_bare_step_ms: float,
    mean_framework_step_ms: float,
    framework_overhead_pct: float,
) -> None:
    summary = prof.summarize()
    header_lines = [
        "# Week 8 — Per-component Overhead (Table 5 candidate)",
        "",
        f"- device: `{device}`",
        f"- platform: `{platform}`",
        f"- n_steps: {n_steps}",
        f"- mean_bare_step_ms: {mean_bare_step_ms:.4f}",
        f"- mean_framework_step_ms: {mean_framework_step_ms:.4f}",
        f"- framework_overhead_pct: {framework_overhead_pct:.4f}",
        f"- total_components_profiled: {len(summary)}",
        "",
    ]
    body = prof.to_markdown()
    path.write_text("\n".join(header_lines) + "\n" + body, encoding="utf-8")


def _write_lag_feature_md(
    path: Path,
    prof: OverheadProfiler,
    *,
    device: str,
    platform: str,
    n_steps: int,
) -> dict:
    """Write the LAG-feature-only overhead report. Returns the LAG summary row."""
    summary = prof.summarize()
    lag_row = summary.get("lag_feature_extract", {})
    p99_ms = float(lag_row.get("p99_ms", float("nan")))
    criterion = "PASS" if (lag_row and p99_ms < 0.5) else "FAIL"

    lines = [
        "# Week 8 — LAG Feature Extract Overhead",
        "",
        f"- device: `{device}`",
        f"- platform: `{platform}`",
        f"- n_steps: {n_steps}",
        "- W8 criterion: `lag_feature_extract.p99_ms < 0.5`",
        f"- result: **{criterion}**",
        "",
        "| component | mean_ms | p50_ms | p99_ms | mem_mb | rss_mb | n_samples |",
        "|---|---|---|---|---|---|---|",
    ]
    if lag_row:
        lines.append(
            f"| lag_feature_extract | {lag_row['mean_ms']:.4f} |"
            f" {lag_row['p50_ms']:.4f} | {lag_row['p99_ms']:.4f} |"
            f" {lag_row['mem_mb']:.4f} | {lag_row['rss_mb']:.4f} |"
            f" {int(lag_row['n_samples'])} |"
        )
    else:
        lines.append("| lag_feature_extract | n/a | n/a | n/a | n/a | n/a | 0 |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return lag_row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-steps", type=int, default=5000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--platform",
        default="nano",
        choices=["nano", "orin_agx"],
    )
    p.add_argument(
        "--with-lag-feature",
        action="store_true",
        default=False,
        help="Also profile LAGFeatureExtractor.extract (Week 8 LAG criterion).",
    )
    p.add_argument(
        "--no-real-tegrastats",
        action="store_true",
        help="Don't spawn the real tegrastats binary; use psutil/vm_stat instead.",
    )
    p.add_argument(
        "--no-real-dvfs",
        action="store_true",
        help="Force DVFS stub mode (don't write sysfs).",
    )
    p.add_argument(
        "--no-strict",
        action="store_true",
        help="Always exit 0 (measurement-only run).",
    )
    p.add_argument(
        "--track-memory",
        dest="track_memory",
        action="store_true",
        default=True,
        help="Enable tracemalloc + RSS deltas in the profiler (default).",
    )
    p.add_argument(
        "--no-track-memory",
        dest="track_memory",
        action="store_false",
        help="Disable per-sample memory tracking.",
    )
    p.add_argument(
        "--effort",
        choices=["max", "fast"],
        default="fast",
        help="Cosmetic effort tag stored in summary.json (W8 spec).",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed everything we control. The replay buffer / framework helpers
    # do their own seeding via ``_RandomArbiter(seed=...)`` etc.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Bare-RL baseline ----
    mean_bare_step_ms, _bare_steps = _bare_rl_pass(
        n_steps=args.n_steps, seed=args.seed,
    )

    # ---- Framework pass ----
    prof, mean_framework_step_ms, _fw_steps, deferred = _framework_pass(
        n_steps=args.n_steps,
        seed=args.seed,
        platform=args.platform,
        use_real_dvfs=not args.no_real_dvfs,
        use_real_tegrastats=not args.no_real_tegrastats,
        track_memory=args.track_memory,
        with_lag_feature=args.with_lag_feature,
    )

    if mean_bare_step_ms > 0.0:
        framework_overhead_pct = (
            (mean_framework_step_ms - mean_bare_step_ms) / mean_bare_step_ms * 100.0
        )
    else:
        framework_overhead_pct = float("nan")

    # ---- Artifacts ----
    device = _device_uname()

    overhead_table_path = out_dir / "overhead_table.md"
    _write_overhead_table_md(
        overhead_table_path,
        prof,
        device=device,
        platform=args.platform,
        n_steps=args.n_steps,
        mean_bare_step_ms=mean_bare_step_ms,
        mean_framework_step_ms=mean_framework_step_ms,
        framework_overhead_pct=framework_overhead_pct,
    )

    overhead_csv_path = out_dir / "overhead_breakdown.csv"
    prof.to_csv(overhead_csv_path)

    lag_row: dict = {}
    lag_md_path: Optional[Path] = None
    if args.with_lag_feature:
        lag_md_path = out_dir / "lag_feature_overhead.md"
        lag_row = _write_lag_feature_md(
            lag_md_path,
            prof,
            device=device,
            platform=args.platform,
            n_steps=args.n_steps,
        )

    # ---- summary.json ----
    summary = {
        "platform": args.platform,
        "effort": args.effort,
        "n_steps": int(args.n_steps),
        "with_lag_feature": bool(args.with_lag_feature),
        "use_real_dvfs": not args.no_real_dvfs,
        "use_real_tegrastats": not args.no_real_tegrastats,
        "track_memory": bool(args.track_memory),
        "hostname": socket.gethostname(),
        "device": device,
        "pid": os.getpid(),
        "mean_bare_step_ms": float(mean_bare_step_ms),
        "mean_framework_step_ms": float(mean_framework_step_ms),
        "framework_overhead_pct": float(framework_overhead_pct),
        "components": prof.summarize(),
        "deferred_dvfs_reason": deferred,
        "artifacts": {
            "overhead_table_md": str(overhead_table_path),
            "overhead_breakdown_csv": str(overhead_csv_path),
            "lag_feature_overhead_md": (
                str(lag_md_path) if lag_md_path is not None else None
            ),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- Acceptance banner ----
    overhead_ok = (
        framework_overhead_pct == framework_overhead_pct  # NaN check (NaN != NaN)
        and framework_overhead_pct < 5.0
    )
    lag_ok = True
    if args.with_lag_feature:
        lag_ok = bool(lag_row) and float(lag_row.get("p99_ms", float("inf"))) < 0.5

    print()
    print(f"mean_bare_step_ms       = {mean_bare_step_ms:.4f}")
    print(f"mean_framework_step_ms  = {mean_framework_step_ms:.4f}")
    print(f"framework_overhead_pct  = {framework_overhead_pct:.4f}")
    if args.with_lag_feature and lag_row:
        print(f"lag_feature_extract.p99_ms = {lag_row['p99_ms']:.4f}")
    if deferred:
        print(f"deferred_dvfs_reason   = {deferred}")

    if overhead_ok and lag_ok:
        print("ACCEPTANCE: PASS")
        return 0
    print("ACCEPTANCE: FAIL")
    if args.no_strict:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
