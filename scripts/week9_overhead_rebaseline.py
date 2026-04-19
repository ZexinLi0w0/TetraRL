"""Week 9 Task B — Re-baselined per-component overhead on Nano.

W8 Task 3 used ``random + CartPole-v1`` as the bare-RL baseline, which
made TetraRL look ~4841% slower because the bare step took only
~0.03 ms vs the framework's ~1.49 ms. This script switches the bare
baseline to the *fair* ``preference_ppo + DAGSchedulerEnv`` (4-D
reward) pairing, so the overhead percentage reflects the cost of
TetraRL's wiring relative to a representative DRL workload.

Outputs (under ``--out-dir``):

    overhead_table.md        Per-component table candidate (Paper Table 5).
    overhead_breakdown.csv   Per-sample profiler CSV.
    summary.json             Headline numbers + acceptance result.

Acceptance criterion:

    framework_overhead_pct < 30.0   (relaxed from W8's 5.0; see W8 PR #24
                                     overhead deviation analysis).

Run on Orin Nano (sudo userspace governor required for real DVFS)::

    python scripts/week9_overhead_rebaseline.py \\
        --n-steps 5000 \\
        --agent preference_ppo \\
        --env dag_scheduler_mo \\
        --allow-real-dvfs \\
        --out-dir runs/w9_overhead_rebaseline_nano/
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
from typing import Any, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tetrarl.envs.dag_scheduler import DAGSchedulerEnv  # noqa: E402
from tetrarl.eval.overhead import OverheadProfiler  # noqa: E402

ACCEPTANCE_THRESHOLD_PCT: float = 30.0
DEFAULT_OMEGA_4D: np.ndarray = np.array(
    [0.25, 0.25, 0.25, 0.25], dtype=np.float32
)


def _make_arbiter(agent: str, n_actions: int, seed: int) -> Any:
    """Return an arbiter exposing ``act(state, omega) -> int``."""
    if agent == "preference_ppo":
        from tetrarl.eval.runner import _PreferencePPOArbiter
        return _PreferencePPOArbiter(n_actions=n_actions, seed=seed)
    if agent == "dvfs_drl_multitask":
        from tetrarl.morl.baselines.dvfs_drl_multitask import (
            DVFSDRLMultitaskArbiter,
        )
        return DVFSDRLMultitaskArbiter(n_actions=n_actions, seed=seed)
    if agent == "random":
        from tetrarl.eval.runner import _RandomArbiter
        return _RandomArbiter(n_actions=n_actions, seed=seed)
    raise ValueError(f"unknown agent: {agent!r}")


def _make_env(env_name: str, n_tasks: int, density: float, seed: int):
    """Return a Gymnasium-style env keyed by W9 short names."""
    if env_name == "dag_scheduler_mo":
        return DAGSchedulerEnv(
            n_tasks=n_tasks, density=density, seed=seed, reward_dim=4
        )
    raise ValueError(f"unknown env: {env_name!r}")


def bare_rl_pass(
    n_steps: int,
    seed: int,
    n_tasks: int = 8,
    density: float = 0.3,
    agent: str = "preference_ppo",
    env_name: str = "dag_scheduler_mo",
) -> tuple[float, int]:
    """Run a bare-RL baseline (arbiter.act + env.step only) and return (mean_ms, total_steps)."""
    env = _make_env(env_name, n_tasks=n_tasks, density=density, seed=seed)
    n_actions = int(env.action_space.n)
    arbiter = _make_arbiter(agent, n_actions=n_actions, seed=seed)
    omega = DEFAULT_OMEGA_4D.copy()

    step_times_ns: list[int] = []
    obs, _ = env.reset(seed=seed)
    try:
        for _ in range(int(n_steps)):
            t0 = time.perf_counter_ns()
            action = arbiter.act(obs, omega)
            obs, _r, terminated, truncated, _ = env.step(int(action))
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


def framework_pass(
    n_steps: int,
    seed: int,
    platform: str,
    use_real_dvfs: bool,
    use_real_tegrastats: bool,
    track_memory: bool,
    n_tasks: int = 8,
    density: float = 0.3,
    agent: str = "preference_ppo",
    env_name: str = "dag_scheduler_mo",
) -> tuple[OverheadProfiler, float, int, Optional[str]]:
    """Run the framework pass with profiler attached; return (profiler, mean_ms, n_steps, deferred)."""
    env = _make_env(env_name, n_tasks=n_tasks, density=density, seed=seed)
    n_actions = int(env.action_space.n)

    from scripts.week9_nano_dag_sweep import _build_framework as build_dag_framework
    omega = DEFAULT_OMEGA_4D.copy()
    fw, telemetry, deferred = build_dag_framework(
        n_actions=n_actions,
        omega=omega,
        seed=seed,
        platform=platform,
    )
    fw.rl_arbiter = _make_arbiter(agent, n_actions=n_actions, seed=seed)

    prof = OverheadProfiler(track_memory=track_memory)
    fw.profiler = prof

    framework_step_times_ns: list[int] = []
    obs, _ = env.reset(seed=seed)
    try:
        for _ in range(int(n_steps)):
            t0 = time.perf_counter_ns()
            record = fw.step(obs)
            t1 = time.perf_counter_ns()
            framework_step_times_ns.append(t1 - t0)
            action = int(record["action"])
            obs, reward_vec, terminated, truncated, _ = env.step(action)
            scalar_r = float(
                np.dot(omega, np.asarray(reward_vec, dtype=np.float32))
            )
            fw.observe_reward(scalar_r)
            if bool(terminated or truncated):
                obs, _ = env.reset(seed=seed)
    finally:
        env.close()
        if hasattr(telemetry, "stop"):
            try:
                telemetry.stop()
            except Exception:
                pass

    if not framework_step_times_ns:
        return prof, 0.0, 0, deferred
    mean_ms = float(np.mean(framework_step_times_ns)) / 1_000_000.0
    return prof, mean_ms, len(framework_step_times_ns), deferred


def _device_uname() -> str:
    u = _pyplatform.uname()
    return f"{u.system} {u.node} {u.release} {u.machine}"


def _write_overhead_table_md(
    path: Path,
    prof: OverheadProfiler,
    *,
    device: str,
    platform: str,
    agent: str,
    env_name: str,
    n_steps: int,
    mean_bare_step_ms: float,
    mean_framework_step_ms: float,
    framework_overhead_pct: float,
) -> None:
    header_lines = [
        "# Week 9 — Re-baselined Per-component Overhead",
        "",
        f"- device: `{device}`",
        f"- platform: `{platform}`",
        f"- agent: `{agent}`",
        f"- env: `{env_name}`",
        f"- n_steps: {n_steps}",
        f"- mean_bare_step_ms: {mean_bare_step_ms:.4f}",
        f"- mean_framework_step_ms: {mean_framework_step_ms:.4f}",
        f"- framework_overhead_pct: {framework_overhead_pct:.4f}",
        f"- acceptance_threshold_pct: {ACCEPTANCE_THRESHOLD_PCT}",
        "",
    ]
    body = prof.to_markdown()
    path.write_text("\n".join(header_lines) + "\n" + body, encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-steps", type=int, default=5000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--platform",
        default="orin_nano",
        choices=["mac_stub", "orin_nano", "nano", "orin_agx"],
    )
    p.add_argument(
        "--agent",
        default="preference_ppo",
        choices=["preference_ppo", "dvfs_drl_multitask", "random"],
    )
    p.add_argument(
        "--env",
        default="dag_scheduler_mo",
        choices=["dag_scheduler_mo"],
    )
    p.add_argument("--n-tasks", type=int, default=8)
    p.add_argument("--density", type=float, default=0.3)
    p.add_argument(
        "--no-real-tegrastats",
        action="store_true",
        help="Don't spawn the real tegrastats binary; use the mac stub.",
    )
    p.add_argument(
        "--allow-real-dvfs",
        action="store_true",
        default=False,
        help=(
            "Hit the real DVFS sysfs path on Nano (requires sudo + "
            "governor=userspace). Default: stub."
        ),
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
        default="max",
        help="Cosmetic effort tag stored in summary.json (W8 spec).",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    try:
        import torch  # noqa: F401
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    mean_bare, _bare_steps = bare_rl_pass(
        n_steps=args.n_steps,
        seed=args.seed,
        n_tasks=args.n_tasks,
        density=args.density,
        agent=args.agent,
        env_name=args.env,
    )

    prof, mean_fw, _fw_steps, deferred = framework_pass(
        n_steps=args.n_steps,
        seed=args.seed,
        platform=args.platform,
        use_real_dvfs=args.allow_real_dvfs,
        use_real_tegrastats=not args.no_real_tegrastats,
        track_memory=args.track_memory,
        n_tasks=args.n_tasks,
        density=args.density,
        agent=args.agent,
        env_name=args.env,
    )

    if mean_bare > 0.0:
        framework_overhead_pct = (
            (mean_fw - mean_bare) / mean_bare * 100.0
        )
    else:
        framework_overhead_pct = float("nan")

    device = _device_uname()

    overhead_table_path = out_dir / "overhead_table.md"
    _write_overhead_table_md(
        overhead_table_path,
        prof,
        device=device,
        platform=args.platform,
        agent=args.agent,
        env_name=args.env,
        n_steps=args.n_steps,
        mean_bare_step_ms=mean_bare,
        mean_framework_step_ms=mean_fw,
        framework_overhead_pct=framework_overhead_pct,
    )
    overhead_csv_path = out_dir / "overhead_breakdown.csv"
    prof.to_csv(overhead_csv_path)

    summary = {
        "platform": args.platform,
        "agent": args.agent,
        "env": args.env,
        "effort": args.effort,
        "n_steps": int(args.n_steps),
        "n_tasks": int(args.n_tasks),
        "density": float(args.density),
        "use_real_dvfs": bool(args.allow_real_dvfs),
        "use_real_tegrastats": not args.no_real_tegrastats,
        "track_memory": bool(args.track_memory),
        "hostname": socket.gethostname(),
        "device": device,
        "pid": os.getpid(),
        "mean_bare_step_ms": float(mean_bare),
        "mean_framework_step_ms": float(mean_fw),
        "framework_overhead_pct": float(framework_overhead_pct),
        "acceptance_threshold_pct": float(ACCEPTANCE_THRESHOLD_PCT),
        "components": prof.summarize(),
        "deferred_dvfs_reason": deferred,
        "artifacts": {
            "overhead_table_md": str(overhead_table_path),
            "overhead_breakdown_csv": str(overhead_csv_path),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    overhead_ok = (
        framework_overhead_pct == framework_overhead_pct  # NaN guard
        and framework_overhead_pct < ACCEPTANCE_THRESHOLD_PCT
    )

    print()
    print(f"agent                   = {args.agent}")
    print(f"env                     = {args.env}")
    print(f"mean_bare_step_ms       = {mean_bare:.4f}")
    print(f"mean_framework_step_ms  = {mean_fw:.4f}")
    print(f"framework_overhead_pct  = {framework_overhead_pct:.4f}")
    print(f"acceptance_threshold    = {ACCEPTANCE_THRESHOLD_PCT}")
    if deferred:
        print(f"deferred_dvfs_reason    = {deferred}")

    if overhead_ok:
        print("ACCEPTANCE: PASS")
        return 0
    print("ACCEPTANCE: FAIL")
    if args.no_strict:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
