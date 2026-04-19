"""Week 9 Task A — Nano deep-validation on the 4-D DAG env.

Sweeps the TetraRL stack (preference plane -> arbiter -> resource
manager -> override layer -> DVFS) across 3 preference vectors omega
(``energy_corner``, ``memory_corner``, ``center``) on
:class:`DAGSchedulerEnv` with 4-D reward
``[throughput, -energy_step, -peak_memory_delta, -energy_norm_step]``.

Wraps :mod:`tetrarl.eval.runner`'s component factories so the ablation,
override, and telemetry contracts match the W8 runner; the only
additional logic is per-config omega injection, the 4-D-reward
scalarisation needed by the framework's scalar :meth:`observe_reward`,
and a platform-specific telemetry / DVFS swap.

Run on Orin Nano (sudo userspace governor required for real DVFS)::

    python scripts/week9_nano_dag_sweep.py \\
        --n-episodes 200 \\
        --omegas energy_corner,memory_corner,center \\
        --out-dir runs/w9_nano_dag/ \\
        --platform orin_nano
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tetrarl.core.framework import (  # noqa: E402
    ResourceManager,
    StaticPreferencePlane,
    TetraRLFramework,
)
from tetrarl.envs.dag_scheduler import DAGSchedulerEnv  # noqa: E402
from tetrarl.morl.native.override import (  # noqa: E402
    HardwareTelemetry,
    OverrideLayer,
    OverrideThresholds,
)

OMEGAS_4D: dict[str, np.ndarray] = {
    "energy_corner": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "memory_corner": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "center": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
}


def parse_omegas(s: str) -> list[tuple[str, np.ndarray]]:
    """Parse a comma-separated string of omega keywords into (name, vector) pairs."""
    if not s or not s.strip():
        raise ValueError("omegas string is empty")
    out: list[tuple[str, np.ndarray]] = []
    for raw in s.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in OMEGAS_4D:
            raise ValueError(
                f"unknown omega {name!r}; choices: {sorted(OMEGAS_4D)}"
            )
        out.append((name, OMEGAS_4D[name].copy()))
    if not out:
        raise ValueError("omegas string contained no recognised names")
    return out


class _DAGRandomArbiter:
    """Uniform-random over ``Discrete(n_tasks)``.

    Invalid actions are no-ops in :class:`DAGSchedulerEnv`, so we don't
    need to consult the valid mask here. Seed-controlled for sweep
    reproducibility.
    """

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = int(n_actions)
        self._rng = np.random.default_rng(int(seed))

    def act(self, state: Any, omega: np.ndarray) -> int:
        return int(self._rng.integers(0, self.n_actions))


class _MacReading:
    __slots__ = ("latency_ema_ms", "energy_remaining_j", "memory_util")

    def __init__(
        self,
        latency_ema_ms: float,
        energy_remaining_j: float,
        memory_util: float,
    ) -> None:
        self.latency_ema_ms = float(latency_ema_ms)
        self.energy_remaining_j = float(energy_remaining_j)
        self.memory_util = float(memory_util)


class _MacStubTelemetry:
    """In-memory telemetry stub; mirrors runner._MacStubTelemetry."""

    def __init__(self, initial_energy_j: float = 1000.0) -> None:
        self._latency = 0.0
        self._energy = float(initial_energy_j)
        self._memory_util = 0.1

    def update(
        self,
        latency_ms: float,
        energy_remaining_j: float,
        memory_util: float,
    ) -> None:
        self._latency = float(latency_ms)
        self._energy = float(energy_remaining_j)
        self._memory_util = float(memory_util)

    def latest(self) -> _MacReading:
        return _MacReading(self._latency, self._energy, self._memory_util)


def _telemetry_to_hw(reading: Any) -> HardwareTelemetry:
    return HardwareTelemetry(
        latency_ema_ms=getattr(reading, "latency_ema_ms", None),
        energy_remaining_j=getattr(reading, "energy_remaining_j", None),
        memory_util=getattr(reading, "memory_util", None),
    )


def _build_framework(
    n_actions: int,
    omega: np.ndarray,
    seed: int,
    platform: str,
) -> tuple[TetraRLFramework, Any, Optional[str]]:
    pref = StaticPreferencePlane(np.asarray(omega, dtype=np.float32).copy())
    rm = ResourceManager()
    arbiter = _DAGRandomArbiter(n_actions=n_actions, seed=seed)
    override = OverrideLayer(
        thresholds=OverrideThresholds(
            max_latency_ms=2.0,
            min_energy_j=0.5,
            max_memory_util=0.95,
        ),
        fallback_action=0,
        cooldown_steps=0,
    )

    deferred: Optional[str] = None
    dvfs = None
    if platform == "mac_stub":
        telemetry: Any = _MacStubTelemetry(initial_energy_j=1000.0)
    else:
        try:
            from scripts.week7_nano_cartpole import (  # noqa: E402
                TegraTelemetrySource,
                _build_dvfs,
            )
            try:
                telemetry = TegraTelemetrySource(platform=platform)
            except Exception as exc:
                deferred = (
                    f"tegra telemetry unavailable ({type(exc).__name__}: {exc!s}); "
                    "falling back to mac_stub telemetry."
                )
                telemetry = _MacStubTelemetry(initial_energy_j=1000.0)
            dvfs, dvfs_deferred = _build_dvfs(
                platform=platform, use_real_dvfs=True
            )
            if dvfs_deferred and deferred is None:
                deferred = dvfs_deferred
        except ImportError as exc:
            deferred = (
                f"week7 nano helpers unavailable ({exc!s}); "
                "falling back to mac_stub telemetry."
            )
            telemetry = _MacStubTelemetry(initial_energy_j=1000.0)

    fw = TetraRLFramework(
        preference_plane=pref,
        rl_arbiter=arbiter,
        resource_manager=rm,
        override_layer=override,
        telemetry_source=telemetry,
        telemetry_adapter=_telemetry_to_hw,
        dvfs_controller=dvfs,
    )
    return fw, telemetry, deferred


def _run_one_omega(
    n_episodes: int,
    omega: np.ndarray,
    out_path: Path,
    seed: int,
    n_tasks: int,
    density: float,
    platform: str,
) -> dict:
    env = DAGSchedulerEnv(
        n_tasks=n_tasks, density=density, seed=seed, reward_dim=4
    )
    n_actions = int(env.action_space.n)
    fw, telemetry, deferred = _build_framework(
        n_actions=n_actions, omega=omega, seed=seed, platform=platform,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rewards_scalar: list[float] = []
    latencies: list[float] = []
    energies: list[float] = []
    memories: list[float] = []
    n_steps = 0
    energy_remaining = 1000.0
    t0 = time.perf_counter()

    with open(out_path, "w", encoding="utf-8") as fh:
        for ep in range(int(n_episodes)):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            episode_step = 0
            while not done:
                t_fw0 = time.perf_counter()
                record = fw.step(obs)
                t_fw1 = time.perf_counter()
                fw_dt_ms = (t_fw1 - t_fw0) * 1000.0
                action = int(record["action"])

                t_env0 = time.perf_counter()
                obs, reward_vec, terminated, truncated, _ = env.step(action)
                t_env1 = time.perf_counter()
                env_dt_ms = (t_env1 - t_env0) * 1000.0

                latency_ms = fw_dt_ms + env_dt_ms
                reward_arr = np.asarray(reward_vec, dtype=np.float32)
                scalar_r = float(np.dot(omega, reward_arr))
                fw.observe_reward(scalar_r)

                energy_step = -float(reward_arr[1])
                memory_step = -float(reward_arr[2])
                energy_remaining = max(0.0, energy_remaining - energy_step)
                memory_util = min(1.0, 0.1 + memory_step / 16.0)
                telemetry.update(
                    latency_ms=latency_ms,
                    energy_remaining_j=energy_remaining,
                    memory_util=memory_util,
                )

                rec = {
                    "episode": int(ep),
                    "step": int(episode_step),
                    "action": int(action),
                    "reward_vec": [float(x) for x in reward_arr],
                    "scalarised_reward": float(scalar_r),
                    "latency_ms": float(latency_ms),
                    "energy_step": float(energy_step),
                    "memory_step": float(memory_step),
                    "omega": [float(x) for x in omega],
                }
                fh.write(json.dumps(rec) + "\n")

                rewards_scalar.append(scalar_r)
                latencies.append(latency_ms)
                energies.append(energy_step)
                memories.append(memory_step)
                episode_step += 1
                n_steps += 1
                done = bool(terminated or truncated)

    wall_time = time.perf_counter() - t0
    if hasattr(telemetry, "stop"):
        try:
            telemetry.stop()
        except Exception:
            pass

    return {
        "n_episodes": int(n_episodes),
        "n_steps": int(n_steps),
        "mean_scalarised_reward": (
            float(np.mean(rewards_scalar)) if rewards_scalar else 0.0
        ),
        "tail_p99_ms": (
            float(np.percentile(latencies, 99)) if latencies else 0.0
        ),
        "mean_energy_step": (
            float(np.mean(energies)) if energies else 0.0
        ),
        "mean_memory_delta": (
            float(np.mean(memories)) if memories else 0.0
        ),
        "wall_time_s": float(wall_time),
        "deferred_dvfs_reason": deferred,
    }


def run_sweep(
    n_episodes: int,
    omegas: list[tuple[str, np.ndarray]],
    out_dir: Path,
    seed: int = 0,
    n_tasks: int = 8,
    density: float = 0.3,
    platform: str = "orin_nano",
) -> dict:
    """Run one DAGSchedulerEnv sweep per omega; aggregate to summary.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for omega_name, omega in omegas:
        per = out_dir / omega_name
        per.mkdir(parents=True, exist_ok=True)
        result = _run_one_omega(
            n_episodes=n_episodes,
            omega=omega,
            out_path=per / "trace.jsonl",
            seed=seed,
            n_tasks=n_tasks,
            density=density,
            platform=platform,
        )
        summary_rows.append({"omega_name": omega_name, **result})

    csv_path = out_dir / "summary.csv"
    cols = [
        "omega_name",
        "n_episodes",
        "n_steps",
        "mean_scalarised_reward",
        "tail_p99_ms",
        "mean_energy_step",
        "mean_memory_delta",
        "wall_time_s",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in summary_rows:
            w.writerow([r.get(c, "") for c in cols])
    return {"per_omega": summary_rows, "summary_csv": str(csv_path)}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument(
        "--omegas",
        default="energy_corner,memory_corner,center",
        help="Comma-separated omega keywords from {energy_corner, memory_corner, center}.",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-tasks", type=int, default=8)
    p.add_argument("--density", type=float, default=0.3)
    p.add_argument(
        "--platform",
        default="orin_nano",
        choices=["mac_stub", "orin_nano", "nano", "orin_agx"],
    )
    args = p.parse_args(argv)

    omegas = parse_omegas(args.omegas)
    summary = run_sweep(
        n_episodes=args.n_episodes,
        omegas=omegas,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        n_tasks=args.n_tasks,
        density=args.density,
        platform=args.platform,
    )

    print(
        json.dumps(
            {
                "per_omega": summary["per_omega"],
                "summary_csv": summary["summary_csv"],
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
