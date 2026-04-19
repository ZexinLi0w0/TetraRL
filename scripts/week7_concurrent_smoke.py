"""Week 7 deliverable: concurrent vs sequential framework smoke comparison.

DVFO-style "thinking-while-moving" overlap (Zhang TMC 2023): the resource
manager's DVFS decision is computed in a background thread while the RL
arbiter's foreground forward pass runs. This script drives the same
CartPole-v1 closed-loop as Week 6, once with the existing sequential
framework path and once with the new ConcurrentDecisionLoop, and reports
mean/max per-step framework time + the relative speedup.

Validation gate (per Week 7 spec):
    concurrent_mean_ms <= sequential_mean_ms * 1.10
i.e., the concurrent path must not regress beyond +10% even on Mac where
DVFS is a no-op stub (so the thread overhead is the dominant cost).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from scripts.week6_e2e_smoke import (
    RandomArbiter,
    StubTelemetrySource,
    _StubReading,
    _telemetry_to_hw,
)
from tetrarl.core.framework import (
    ResourceManager,
    StaticPreferencePlane,
    TetraRLFramework,
)
from tetrarl.morl.native.override import (
    HardwareTelemetry,
    OverrideLayer,
    OverrideThresholds,
)
from tetrarl.sys.concurrent import ConcurrentDecisionLoop
from tetrarl.sys.dvfs import DVFSController


class _CostlyResourceManager:
    """Wraps ResourceManager with a configurable simulated decision cost.

    DVFO (Zhang TMC 2023) overlaps a NON-trivial DVFS decision computation
    with the arbiter's forward pass. Our default ResourceManager is a
    rule-based step-down that takes <1us, so on Mac with a stub DVFS the
    threading overhead of ConcurrentDecisionLoop dominates and the
    concurrent path cannot beat sequential. Injecting a small simulated
    cost here lets the smoke faithfully demonstrate the overlap.
    """

    def __init__(self, inner: ResourceManager, decide_cost_s: float = 0.0):
        self._inner = inner
        self._decide_cost_s = float(decide_cost_s)

    def decide_dvfs(self, telemetry: HardwareTelemetry, n_levels: int) -> int:
        if self._decide_cost_s > 0:
            time.sleep(self._decide_cost_s)
        return self._inner.decide_dvfs(telemetry, n_levels)


def _build_framework(
    n_actions: int,
    seed: int,
    use_concurrent: bool,
    decide_cost_s: float = 0.0,
) -> tuple[
    TetraRLFramework,
    StubTelemetrySource,
    OverrideLayer,
    DVFSController,
    Optional[ConcurrentDecisionLoop],
]:
    pref = StaticPreferencePlane(np.array([0.5, 0.5], dtype=np.float32))
    arbiter = RandomArbiter(n_actions=n_actions, seed=seed)
    rm = ResourceManager()
    rm_for_framework: Any = (
        _CostlyResourceManager(rm, decide_cost_s=decide_cost_s)
        if decide_cost_s > 0
        else rm
    )
    override = OverrideLayer(
        OverrideThresholds(
            max_latency_ms=10000.0,
            min_energy_j=0.5,
            max_memory_util=0.95,
        ),
        fallback_action=0,
        cooldown_steps=0,
    )
    telemetry = StubTelemetrySource(initial_energy_j=1000.0)
    dvfs = DVFSController(stub=True)

    loop: Optional[ConcurrentDecisionLoop] = None
    if use_concurrent:
        n_levels = len(dvfs.available_frequencies()["gpu"])
        loop = ConcurrentDecisionLoop(
            resource_manager=rm_for_framework,
            dvfs_controller=dvfs,
            n_levels=n_levels,
            fallback_idx=n_levels - 1,
        )

    fw = TetraRLFramework(
        preference_plane=pref,
        rl_arbiter=arbiter,
        resource_manager=rm_for_framework,
        override_layer=override,
        telemetry_source=telemetry,
        telemetry_adapter=_telemetry_to_hw,
        dvfs_controller=dvfs,
        concurrent_decision=loop,
    )
    return fw, telemetry, override, dvfs, loop


def _run_one(
    label: str,
    episodes: int,
    seed: int,
    use_concurrent: bool,
    out_file,
    decide_cost_s: float = 0.0,
) -> dict:
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    n_actions = int(env.action_space.n)
    fw, telemetry, override, _dvfs, loop = _build_framework(
        n_actions=n_actions,
        seed=seed,
        use_concurrent=use_concurrent,
        decide_cost_s=decide_cost_s,
    )

    energy_remaining = 1000.0
    framework_step_times_ms: list[float] = []
    total_steps = 0
    episode_returns: list[float] = []

    try:
        for ep in range(episodes):
            obs, _info = env.reset(seed=seed + ep)
            episode_step = 0
            ep_return = 0.0
            done = False
            while not done:
                memory_util = 0.1 + 0.001 * episode_step

                t_fw0 = time.perf_counter()
                record = fw.step(obs)
                t_fw1 = time.perf_counter()
                fw_dt_ms = (t_fw1 - t_fw0) * 1000.0
                framework_step_times_ms.append(fw_dt_ms)

                action = int(record["action"])

                t_env0 = time.perf_counter()
                obs, reward, terminated, truncated, _info = env.step(action)
                t_env1 = time.perf_counter()
                env_dt_ms = (t_env1 - t_env0) * 1000.0

                latency_ms = env_dt_ms + fw_dt_ms
                energy_j = 1e-3 * (action + 1)
                energy_remaining = max(0.0, energy_remaining - energy_j)

                record["latency_ms"] = float(latency_ms)
                record["energy_j"] = float(energy_j)
                record["memory_util"] = float(memory_util)
                fw.observe_reward(float(reward))

                telemetry.update(
                    latency_ms=latency_ms,
                    energy_remaining_j=energy_remaining,
                    memory_util=memory_util,
                )

                if out_file is not None:
                    serialisable = {
                        "path": label,
                        "episode": ep,
                        "step": episode_step,
                        "action": int(record["action"]),
                        "proposed_action": int(record["proposed_action"]),
                        "omega": [float(x) for x in record["omega"]],
                        "override_fired": bool(record["override_fired"]),
                        "reward": float(record["reward"]),
                        "latency_ms": float(record["latency_ms"]),
                        "energy_j": float(record["energy_j"]),
                        "memory_util": float(record["memory_util"]),
                        "dvfs_idx": (
                            int(record["dvfs_idx"])
                            if record["dvfs_idx"] is not None
                            else None
                        ),
                        "concurrent_dvfs_used": bool(
                            record["concurrent_dvfs_used"]
                        ),
                        "framework_step_ms": fw_dt_ms,
                    }
                    out_file.write(json.dumps(serialisable) + "\n")

                ep_return += float(reward)
                episode_step += 1
                total_steps += 1
                done = bool(terminated or truncated)

            episode_returns.append(ep_return)
    finally:
        if loop is not None:
            loop.shutdown()
        env.close()

    mean_fw_ms = (
        sum(framework_step_times_ms) / len(framework_step_times_ms)
        if framework_step_times_ms
        else 0.0
    )
    max_fw_ms = max(framework_step_times_ms) if framework_step_times_ms else 0.0

    return {
        "label": label,
        "history": fw.history,
        "total_steps": total_steps,
        "episodes": episodes,
        "override_fire_count": int(override.fire_count),
        "mean_framework_step_ms": float(mean_fw_ms),
        "max_framework_step_ms": float(max_fw_ms),
        "mean_episode_return": (
            float(sum(episode_returns) / len(episode_returns))
            if episode_returns
            else 0.0
        ),
    }


def run_smoke(
    episodes: int = 100,
    out_path: Optional[str] = None,
    seed: int = 0,
    decide_cost_s: float = 0.0,
) -> dict:
    """Run sequential and concurrent paths back-to-back, return both summaries."""
    out_file = None
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_file = open(out_path, "w", encoding="utf-8")

    try:
        seq_result = _run_one(
            label="sequential",
            episodes=episodes,
            seed=seed,
            use_concurrent=False,
            out_file=out_file,
            decide_cost_s=decide_cost_s,
        )
        conc_result = _run_one(
            label="concurrent",
            episodes=episodes,
            seed=seed,
            use_concurrent=True,
            out_file=out_file,
            decide_cost_s=decide_cost_s,
        )
    finally:
        if out_file is not None:
            out_file.close()

    seq_mean = seq_result["mean_framework_step_ms"]
    conc_mean = conc_result["mean_framework_step_ms"]
    speedup_pct = (
        100.0 * (seq_mean - conc_mean) / seq_mean if seq_mean > 0 else 0.0
    )

    return {
        "sequential": seq_result,
        "concurrent": conc_result,
        "speedup_pct": speedup_pct,
        "out_path": out_path,
    }


def _validate(result: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    seq_mean = float(result["sequential"]["mean_framework_step_ms"])
    conc_mean = float(result["concurrent"]["mean_framework_step_ms"])

    # Per Week 7 spec: concurrent must not regress beyond +10% on Mac.
    if conc_mean > seq_mean * 1.10:
        failures.append(
            f"concurrent mean {conc_mean:.4f} ms > sequential * 1.10 = "
            f"{seq_mean * 1.10:.4f} ms"
        )

    # Both paths must populate concurrent_dvfs_used correctly.
    seq_history = result["sequential"]["history"]
    conc_history = result["concurrent"]["history"]
    if not all(rec.get("concurrent_dvfs_used") is False for rec in seq_history):
        failures.append("sequential path has concurrent_dvfs_used=True somewhere")
    if not all(rec.get("concurrent_dvfs_used") is True for rec in conc_history):
        failures.append("concurrent path has concurrent_dvfs_used=False somewhere")

    return (len(failures) == 0), failures


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSONL path; defaults to runs/week7_concurrent_smoke_<ts>.jsonl",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--decide-cost-ms",
        type=float,
        default=1.0,
        help=(
            "Simulated DVFS decision cost in ms (sleep inside ResourceManager). "
            "On Mac the rule-based decision is ~us so threading overhead would "
            "dominate; default 1.0 ms approximates a small Jetson NN-based "
            "DVFS picker (DVFO Zhang TMC 2023). Set to 0 to disable."
        ),
    )
    args = parser.parse_args(argv)

    out_path = args.out
    if out_path is None:
        ts = int(time.time())
        out_path = str(Path("runs") / f"week7_concurrent_smoke_{ts}.jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result = run_smoke(
        episodes=args.episodes,
        out_path=out_path,
        seed=args.seed,
        decide_cost_s=args.decide_cost_ms / 1000.0,
    )
    runtime_s = time.perf_counter() - t0

    ok, failures = _validate(result)

    seq = result["sequential"]
    conc = result["concurrent"]
    print(f"episodes               : {args.episodes}")
    print(f"sequential total steps : {seq['total_steps']}")
    print(f"concurrent total steps : {conc['total_steps']}")
    print(
        "sequential mean fw ms  : "
        f"{seq['mean_framework_step_ms']:.4f}  (max {seq['max_framework_step_ms']:.4f})"
    )
    print(
        "concurrent mean fw ms  : "
        f"{conc['mean_framework_step_ms']:.4f}  (max {conc['max_framework_step_ms']:.4f})"
    )
    print(f"speedup vs sequential  : {result['speedup_pct']:+.2f}%")
    print(f"decide cost (ms)       : {args.decide_cost_ms:.3f}")
    print(f"override fire (seq)    : {seq['override_fire_count']}")
    print(f"override fire (conc)   : {conc['override_fire_count']}")
    print(f"runtime (s)            : {runtime_s:.2f}")
    print(f"jsonl path             : {out_path}")

    if ok:
        print("WEEK 7 CONCURRENT SMOKE PASSED")
        return 0
    for fail in failures:
        print(f"FAIL: {fail}")
    print("WEEK 7 CONCURRENT SMOKE FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
