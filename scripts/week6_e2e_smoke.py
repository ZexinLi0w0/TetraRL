"""Week 6 deliverable: end-to-end TetraRLFramework smoke on CartPole-v1.

Validates the 4-component framework wiring (Preference Plane, RL Arbiter,
Resource Manager, Hardware Override) against a real Gymnasium env with a
synthetic 4-D telemetry stream (reward, latency, energy, memory). Runs in
<60s on Mac CPU. Real-tegrastats Orin runs are performed by a separate
sequential agent on hardware.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

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
from tetrarl.sys.dvfs import DVFSController


# Per Week 6 spec, the RL Arbiter is treated as a black box; this random
# arbiter exercises the framework wiring without any policy training.
class RandomArbiter:
    """Uniform-random discrete-action arbiter (no training)."""

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = int(n_actions)
        self._rng = random.Random(seed)

    def act(self, state: Any, omega: np.ndarray) -> int:
        return self._rng.randint(0, self.n_actions - 1)


@dataclass
class _StubReading:
    latency_ema_ms: Optional[float] = None
    energy_remaining_j: Optional[float] = None
    memory_util: Optional[float] = None


class StubTelemetrySource:
    """In-memory telemetry stub fed by the smoke loop each step.

    Avoids the heavyweight ``TegrastatsDaemon`` so the smoke runs on Mac
    without sudo. The smoke loop writes synthetic 4-D values via
    :meth:`update` between env steps; ``latest()`` returns the most
    recent reading (framework calls latest() once per step).
    """

    def __init__(self, initial_energy_j: float = 1000.0):
        self._reading = _StubReading(
            latency_ema_ms=0.0,
            energy_remaining_j=float(initial_energy_j),
            memory_util=0.1,
        )

    def update(
        self,
        latency_ms: float,
        energy_remaining_j: float,
        memory_util: float,
    ) -> None:
        self._reading = _StubReading(
            latency_ema_ms=float(latency_ms),
            energy_remaining_j=float(energy_remaining_j),
            memory_util=float(memory_util),
        )

    def latest(self) -> _StubReading:
        return self._reading


def _telemetry_to_hw(reading: _StubReading) -> HardwareTelemetry:
    return HardwareTelemetry(
        latency_ema_ms=reading.latency_ema_ms,
        energy_remaining_j=reading.energy_remaining_j,
        memory_util=reading.memory_util,
    )


def make_framework(
    n_actions: int = 2,
    seed: int = 0,
    omega: Optional[np.ndarray] = None,
) -> tuple[TetraRLFramework, StubTelemetrySource, OverrideLayer]:
    """Build a fully-wired TetraRLFramework for the smoke test.

    ``omega`` selects the static preference vector; defaults to
    ``[0.5, 0.5]`` (the W6/W7 production preference).
    """
    if omega is None:
        omega = np.array([0.5, 0.5], dtype=np.float32)
    pref = StaticPreferencePlane(np.asarray(omega, dtype=np.float32))
    arbiter = RandomArbiter(n_actions=n_actions, seed=seed)
    rm = ResourceManager()
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
    fw = TetraRLFramework(
        preference_plane=pref,
        rl_arbiter=arbiter,
        resource_manager=rm,
        override_layer=override,
        telemetry_source=telemetry,
        telemetry_adapter=_telemetry_to_hw,
        dvfs_controller=dvfs,
    )
    return fw, telemetry, override


def run_smoke(
    episodes: int = 100,
    out_path: Optional[str] = None,
    seed: int = 0,
) -> dict:
    """Run the framework smoke for ``episodes`` CartPole-v1 episodes.

    Returns a summary dict containing the per-step ``history``, total
    steps, override fire count, mean framework wall-time per step, and
    JSONL output path (if any).
    """
    import gymnasium as gym  # imported lazily so test collection is cheap

    env = gym.make("CartPole-v1")
    n_actions = int(env.action_space.n)
    fw, telemetry, override = make_framework(n_actions=n_actions, seed=seed)

    out_file = None
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_file = open(out_path, "w", encoding="utf-8")

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
                # 1. Synthesize telemetry BEFORE the framework step (we
                #    will refine latency_ms after we time the env step).
                memory_util = 0.1 + 0.001 * episode_step

                # 2. Time the framework.step() so we can record overhead.
                t_fw0 = time.perf_counter()
                record = fw.step(obs)
                t_fw1 = time.perf_counter()
                fw_dt_ms = (t_fw1 - t_fw0) * 1000.0
                framework_step_times_ms.append(fw_dt_ms)

                action = int(record["action"])

                # 3. Step the env, time the wall-clock around it; this is
                #    the simulated "latency" component of the 4-D vector.
                t_env0 = time.perf_counter()
                obs, reward, terminated, truncated, _info = env.step(action)
                t_env1 = time.perf_counter()
                env_dt_ms = (t_env1 - t_env0) * 1000.0

                # 4. Synthetic placeholders for the remaining 3 dims.
                latency_ms = env_dt_ms + fw_dt_ms
                energy_j = 1e-3 * (action + 1)
                energy_remaining = max(0.0, energy_remaining - energy_j)

                # 5. Push these into the framework's records (overwriting
                #    the placeholders the framework wrote based on
                #    pre-step telemetry) and observe the env reward.
                record["latency_ms"] = float(latency_ms)
                record["energy_j"] = float(energy_j)
                record["memory_util"] = float(memory_util)
                fw.observe_reward(float(reward))

                # 6. Update the telemetry source for the NEXT step.
                telemetry.update(
                    latency_ms=latency_ms,
                    energy_remaining_j=energy_remaining,
                    memory_util=memory_util,
                )

                if out_file is not None:
                    serialisable = {
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
                        "framework_step_ms": fw_dt_ms,
                    }
                    out_file.write(json.dumps(serialisable) + "\n")

                ep_return += float(reward)
                episode_step += 1
                total_steps += 1
                done = bool(terminated or truncated)

            episode_returns.append(ep_return)
    finally:
        if out_file is not None:
            out_file.close()
        env.close()

    mean_fw_ms = (
        sum(framework_step_times_ms) / len(framework_step_times_ms)
        if framework_step_times_ms
        else 0.0
    )
    max_fw_ms = max(framework_step_times_ms) if framework_step_times_ms else 0.0

    return {
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
        "out_path": out_path,
    }


def _validate(result: dict) -> tuple[bool, list[str]]:
    """Run the Week 6 acceptance checks against a smoke result dict."""
    failures: list[str] = []

    history = result["history"]
    total_steps = result["total_steps"]

    if not (len(history) > 0 and len(history) == total_steps):
        failures.append(
            f"history length {len(history)} != total_steps {total_steps}"
        )

    required = ("reward", "latency_ms", "energy_j", "memory_util")
    for i, rec in enumerate(history):
        for key in required:
            if key not in rec:
                failures.append(f"step {i}: missing field {key}")
                continue
            v = rec[key]
            if v is None:
                failures.append(f"step {i}: field {key} is None")
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                failures.append(f"step {i}: field {key}={v!r} not numeric")
                continue
            if math.isnan(fv):
                failures.append(f"step {i}: field {key} is NaN")
        if failures and len(failures) > 8:
            break  # cap noise

    if "override_fire_count" not in result:
        failures.append("override fire_count not recorded")

    mean_ms = float(result["mean_framework_step_ms"])
    if not (mean_ms < 5.0):
        failures.append(
            f"mean framework step {mean_ms:.3f} ms exceeds 5 ms budget"
        )

    return (len(failures) == 0), failures


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSONL path; defaults to runs/week6_smoke_<ts>.jsonl",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out_path = args.out
    if out_path is None:
        ts = int(time.time())
        out_path = str(Path("runs") / f"week6_smoke_{ts}.jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result = run_smoke(episodes=args.episodes, out_path=out_path, seed=args.seed)
    runtime_s = time.perf_counter() - t0

    ok, failures = _validate(result)

    jsonl_lines = 0
    if out_path is not None and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for _ in f:
                jsonl_lines += 1

    print(f"episodes              : {args.episodes}")
    print(f"total steps           : {result['total_steps']}")
    print(f"history length        : {len(result['history'])}")
    print(f"override fire_count   : {result['override_fire_count']}")
    print(f"mean fw step (ms)     : {result['mean_framework_step_ms']:.4f}")
    print(f"max  fw step (ms)     : {result['max_framework_step_ms']:.4f}")
    print(f"mean episode return   : {result['mean_episode_return']:.2f}")
    print(f"runtime (s)           : {runtime_s:.2f}")
    print(f"jsonl path            : {out_path}")
    print(f"jsonl lines           : {jsonl_lines}")

    if ok:
        print("SMOKE TEST PASSED")
        return 0
    for fail in failures:
        print(f"FAIL: {fail}")
    print("SMOKE TEST FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
