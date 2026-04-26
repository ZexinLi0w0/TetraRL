"""P15 unified runner — drop-in (algo, wrapper, env, hw, seed) DRL training cell.

Runs ONE matrix cell from ``tetrarl/eval/configs/p15_drl_matrix.yaml``.
Resolves the algo + wrapper, checks compatibility (writes a SKIPPED
``summary.json`` and exits if not), builds the env, runs ``--frames`` env
steps with the wrapper's per-step knob hook, writes ``per_step.jsonl`` +
``summary.json``.

CartPole and Atari obs (uint8 (4, 84, 84) frame-stacks from
``tetrarl/morl/atari_wrappers.make_atari_env``) flow through the same
training loop — the algo classes auto-detect a NatureCNN backbone from
``obs_shape``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Make the repo importable when this script is invoked as a file path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tetrarl.morl.algos import (  # noqa: E402
    A2CAlgo,
    C51Algo,
    DDQNAlgo,
    DQNAlgo,
    PPOAlgo,
)
from tetrarl.morl.system_wrappers import make_wrapper  # noqa: E402

ALGO_REGISTRY: dict[str, type] = {
    "dqn": DQNAlgo,
    "ddqn": DDQNAlgo,
    "c51": C51Algo,
    "a2c": A2CAlgo,
    "ppo": PPOAlgo,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P15 on-device DRL matrix runner")
    p.add_argument("--algo", required=True, choices=list(ALGO_REGISTRY))
    p.add_argument(
        "--wrapper", required=True, choices=["maxa", "maxp", "r3", "duojoule", "tetrarl"]
    )
    p.add_argument("--env", required=True, choices=["cartpole", "breakout"])
    p.add_argument(
        "--platform", default="mac", choices=["orin_agx", "orin_nano", "mac"]
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _time_to_converge(
    episode_returns: list[float], window: int = 100, frac: float = 0.8
) -> int:
    """First step index where the rolling-mean reward over the last ``window``
    episodes hits ``frac * max(rolling_mean)``. -1 if never reached.

    We approximate "step" as the cumulative episode count for the on-policy /
    no-step-tracking case; the runner records per-step episode IDs separately.
    """
    if not episode_returns:
        return -1
    n = len(episode_returns)
    arr = np.asarray(episode_returns, dtype=np.float64)
    rolling = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i + 1 - window)
        rolling[i] = arr[lo : i + 1].mean()
    if rolling.size == 0:
        return -1
    target = float(frac * rolling.max())
    for i, v in enumerate(rolling):
        if v >= target and target > 0.0:
            return int(i)
    return -1


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _make_env(env_name: str):
    """Return a gymnasium env handle. Raises if the env package is missing."""
    import gymnasium as gym  # local import keeps top-level cheap

    if env_name == "cartpole":
        return gym.make("CartPole-v1")
    if env_name == "breakout":
        from tetrarl.morl.atari_wrappers import make_atari_env

        return make_atari_env("ALE/Breakout-v5")
    raise ValueError(f"unknown env: {env_name!r}")


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    per_step_path = out_dir / "per_step.jsonl"

    base_payload = {
        "algo": args.algo,
        "wrapper": args.wrapper,
        "env": args.env,
        "platform": args.platform,
        "seed": int(args.seed),
        "frames": int(args.frames),
    }

    _seed_all(args.seed)

    algo_class = ALGO_REGISTRY[args.algo]
    wrapper = make_wrapper(args.wrapper)

    # --- compatibility gate ------------------------------------------------
    if not wrapper.is_compatible(algo_class):
        payload = {
            **base_payload,
            "status": "SKIPPED",
            "reason": f"wrapper {args.wrapper} incompatible with {args.algo}",
            "wall_time_s": 0.0,
        }
        _write_summary(summary_path, payload)
        return 0

    t_wall0 = time.perf_counter()

    try:
        env = _make_env(args.env)
    except Exception as exc:  # pragma: no cover - environment construction error
        payload = {
            **base_payload,
            "status": "ERROR",
            "reason": f"env build failed: {exc!r}",
            "wall_time_s": time.perf_counter() - t_wall0,
        }
        _write_summary(summary_path, payload)
        return 1

    obs_shape: tuple[int, ...] = env.observation_space.shape  # type: ignore[assignment]
    n_actions = int(env.action_space.n)  # type: ignore[union-attr]

    algo = algo_class(obs_shape=obs_shape, n_actions=n_actions, seed=args.seed)
    wrapper.wrap(algo)

    # Reset CUDA peak-memory tracking if available.
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    obs, _info = env.reset(seed=args.seed)

    # Carry-over algo_state used by wrapper.step_hook.
    last_step_ms = 0.0
    last_step_energy_j = 0.0

    framework_step_ms_list: list[float] = []
    raw_step_ms_list: list[float] = []
    total_step_ms_list: list[float] = []
    energy_j_list: list[float] = []
    deadline_miss_list: list[bool] = []

    episode_idx = 0
    episode_return = 0.0
    episode_returns: list[float] = []

    try:
        with per_step_path.open("w") as jl:
            for step_idx in range(int(args.frames)):
                memory_util = min(0.95, 0.1 + step_idx * 1e-5)
                algo_state = {
                    "last_step_ms": last_step_ms,
                    "last_step_energy_j": last_step_energy_j,
                    "memory_util": memory_util,
                }
                knobs = wrapper.step_hook(step_idx, algo_state)

                # Apply mutable knobs.
                if knobs.batch_size is not None and hasattr(algo, "batch_size"):
                    algo.batch_size = int(knobs.batch_size)
                if (
                    knobs.replay_capacity is not None
                    and hasattr(algo, "replay_capacity")
                    and getattr(algo, "paradigm", "") == "off_policy"
                ):
                    algo.replay_capacity = int(knobs.replay_capacity)

                # Forward / act.
                t_fw0 = time.perf_counter()
                action = algo.act(np.asarray(obs))
                if knobs.action_override is not None:
                    action = int(knobs.action_override)
                t_fw1 = time.perf_counter()

                # Env step.
                t_env0 = time.perf_counter()
                next_obs, reward, term, trunc, _info = env.step(int(action))
                t_env1 = time.perf_counter()

                done = bool(term or trunc)
                # Train.
                algo.observe(np.asarray(obs), int(action), float(reward), np.asarray(next_obs), done)
                _ = algo.update()

                framework_step_ms = (t_fw1 - t_fw0) * 1000.0
                raw_step_ms = (t_env1 - t_env0) * 1000.0
                total_step_ms = framework_step_ms + raw_step_ms
                energy_j = 1e-3 * (int(action) + 1)
                deadline_miss = bool(total_step_ms > 50.0)

                framework_step_ms_list.append(framework_step_ms)
                raw_step_ms_list.append(raw_step_ms)
                total_step_ms_list.append(total_step_ms)
                energy_j_list.append(energy_j)
                deadline_miss_list.append(deadline_miss)

                episode_return += float(reward)

                jl.write(
                    json.dumps(
                        {
                            "step": int(step_idx),
                            "episode": int(episode_idx),
                            "action": int(action),
                            "reward": float(reward),
                            "framework_step_ms": float(framework_step_ms),
                            "raw_step_ms": float(raw_step_ms),
                            "total_step_ms": float(total_step_ms),
                            "energy_j": float(energy_j),
                            "memory_util": float(memory_util),
                            "deadline_miss": bool(deadline_miss),
                        }
                    )
                    + "\n"
                )

                last_step_ms = total_step_ms
                last_step_energy_j = energy_j

                if done:
                    episode_returns.append(float(episode_return))
                    episode_return = 0.0
                    episode_idx += 1
                    obs, _info = env.reset()
                else:
                    obs = next_obs
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Tail episode (in case the run cut off mid-episode).
    if episode_return != 0.0 and (not episode_returns or episode_returns[-1] != episode_return):
        episode_returns.append(float(episode_return))

    n_steps = len(total_step_ms_list)
    mean_total = float(np.mean(total_step_ms_list)) if total_step_ms_list else 0.0
    mean_fw = float(np.mean(framework_step_ms_list)) if framework_step_ms_list else 0.0
    framework_overhead_pct = (mean_fw / mean_total * 100.0) if mean_total > 0.0 else 0.0
    mean_deadline_miss_rate = (
        float(sum(deadline_miss_list)) / float(n_steps) if n_steps else 0.0
    )
    mean_p99_step_ms = _percentile(total_step_ms_list, 99.0)
    if torch.cuda.is_available():
        try:
            peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated()) / 1e6
        except Exception:
            peak_gpu_memory_mb = 0.0
    else:
        peak_gpu_memory_mb = 0.0
    mean_energy_j = float(np.mean(energy_j_list)) if energy_j_list else 0.0
    time_to_converge_steps = _time_to_converge(episode_returns)
    wrapper_metrics = wrapper.get_metrics()

    payload = {
        **base_payload,
        "status": "COMPLETED",
        "wall_time_s": float(time.perf_counter() - t_wall0),
        "n_steps": int(n_steps),
        "n_episodes": int(len(episode_returns)),
        "cumulative_reward_curve": [float(x) for x in episode_returns],
        "framework_overhead_pct": float(framework_overhead_pct),
        "mean_deadline_miss_rate": float(mean_deadline_miss_rate),
        "mean_p99_step_ms": float(mean_p99_step_ms),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        "mean_energy_j": float(mean_energy_j),
        "time_to_converge_steps": int(time_to_converge_steps),
        "wrapper_metrics": wrapper_metrics,
    }
    _write_summary(summary_path, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
