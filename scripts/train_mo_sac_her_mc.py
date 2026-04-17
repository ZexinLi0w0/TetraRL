#!/usr/bin/env python3
"""Train MO-SAC-HER on multi-objective MountainCarContinuous.

Sanity check for the continuous-action PD-MORL port. Two objectives:
position reward (reach goal) vs energy efficiency (minimize |action|^2).

Usage:
    python scripts/train_mo_sac_her_mc.py --frames 100000 --device cuda
"""

import argparse
import json
import os
import tempfile
import time

import numpy as np
import torch

from tetrarl.envs.mo_mountaincar import MOMountainCarContinuous
from tetrarl.eval.hypervolume import hypervolume, pareto_filter
from tetrarl.morl.agents.mo_sac_her import MOSACHERAgent, MOTransition
from tetrarl.morl.preference_sampling import (
    sample_anchor_preferences,
    sample_preference,
)


def evaluate(
    agent: MOSACHERAgent,
    n_objectives: int,
    n_eval: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Evaluate on anchor + sampled preferences, return Pareto front and HV."""
    anchors = sample_anchor_preferences(n_objectives)
    rng = np.random.default_rng(seed)
    extra = sample_preference(n_objectives, max(0, n_eval - len(anchors)), rng=rng)
    eval_prefs = np.concatenate([anchors, extra], axis=0)[:n_eval]

    returns = []
    for omega in eval_prefs:
        env = MOMountainCarContinuous()
        obs, _ = env.reset(seed=seed)
        total_reward = np.zeros(n_objectives, dtype=np.float32)
        done = False
        step = 0
        while not done and step < 999:
            action = agent.act(obs, omega, deterministic=True)
            obs, reward_vec, terminated, truncated, _ = env.step(action)
            total_reward += reward_vec
            done = terminated or truncated
            step += 1
        returns.append(total_reward)
        env.close()

    returns_arr = np.array(returns)
    front = pareto_filter(returns_arr)
    ref_point = MOMountainCarContinuous.REFERENCE_POINT
    hv = hypervolume(front, ref_point)
    return front, hv


def atomic_write_json(path: str, data: object) -> None:
    """Write JSON atomically via temp file + rename."""
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="MO-SAC-HER on MountainCarContinuous")
    parser.add_argument("--frames", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--logdir", type=str, default="runs/mo_sac_her_mc")
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--n-relabel", type=int, default=4)
    parser.add_argument("--dir-reg-coeff", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--initial-alpha", type=float, default=0.2)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.logdir, exist_ok=True)

    env = MOMountainCarContinuous()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    n_objectives = env.reward_dim

    agent = MOSACHERAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        n_objectives=n_objectives,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_size,
        initial_alpha=args.initial_alpha,
        n_relabel=args.n_relabel,
        dir_reg_coeff=args.dir_reg_coeff,
        device=device,
    )

    progress: list[dict] = []
    total_frames = 0
    episode = 0
    best_hv = 0.0
    start_time = time.time()
    rng = np.random.default_rng(args.seed)

    print(
        f"Training MO-SAC-HER on MountainCarContinuous"
        f" | device={device} | frames={args.frames}"
    )
    print(f"Logging to: {args.logdir}")

    while total_frames < args.frames:
        omega = sample_preference(n_objectives, 1, rng=rng)[0]
        obs, _ = env.reset()
        episode_reward = np.zeros(n_objectives, dtype=np.float32)
        done = False
        ep_steps = 0
        last_metrics: dict[str, float] = {}

        while not done and total_frames < args.frames:
            action = agent.act(obs, omega)
            next_obs, reward_vec, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            transition = MOTransition(
                state=obs,
                action=action,
                reward_vec=reward_vec,
                next_state=next_obs,
                done=terminated,
                omega=omega,
            )
            agent.store(transition)
            metrics = agent.update()
            if metrics:
                last_metrics = metrics

            obs = next_obs
            episode_reward += reward_vec
            total_frames += 1
            ep_steps += 1

            if total_frames % args.eval_freq == 0:
                front, hv = evaluate(agent, n_objectives)
                elapsed = time.time() - start_time
                fps = total_frames / max(elapsed, 1e-6)
                record = {
                    "frames": total_frames,
                    "hv": float(hv),
                    "n_pareto": len(front),
                    "episode": episode,
                    "elapsed_s": round(elapsed, 1),
                    "fps": round(fps, 1),
                    "alpha": last_metrics.get("alpha"),
                    "critic_loss": last_metrics.get("critic_loss"),
                    "actor_loss": last_metrics.get("actor_loss"),
                    "dir_reg": last_metrics.get("dir_reg"),
                }
                progress.append(record)
                atomic_write_json(
                    os.path.join(args.logdir, "progress.json"), progress
                )
                if hv > best_hv:
                    best_hv = hv
                    agent.save(os.path.join(args.logdir, "best_model.pt"))
                print(
                    f"[{total_frames:>7d}/{args.frames}] "
                    f"HV={hv:.2f} (best={best_hv:.2f}) "
                    f"|PF|={len(front)} "
                    f"alpha={last_metrics.get('alpha', 0):.3f} "
                    f"fps={fps:.0f} "
                    f"t={elapsed:.0f}s"
                )

                nan_detected = any(
                    v is not None and (isinstance(v, float) and (v != v))
                    for v in last_metrics.values()
                )
                if nan_detected:
                    print("WARNING: NaN detected in losses, stopping early!")
                    agent.save(os.path.join(args.logdir, "nan_checkpoint.pt"))
                    record["nan_detected"] = True
                    atomic_write_json(
                        os.path.join(args.logdir, "progress.json"), progress
                    )
                    return

        episode += 1

    agent.save(os.path.join(args.logdir, "final_model.pt"))
    atomic_write_json(os.path.join(args.logdir, "progress.json"), progress)

    _, final_hv = evaluate(agent, n_objectives)
    print(f"\nTraining complete. Final HV={final_hv:.2f} (best={best_hv:.2f})")
    print(f"Results saved to: {args.logdir}")


if __name__ == "__main__":
    main()
