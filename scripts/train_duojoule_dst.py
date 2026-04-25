#!/usr/bin/env python3
"""Training driver for DuoJoule (re-implementation per Yan et al. RTSS'24;
original closed-source). Wraps PD-MORL with greedy (B, R) controller
switched at episode boundaries.

Emits the same eval-result JSON schema as ``scripts/eval_pd_morl_dst.py``
plus a ``baseline`` field set to ``"duojoule"`` and a snapshot of the
controller's final (B, R) state, so downstream aggregators can compare
baselines apples-to-apples in the head-to-head DST sweep.

Usage:
    python scripts/train_duojoule_dst.py --frames 200000 --seed 0 \
        --device auto --logdir runs/p11_dst_headtohead/duojoule_seed0
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from tetrarl.envs.dst import DeepSeaTreasure
from tetrarl.eval.hypervolume import hypervolume, pareto_filter
from tetrarl.morl.agents.pd_morl import Transition
from tetrarl.morl.baselines.duojoule import DuoJouleAgent
from tetrarl.morl.preference_sampling import (
    sample_anchor_preferences,
    sample_preference,
)

DST_REFERENCE_POINT = np.array([0.0, -25.0])
DST_REFERENCE_HV = 229.0


def make_anchors_2d(n: int) -> np.ndarray:
    """Generate n uniformly-spaced preference vectors on the 2-D simplex."""
    if n == 1:
        return np.array([[0.5, 0.5]], dtype=np.float32)
    ws = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.column_stack([ws, 1.0 - ws])


def evaluate(
    agent: DuoJouleAgent, env_cls: type, num_objectives: int, seed: int = 42
) -> tuple[np.ndarray, float]:
    """Evaluate the agent on anchor preferences and compute HV.

    Mirrors ``scripts/train_pd_morl_dst.py::evaluate`` so periodic in-loop
    HV measurements stay comparable across baselines.
    """
    anchors = sample_anchor_preferences(num_objectives)
    n_eval = 11
    extra = sample_preference(num_objectives, max(0, n_eval - len(anchors)),
                              rng=np.random.default_rng(seed))
    if len(anchors) < n_eval:
        eval_prefs = np.concatenate([anchors, extra], axis=0)
    else:
        eval_prefs = anchors[:n_eval]

    returns = []
    for omega in eval_prefs:
        env = env_cls()
        obs, _ = env.reset(seed=seed)
        total_reward = np.zeros(num_objectives, dtype=np.float32)
        done = False
        step = 0
        while not done and step < 200:
            action = agent.act(obs, omega, explore=False)
            obs, reward_vec, terminated, truncated, _ = env.step(action)
            total_reward += reward_vec
            done = terminated or truncated
            step += 1
        returns.append(total_reward)
        env.close()

    returns = np.array(returns)
    front = pareto_filter(returns)
    ref_point = np.array([0.0, -25.0])
    hv = hypervolume(front, ref_point)
    return front, hv


def run_final_eval(
    agent: DuoJouleAgent,
    n_anchors: int,
    episodes_per_anchor: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run the locked head-to-head eval protocol.

    Returns ``(anchors, all_returns, achieved_hv, elapsed_s)``.
    """
    anchors = make_anchors_2d(n_anchors)
    all_returns: list[np.ndarray] = []
    t0 = time.time()
    for omega in anchors:
        for ep in range(episodes_per_anchor):
            env = DeepSeaTreasure()
            obs, _ = env.reset(seed=seed + ep)
            total_reward = np.zeros(2, dtype=np.float32)
            done = False
            step = 0
            while not done and step < 200:
                action = agent.act(obs, omega, explore=False)
                obs, reward_vec, terminated, truncated, _ = env.step(action)
                total_reward += reward_vec
                done = terminated or truncated
                step += 1
            all_returns.append(total_reward)
            env.close()
    elapsed = time.time() - t0
    returns_arr = np.array(all_returns)
    front = pareto_filter(returns_arr)
    hv = hypervolume(front, DST_REFERENCE_POINT)
    return anchors, returns_arr, float(hv), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="DuoJoule on Deep Sea Treasure")
    parser.add_argument("--frames", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--logdir", type=str, default="runs/duojoule_dst")
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial-batch-idx", type=int, default=1,
                        help="index into (32, 64, 128, 256)")
    parser.add_argument("--initial-replay-idx", type=int, default=0,
                        help="index into (1, 2, 4)")
    parser.add_argument("--w-energy", type=float, default=0.5)
    parser.add_argument("--w-latency", type=float, default=0.5)
    parser.add_argument("--n-anchors", type=int, default=11)
    parser.add_argument("--episodes-per-anchor", type=int, default=5)
    parser.add_argument("--n-relabel", type=int, default=4)
    parser.add_argument("--epsilon-decay", type=int, default=50_000)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    import random as _py_random

    random_state = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _py_random.seed(args.seed)

    os.makedirs(args.logdir, exist_ok=True)

    env = DeepSeaTreasure()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    n_objectives = env.reward_dim

    agent = DuoJouleAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        n_objectives=n_objectives,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        device=device,
        seed=args.seed,
        initial_batch_idx=args.initial_batch_idx,
        initial_replay_idx=args.initial_replay_idx,
        w_energy=args.w_energy,
        w_latency=args.w_latency,
        # Forwarded to the underlying PDMORLAgent via **base_kwargs:
        n_relabel=args.n_relabel,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        buffer_capacity=args.buffer_size,
    )

    progress: list[dict] = []
    controller_log: list[dict] = []
    total_frames = 0
    episode = 0
    best_hv = 0.0
    start_time = time.time()

    print(
        f"Training DuoJoule on DST | device={device} | frames={args.frames} "
        f"| seed={args.seed}"
    )
    print(
        f"Initial (B, R) = ({agent.current_batch_size}, "
        f"{agent.current_replay_ratio})"
    )
    print(f"Logging to: {args.logdir}")

    while total_frames < args.frames:
        omega = sample_preference(n_objectives, 1, rng=random_state)[0]
        obs, _ = env.reset()
        episode_reward = np.zeros(n_objectives, dtype=np.float32)
        done = False

        while not done and total_frames < args.frames:
            action = agent.act(obs, omega, explore=True)
            next_obs, reward_vec, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            transition = Transition(
                state=obs,
                action=action,
                reward_vec=reward_vec,
                next_state=next_obs,
                done=terminated,
                omega=omega,
            )
            agent.store(transition)
            metrics = agent.update()

            obs = next_obs
            episode_reward += reward_vec
            total_frames += 1

            if total_frames % args.eval_freq == 0:
                front, hv = evaluate(agent, DeepSeaTreasure, n_objectives)
                elapsed = time.time() - start_time
                record = {
                    "frames": total_frames,
                    "hv": float(hv),
                    "n_pareto": len(front),
                    "episode": episode,
                    "elapsed_s": round(elapsed, 1),
                    "epsilon": agent.base._epsilon(),
                    "loss": metrics.get("loss", None),
                    "batch_size": int(agent.current_batch_size),
                    "replay_ratio": int(agent.current_replay_ratio),
                }
                progress.append(record)
                if hv > best_hv:
                    best_hv = hv
                    agent.save(os.path.join(args.logdir, "best_model.pt"))
                print(
                    f"[{total_frames:>7d}/{args.frames}] "
                    f"HV={hv:.2f} (best={best_hv:.2f}) "
                    f"|PF|={len(front)} "
                    f"eps={agent.base._epsilon():.3f} "
                    f"B={agent.current_batch_size} "
                    f"R={agent.current_replay_ratio} "
                    f"t={elapsed:.0f}s"
                )

        # Episode finished -- run the greedy (B, R) controller, then log.
        agent.end_episode()
        controller_log.append(
            {
                "episode": episode,
                "frames": total_frames,
                "batch_size": int(agent.current_batch_size),
                "replay_ratio": int(agent.current_replay_ratio),
                "last_action": agent._last_action,
            }
        )

        episode += 1

    agent.save(os.path.join(args.logdir, "final_model.pt"))

    progress_payload = {
        "records": progress,
        "controller_log": controller_log,
    }
    with open(os.path.join(args.logdir, "progress.json"), "w") as f:
        json.dump(progress_payload, f, indent=2)

    print(
        f"\nTraining complete. best (in-loop) HV={best_hv:.2f} "
        f"| episodes={episode} | controller_log entries={len(controller_log)}"
    )

    print(
        f"Running final eval: {args.n_anchors} anchors x "
        f"{args.episodes_per_anchor} episodes ..."
    )
    anchors, all_returns, achieved_hv, eval_elapsed = run_final_eval(
        agent, args.n_anchors, args.episodes_per_anchor
    )
    front = pareto_filter(all_returns)
    gap_pct = (1.0 - achieved_hv / DST_REFERENCE_HV) * 100.0

    best_model_path = os.path.join(args.logdir, "best_model.pt")
    if not os.path.exists(best_model_path):
        # No periodic eval ever beat HV=0 (very short smoke runs); fall back
        # to the final checkpoint so the path field still resolves.
        best_model_path = os.path.join(args.logdir, "final_model.pt")

    results = {
        "baseline": "duojoule",
        "achieved_hv": round(float(achieved_hv), 4),
        "reference_hv": DST_REFERENCE_HV,
        "gap_pct": round(float(gap_pct), 2),
        "pareto_front": front.tolist(),
        "all_returns": all_returns.tolist(),
        "anchors": anchors.tolist(),
        "reference_point": DST_REFERENCE_POINT.tolist(),
        "model_path": best_model_path,
        "n_anchors": args.n_anchors,
        "episodes_per_anchor": args.episodes_per_anchor,
        "step_count": int(agent.step_count),
        "device": device,
        "eval_time_s": round(eval_elapsed, 2),
        "controller_final_state": {
            "batch_size": int(agent.current_batch_size),
            "replay_ratio": int(agent.current_replay_ratio),
        },
    }

    eval_path = os.path.join(args.logdir, "eval.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("  DuoJoule DST Final Eval")
    print("=" * 50)
    print(f"  Achieved HV : {achieved_hv:.4f}")
    print(f"  Reference HV: {DST_REFERENCE_HV}")
    print(f"  Gap          : {gap_pct:.2f}%")
    print(f"  |Pareto front|: {len(front)}")
    print(
        f"  Final (B, R) : ({agent.current_batch_size}, "
        f"{agent.current_replay_ratio})"
    )
    print(f"  Eval time    : {eval_elapsed:.1f}s")
    print(f"  Outputs in   : {args.logdir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
