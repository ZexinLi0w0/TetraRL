#!/usr/bin/env python3
"""Train PD-MORL (MO-DDQN-HER) on Deep Sea Treasure.

Reproduction target: HV >= 229 within 200k environment steps,
per Basaklar et al. (ICLR 2023) Table 1.

Usage:
    python scripts/train_pd_morl_dst.py --frames 200000 --seed 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from tetrarl.envs.dst import DeepSeaTreasure
from tetrarl.eval.hypervolume import hypervolume, pareto_filter
from tetrarl.morl.agents.pd_morl import PDMORLAgent, Transition
from tetrarl.morl.preference_sampling import (
    sample_anchor_preferences,
    sample_preference,
)


def evaluate(
    agent: PDMORLAgent, env_cls: type, num_objectives: int, seed: int = 42
) -> tuple[np.ndarray, float]:
    """Evaluate the agent on anchor preferences and compute HV.

    Returns the discovered Pareto front and its hypervolume.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="PD-MORL on Deep Sea Treasure")
    parser.add_argument("--frames", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--logdir", type=str, default="runs/pd_morl_dst")
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    parser.add_argument("--n-relabel", type=int, default=4)
    parser.add_argument("--epsilon-decay", type=int, default=50_000)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    random_state = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.logdir, exist_ok=True)

    env = DeepSeaTreasure()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    n_objectives = env.reward_dim

    agent = PDMORLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        n_objectives=n_objectives,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        buffer_capacity=args.buffer_size,
        n_relabel=args.n_relabel,
        epsilon_decay=args.epsilon_decay,
        device=device,
    )

    progress = []
    total_frames = 0
    episode = 0
    best_hv = 0.0
    start_time = time.time()

    print(f"Training PD-MORL on DST | device={device} | frames={args.frames}")
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
                    "epsilon": agent._epsilon(),
                    "loss": metrics.get("loss", None),
                }
                progress.append(record)
                if hv > best_hv:
                    best_hv = hv
                    agent.save(os.path.join(args.logdir, "best_model.pt"))
                print(
                    f"[{total_frames:>7d}/{args.frames}] "
                    f"HV={hv:.2f} (best={best_hv:.2f}) "
                    f"|PF|={len(front)} "
                    f"eps={agent._epsilon():.3f} "
                    f"t={elapsed:.0f}s"
                )

        episode += 1

    agent.save(os.path.join(args.logdir, "final_model.pt"))
    with open(os.path.join(args.logdir, "progress.json"), "w") as f:
        json.dump(progress, f, indent=2)

    _, final_hv = evaluate(agent, DeepSeaTreasure, n_objectives)
    print(f"\nTraining complete. Final HV={final_hv:.2f} (best={best_hv:.2f})")
    print(f"Results saved to: {args.logdir}")


if __name__ == "__main__":
    main()
