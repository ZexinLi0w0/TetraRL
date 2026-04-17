#!/usr/bin/env python3
"""Evaluate a saved PD-MORL checkpoint on Deep Sea Treasure.

Loads a best_model.pt checkpoint, runs evaluation episodes across
uniformly-spaced preference anchors on the 2-D simplex, computes
the achieved Pareto front and its hypervolume indicator.

Usage:
    python scripts/eval_pd_morl_dst.py \
        --model-path runs/week1_orin_validation/best_model.pt \
        --device cuda --out-json eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from tetrarl.envs.dst import DeepSeaTreasure
from tetrarl.eval.hypervolume import hypervolume, pareto_filter
from tetrarl.morl.agents.pd_morl import PDMORLAgent

DST_REFERENCE_POINT = np.array([0.0, -25.0])
DST_REFERENCE_HV = 229.0
DST_PARETO_OPTIMAL = [
    (1, -1), (2, -3), (3, -5), (5, -7), (8, -8),
    (16, -9), (24, -13), (50, -14), (74, -17), (124, -19),
]


def make_anchors_2d(n: int) -> np.ndarray:
    """Generate n uniformly-spaced preference vectors on the 2-D simplex."""
    if n == 1:
        return np.array([[0.5, 0.5]], dtype=np.float32)
    ws = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.column_stack([ws, 1.0 - ws])


def evaluate_policy(
    agent: PDMORLAgent,
    anchors: np.ndarray,
    episodes_per_anchor: int,
    seed: int = 42,
) -> np.ndarray:
    """Run evaluation episodes and return all achieved return vectors."""
    all_returns = []
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
    return np.array(all_returns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PD-MORL checkpoint on DST"
    )
    parser.add_argument("--model-path", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-anchors", type=int, default=11)
    parser.add_argument("--episodes-per-anchor", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out-json", default="eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    env = DeepSeaTreasure()
    state_dim = 2
    action_dim = 4
    n_objectives = env.reward_dim
    env.close()

    agent = PDMORLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        n_objectives=n_objectives,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    print(f"Loading checkpoint: {args.model_path}")
    agent.load(args.model_path)
    print(f"  step_count = {agent.step_count}")

    anchors = make_anchors_2d(args.n_anchors)
    n, e = args.n_anchors, args.episodes_per_anchor
    print(f"Evaluating on {n} anchors x {e} episodes ...")

    t0 = time.time()
    returns = evaluate_policy(agent, anchors, args.episodes_per_anchor, args.seed)
    elapsed = time.time() - t0

    front = pareto_filter(returns)
    hv = hypervolume(front, DST_REFERENCE_POINT)
    gap_pct = (1.0 - hv / DST_REFERENCE_HV) * 100.0

    results = {
        "achieved_hv": round(float(hv), 4),
        "reference_hv": DST_REFERENCE_HV,
        "gap_pct": round(float(gap_pct), 2),
        "pareto_front": front.tolist(),
        "all_returns": returns.tolist(),
        "anchors": anchors.tolist(),
        "reference_point": DST_REFERENCE_POINT.tolist(),
        "model_path": args.model_path,
        "n_anchors": args.n_anchors,
        "episodes_per_anchor": args.episodes_per_anchor,
        "step_count": agent.step_count,
        "device": device,
        "eval_time_s": round(elapsed, 2),
    }

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.out_json}")

    print("\n" + "=" * 50)
    print("  PD-MORL DST Evaluation Summary")
    print("=" * 50)
    print(f"  Achieved HV : {hv:.4f}")
    print(f"  Reference HV: {DST_REFERENCE_HV}")
    print(f"  Gap          : {gap_pct:.2f}%")
    print(f"  |Pareto front|: {len(front)}")
    print("  Pareto points:")
    for pt in sorted(front.tolist(), key=lambda x: x[0]):
        print(f"    treasure={pt[0]:>6.1f}  time_penalty={pt[1]:>6.1f}")
    print(f"  Eval time    : {elapsed:.1f}s")
    print("=" * 50)

    optimal = np.array(DST_PARETO_OPTIMAL)
    n_hit = 0
    for opt_pt in optimal:
        for fpt in front:
            if abs(fpt[0] - opt_pt[0]) < 0.5 and abs(fpt[1] - opt_pt[1]) < 0.5:
                n_hit += 1
                break
    print(f"  Optimal coverage: {n_hit}/{len(optimal)} Pareto-optimal points hit")

    sys.exit(0)


if __name__ == "__main__":
    main()
