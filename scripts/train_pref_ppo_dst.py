#!/usr/bin/env python3
"""Train TetraRL preference-conditioned PPO on Deep Sea Treasure.

Trains a single-process preference-conditioned PPO agent and emits a final
evaluation matching the schema produced by ``scripts/eval_pd_morl_dst.py``,
so that downstream aggregators can compare baselines apples-to-apples.

Usage:
    python scripts/train_pref_ppo_dst.py --frames 200000 --seed 0 \
        --device auto --logdir runs/p11_dst_headtohead/pref_ppo_seed0
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
from tetrarl.morl.native.preference_ppo import (
    PreferencePPOConfig,
    evaluate_policy,
    train_preference_ppo,
)

DST_REFERENCE_POINT = np.array([0.0, -25.0])
DST_REFERENCE_HV = 229.0


def make_anchors_2d(n: int) -> np.ndarray:
    """Generate n uniformly-spaced preference vectors on the 2-D simplex."""
    if n == 1:
        return np.array([[0.5, 0.5]], dtype=np.float32)
    ws = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.column_stack([ws, 1.0 - ws])


def run_final_eval(
    network: torch.nn.Module,
    n_anchors: int,
    episodes_per_anchor: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run the locked head-to-head eval protocol.

    Returns (anchors, all_returns, achieved_hv, elapsed_s).
    """
    anchors = make_anchors_2d(n_anchors)
    all_returns: list[np.ndarray] = []
    t0 = time.time()
    eval_env = DeepSeaTreasure()
    try:
        for omega in anchors:
            for _ in range(episodes_per_anchor):
                # evaluate_policy averages over n_episodes; call with 1 to
                # get per-episode samples and aggregate them ourselves.
                ret = evaluate_policy(
                    network,
                    eval_env,
                    omega,
                    n_episodes=1,
                    device=device,
                    deterministic=True,
                )
                all_returns.append(np.asarray(ret, dtype=np.float64))
    finally:
        eval_env.close()
    elapsed = time.time() - t0
    returns_arr = np.array(all_returns)
    front = pareto_filter(returns_arr)
    hv = hypervolume(front, DST_REFERENCE_POINT)
    return anchors, returns_arr, float(hv), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TetraRL preference-PPO on Deep Sea Treasure"
    )
    parser.add_argument("--frames", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--logdir", type=str, default="runs/pref_ppo_dst")
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--n-anchors", type=int, default=11)
    parser.add_argument("--episodes-per-anchor", type=int, default=5)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.logdir, exist_ok=True)

    config = PreferencePPOConfig(
        n_objectives=2,
        total_timesteps=args.frames,
        num_steps=args.num_steps,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        eval_interval=args.eval_interval,
        ref_point=[float(DST_REFERENCE_POINT[0]), float(DST_REFERENCE_POINT[1])],
        seed=args.seed,
    )

    def env_fn() -> DeepSeaTreasure:
        return DeepSeaTreasure()

    print(
        f"Training TetraRL pref-PPO on DST | device={device} "
        f"| frames={args.frames} | seed={args.seed}"
    )
    print(f"Logging to: {args.logdir}")

    train_start = time.time()
    result = train_preference_ppo(
        config, env_fn, device=device, verbose=True
    )
    train_elapsed = time.time() - train_start

    network = result["network"]
    hv_history = result["hv_history"]
    best_hv = float(result["best_hv"])
    global_step = int(result["global_step"])

    progress = []
    for record in hv_history:
        # hv_history entries are (global_step, hv, n_pareto) tuples.
        gs, hv_val, n_pf = record[0], record[1], record[2]
        progress.append(
            {
                "frames": int(gs),
                "hv": float(hv_val),
                "n_pareto": int(n_pf),
            }
        )

    progress_path = os.path.join(args.logdir, "progress.json")
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    best_model_path = os.path.join(args.logdir, "best_model.pt")
    torch.save(network.state_dict(), best_model_path)

    print(
        f"\nTraining complete in {train_elapsed:.1f}s | "
        f"best (in-loop) HV={best_hv:.2f} | global_step={global_step}"
    )

    print(
        f"Running final eval: {args.n_anchors} anchors x "
        f"{args.episodes_per_anchor} episodes ..."
    )
    anchors, all_returns, achieved_hv, eval_elapsed = run_final_eval(
        network, args.n_anchors, args.episodes_per_anchor, device
    )
    front = pareto_filter(all_returns)
    gap_pct = (1.0 - achieved_hv / DST_REFERENCE_HV) * 100.0

    results = {
        "baseline": "tetrarl_pref_ppo",
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
        "step_count": global_step,
        "device": device,
        "eval_time_s": round(eval_elapsed, 2),
    }

    eval_path = os.path.join(args.logdir, "eval.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("  TetraRL pref-PPO DST Final Eval")
    print("=" * 50)
    print(f"  Achieved HV : {achieved_hv:.4f}")
    print(f"  Reference HV: {DST_REFERENCE_HV}")
    print(f"  Gap          : {gap_pct:.2f}%")
    print(f"  |Pareto front|: {len(front)}")
    print(f"  Eval time    : {eval_elapsed:.1f}s")
    print(f"  Outputs in   : {args.logdir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
