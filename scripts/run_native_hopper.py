"""Train preference-conditioned PPO on MO-Hopper-v4 (mo-gymnasium).

Single-process, GPU-capable training for 3-objective Hopper.

Usage:
    python scripts/run_native_hopper.py [--device cuda] [--steps 1000000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Native preference-conditioned PPO on MO-Hopper"
    )
    parser.add_argument(
        "--device", default="cpu", help="cpu or cuda"
    )
    parser.add_argument(
        "--steps", type=int, default=1_000_000,
        help="Total training timesteps",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-dir",
        default="results/native_hopper",
        help="Directory to save results",
    )
    args = parser.parse_args()

    try:
        import mo_gymnasium  # noqa: F401
    except ImportError:
        print(
            "ERROR: mo-gymnasium not installed. "
            "Install with: pip install mo-gymnasium[mujoco]"
        )
        return 1

    from tetrarl.morl.native.agent import TetraRLNativeAgent

    print("=== TetraRL Native PPO — MO-Hopper-v4 ===\n")
    print(f"Device: {args.device}")
    print(f"Steps:  {args.steps:,}")
    print(f"Seed:   {args.seed}")
    print()

    agent = TetraRLNativeAgent(
        env_name="mo-hopper-v4",
        obj_num=3,
        ref_point=[0.0, 0.0, 0.0],
        total_timesteps=args.steps,
        num_steps=2048,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        seed=args.seed,
        device=args.device,
        eval_interval=10,
        eval_episodes=3,
        n_eval_interior=15,
    )

    results = agent.train(verbose=True)
    front = agent.get_pareto_front()

    print(f"\n{'='*50}")
    print(f"Final HV: {front['hv']:.4f}")
    print(f"|Pareto front|: {len(front['objectives'])}")

    save_path = Path(args.save_dir)
    agent.save(save_path)
    print(f"Model saved to {save_path}")

    np.savetxt(
        save_path / "hv_history.csv",
        np.array(front["hv_history"]),
        delimiter=",",
        header="step,hv,n_pareto",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
