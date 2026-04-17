#!/usr/bin/env python3
"""Plot achieved vs. optimal Pareto front for DST evaluation.

Reads eval_results.json produced by eval_pd_morl_dst.py and generates
a scatter plot comparing the achieved Pareto front against the known
optimal DST Pareto front.

Usage:
    python scripts/plot_dst_pareto.py \
        --eval-json results/week1_orin_validation/eval_results.json \
        --out docs/figures/week1_dst_pareto_eval.png
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DST_PARETO_OPTIMAL = np.array([
    [1, -1], [2, -3], [3, -5], [5, -7], [8, -8],
    [16, -9], [24, -13], [50, -14], [74, -17], [124, -19],
], dtype=np.float64)


def plot_dst_pareto(
    eval_json: str,
    out: str | Path,
    dpi: int = 150,
) -> Path:
    """Generate Pareto front comparison plot."""
    with open(eval_json) as f:
        data = json.load(f)

    achieved = np.array(data["pareto_front"])
    all_returns = np.array(data["all_returns"])
    hv = data["achieved_hv"]
    gap = data["gap_pct"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        DST_PARETO_OPTIMAL[:, 0],
        DST_PARETO_OPTIMAL[:, 1],
        "o--",
        color="#999999",
        markerfacecolor="none",
        markeredgecolor="#999999",
        markersize=8,
        linewidth=1.5,
        label="DST optimal Pareto front",
        zorder=2,
    )

    if len(all_returns) > len(achieved):
        ax.scatter(
            all_returns[:, 0],
            all_returns[:, 1],
            c="#aec7e8",
            s=30,
            alpha=0.4,
            label="All eval returns",
            zorder=1,
        )

    ax.scatter(
        achieved[:, 0],
        achieved[:, 1],
        c="#1f77b4",
        s=80,
        marker="D",
        edgecolors="black",
        linewidth=0.8,
        label=f"Achieved Pareto front (HV={hv:.1f})",
        zorder=3,
    )

    ax.set_xlabel("Treasure Value", fontsize=12)
    ax.set_ylabel("Time Penalty", fontsize=12)
    ax.set_title(
        f"PD-MORL DST Evaluation — HV={hv:.1f} (gap={gap:.1f}%)",
        fontsize=13,
    )
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = Path(out)
    os.makedirs(out_path.parent, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pareto front plot to: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DST Pareto front")
    parser.add_argument("--eval-json", required=True)
    parser.add_argument(
        "--out", default="docs/figures/week1_dst_pareto_eval.png"
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    plot_dst_pareto(args.eval_json, args.out, args.dpi)


if __name__ == "__main__":
    main()
