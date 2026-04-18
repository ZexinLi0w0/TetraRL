#!/usr/bin/env python3
"""
Analyze and compare v6 vs v7 Building-3d init-only training runs.

Loads progress_task{0-5}.json for both versions, computes convergence
metrics, and produces three publication-quality figures.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "/Users/zexinli/Downloads/TetraRL"
V6_DIR = os.path.join(BASE, "results", "week2_building3d_v6_init_only")
V7_DIR = os.path.join(BASE, "results", "week2_building3d_v7_init_only")
FIG_DIR = os.path.join(BASE, "docs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TASKS = list(range(6))
VERSIONS = {"v6": V6_DIR, "v7": V7_DIR}
TARGET_ITER = 488

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_version(version_dir):
    """Return dict  task_id -> list[dict]  sorted by iter."""
    data = {}
    for t in TASKS:
        path = os.path.join(version_dir, f"progress_task{t}.json")
        with open(path) as f:
            records = json.load(f)
        records.sort(key=lambda r: r["iter"])
        data[t] = records
    return data


def extract_series(records, field):
    """Return (iters, values) numpy arrays for a field."""
    iters = np.array([r["iter"] for r in records])
    vals = np.array([r[field] for r in records])
    return iters, vals


def convergence_iter(iters, values, tol=0.10):
    """
    Return the first iteration at which value_loss stabilises within
    `tol` fraction of its final value for all remaining iterations.
    """
    final = values[-1]
    if final == 0:
        return iters[0]
    for i in range(len(values)):
        remaining = values[i:]
        if np.all(np.abs(remaining - final) <= tol * np.abs(final)):
            return int(iters[i])
    return int(iters[-1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_data = {v: load_version(d) for v, d in VERSIONS.items()}

    # ---- Compute summary metrics ------------------------------------------
    summary_rows = []
    for ver in ("v6", "v7"):
        for t in TASKS:
            recs = all_data[ver][t]
            final = recs[-1]
            iters_vl, vals_vl = extract_series(recs, "value_loss")
            conv = convergence_iter(iters_vl, vals_vl)
            summary_rows.append({
                "version": ver,
                "task": t,
                "final_iter": final["iter"],
                "final_value_loss": final["value_loss"],
                "final_action_loss": final["action_loss"],
                "final_dist_entropy": final["dist_entropy"],
                "convergence_iter": conv,
            })

    # ---- Print summary table ----------------------------------------------
    hdr = (
        f"{'Ver':<4} {'Task':<5} {'FinalIter':>9} "
        f"{'ValLoss':>10} {'ActLoss':>10} {'Entropy':>10} {'ConvIter':>9}"
    )
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in summary_rows:
        print(
            f"{r['version']:<4} {r['task']:<5} {r['final_iter']:>9} "
            f"{r['final_value_loss']:>10.6f} {r['final_action_loss']:>10.6f} "
            f"{r['final_dist_entropy']:>10.4f} {r['convergence_iter']:>9}"
        )
    print("=" * len(hdr))

    # ---- Style helpers ----------------------------------------------------
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:6]

    # ====================================================================
    # Plot 1: Convergence curves  (3 rows x 2 cols)
    # ====================================================================
    fields = ["value_loss", "action_loss", "dist_entropy"]
    field_labels = ["Value Loss", "Action Loss", "Dist Entropy"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for col, ver in enumerate(("v6", "v7")):
        for row, (field, label) in enumerate(zip(fields, field_labels)):
            ax = axes[row, col]
            for t in TASKS:
                iters, vals = extract_series(all_data[ver][t], field)
                ax.plot(iters, vals, label=f"Task {t}", color=colors[t],
                        linewidth=1.2)
            ax.set_ylabel(label, fontsize=10)
            ax.set_title(f"{ver.upper()} - {label}", fontsize=11)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            if row == 0:
                ax.legend(fontsize=7, ncol=3, loc="upper right")
            if row == 2:
                ax.set_xlabel("Iteration", fontsize=10)

    fig.suptitle("V6 vs V7 Training Convergence", fontsize=13, y=1.01)
    fig.tight_layout()
    path1 = os.path.join(FIG_DIR, "week2_v6v7_convergence.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path1}")

    # ====================================================================
    # Plot 2: Completion bar chart
    # ====================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35
    x = np.arange(len(TASKS))
    for i, ver in enumerate(("v6", "v7")):
        final_iters = [all_data[ver][t][-1]["iter"] for t in TASKS]
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, final_iters, bar_width, label=ver.upper(),
                      color=colors[i], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, final_iters):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    str(val), ha="center", va="bottom", fontsize=8)

    ax.axhline(y=TARGET_ITER, color="red", linestyle="--", linewidth=1.2,
               label=f"Target ({TARGET_ITER})")
    ax.set_xlabel("Task ID", fontsize=11)
    ax.set_ylabel("Final Iteration Reached", fontsize=11)
    ax.set_title("Training Completion per Task", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Task {t}" for t in TASKS])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    path2 = os.path.join(FIG_DIR, "week2_v6v7_completion_table.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path2}")

    # ====================================================================
    # Plot 3: Throughput (steps/sec) over iterations
    # ====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for col, ver in enumerate(("v6", "v7")):
        ax = axes[col]
        for t in TASKS:
            recs = all_data[ver][t]
            iters_arr = np.array([r["iter"] for r in recs])
            steps_arr = np.array([r["steps_done"] for r in recs])
            time_arr = np.array([r["elapsed_sec"] for r in recs])
            # Compute deltas between consecutive records
            d_steps = np.diff(steps_arr)
            d_time = np.diff(time_arr)
            # Avoid division by zero
            mask = d_time > 0
            throughput = np.full_like(d_steps, dtype=float, fill_value=np.nan)
            throughput[mask] = d_steps[mask] / d_time[mask]
            ax.plot(iters_arr[1:], throughput, label=f"Task {t}",
                    color=colors[t], linewidth=1.0, alpha=0.85)

        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_title(f"{ver.upper()} - Throughput", fontsize=11)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=7, ncol=3, loc="upper right")

    axes[0].set_ylabel("Steps / sec", fontsize=10)
    fig.suptitle("V6 vs V7 Training Throughput", fontsize=13, y=1.01)
    fig.tight_layout()
    path3 = os.path.join(FIG_DIR, "week2_v6v7_overhead.png")
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path3}")


if __name__ == "__main__":
    main()
