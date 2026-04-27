#!/usr/bin/env python3
"""Generate 9 publication-quality figures (3 per phase x 3 phases) for P15 paper §VII.

Per phase:
  Figure 1 p15_phase{N}_cumulative_reward.pdf  - 5-panel (one per algo) of smoothed
                                                  cumulative episode reward, 5 lines per
                                                  panel (one per wrapper), shaded +/-std.
  Figure 2 p15_phase{N}_latency_cdf.pdf        - ECDF of mean_p99_step_ms per wrapper,
                                                  pooled across (algo, seed).
  Figure 3 p15_phase{N}_r3_3panel.pdf          - 3-panel for DQN: ECDF of p99 step,
                                                  mean energy/step bar chart, final
                                                  reward bar chart.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = ROOT / "tetrarl" / "eval" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

PHASE3_RUNS = Path("/Users/zexinli/Downloads/TetraRL-p15-phase3/runs/p15_phase3_orin_agx_atari")

PHASES = [
    {
        "n": 1,
        "label": "CartPole x Orin AGX",
        "runs": ROOT / "runs" / "p15_phase1_orin_agx_cartpole",
        "smooth_window": 50,
    },
    {
        "n": 2,
        "label": "CartPole x Orin Nano",
        "runs": ROOT / "runs" / "p15_phase2_orin_nano_cartpole",
        "smooth_window": 50,
    },
    {
        "n": 3,
        "label": "Atari Breakout x Orin AGX",
        "runs": PHASE3_RUNS,
        "smooth_window": 200,
    },
]

ALGOS = ["dqn", "ddqn", "c51", "a2c", "ppo"]
WRAPPERS = ["maxa", "maxp", "r3", "duojoule", "tetrarl"]
ALGO_DISPLAY = {"dqn": "DQN", "ddqn": "DDQN", "c51": "C51", "a2c": "A2C", "ppo": "PPO"}
WRAPPER_DISPLAY = {
    "maxa": "MAX-A",
    "maxp": "MAX-P",
    "r3": r"R$^3$",
    "duojoule": "DuoJoule",
    "tetrarl": "TetraRL",
}
WRAPPER_COLOR = {
    "maxa": "#1f77b4",
    "maxp": "#ff7f0e",
    "r3": "#2ca02c",
    "duojoule": "#d62728",
    "tetrarl": "#9467bd",
}


def load_phase(phase):
    """Return cells dict keyed by (algo, wrapper) -> {"completed": [d, ...], "skipped": bool}."""
    runs_dir = phase["runs"]
    cells = {}
    if not runs_dir.is_dir():
        print(f"WARN: phase {phase['n']} runs dir missing: {runs_dir}")
        return cells
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        sj = sub / "summary.json"
        if not sj.is_file():
            continue
        try:
            d = json.loads(sj.read_text())
        except Exception as e:
            print(f"WARN: parse {sj}: {e}")
            continue
        algo = d.get("algo", "")
        wrapper = d.get("wrapper", "")
        key = (algo, wrapper)
        bucket = cells.setdefault(key, {"completed": [], "skipped": False})
        status = d.get("status", "")
        if status == "COMPLETED":
            bucket["completed"].append(d)
        elif status == "SKIPPED":
            bucket["skipped"] = True
    return cells


def rolling_mean(arr, window):
    """Centered rolling mean using np.convolve; window auto-clipped to len(arr)."""
    if arr.size == 0:
        return arr
    w = max(1, min(window, max(1, arr.size // 10)))
    if w <= 1:
        return arr.astype(float)
    kernel = np.ones(w) / w
    return np.convolve(arr.astype(float), kernel, mode="same")


def stack_curves(cell, smooth_window):
    """Return (mean_curve, std_curve) for a cell's per-seed reward curves, truncated to shortest."""
    curves = []
    for d in cell["completed"]:
        c = d.get("cumulative_reward_curve") or []
        if c:
            curves.append(np.asarray(c, dtype=float))
    if not curves:
        return None, None
    min_len = min(len(c) for c in curves)
    if min_len < 1:
        return None, None
    mat = np.stack([c[:min_len] for c in curves], axis=0)  # (n_seeds, min_len)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    mean_s = rolling_mean(mean, smooth_window)
    std_s = rolling_mean(std, smooth_window)
    return mean_s, std_s


def fig1_cumulative_reward(phase, cells, out_path):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), sharey=False)
    for col, algo in enumerate(ALGOS):
        ax = axes[col]
        any_line = False
        for wrapper in WRAPPERS:
            cell = cells.get((algo, wrapper))
            if cell is None or cell.get("skipped") or not cell.get("completed"):
                continue
            mean_s, std_s = stack_curves(cell, phase["smooth_window"])
            if mean_s is None:
                continue
            x = np.arange(mean_s.size)
            color = WRAPPER_COLOR[wrapper]
            ax.plot(x, mean_s, color=color, label=WRAPPER_DISPLAY[wrapper], linewidth=1.4)
            ax.fill_between(x, mean_s - std_s, mean_s + std_s, color=color, alpha=0.15, linewidth=0)
            any_line = True
        ax.set_title(ALGO_DISPLAY[algo], fontsize=11)
        ax.set_xlabel("Episode")
        if col == 0:
            ax.set_ylabel("Reward (smoothed)")
        ax.grid(True, alpha=0.3, linestyle=":")
        if not any_line:
            ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes, ha="center", va="center", color="0.5")
    # Single legend in upper-left of first panel
    h, l = axes[0].get_legend_handles_labels()
    if h:
        axes[0].legend(h, l, loc="upper left", fontsize=8, framealpha=0.85)
    fig.suptitle(f"Phase {phase['n']}: {phase['label']} - cumulative reward (window={phase['smooth_window']})",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"OK: wrote {out_path}")


def pool_p99(cells, wrapper):
    vals = []
    for algo in ALGOS:
        cell = cells.get((algo, wrapper))
        if cell is None or cell.get("skipped") or not cell.get("completed"):
            continue
        for d in cell["completed"]:
            v = d.get("mean_p99_step_ms")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            vals.append(float(v))
    return np.asarray(sorted(vals), dtype=float)


def ecdf(x):
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def fig2_latency_cdf(phase, cells, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    all_vals = []
    plotted = 0
    for wrapper in WRAPPERS:
        v = pool_p99(cells, wrapper)
        if v.size == 0:
            continue
        all_vals.extend(v.tolist())
        x, y = ecdf(v)
        ax.step(x, y, where="post", label=f"{WRAPPER_DISPLAY[wrapper]} (n={v.size})",
                color=WRAPPER_COLOR[wrapper], linewidth=1.6)
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes, ha="center", va="center", color="0.5")
    if all_vals:
        vmin, vmax = min(all_vals), max(all_vals)
        if vmin > 0 and vmax / vmin > 100:
            ax.set_xscale("log")
    ax.set_xlabel("Per-step p99 latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Phase {phase['n']}: {phase['label']} - p99 step latency ECDF (pooled algo x seed)")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"OK: wrote {out_path}")


def _final_reward(curve):
    if not curve:
        return float("nan")
    tail = curve[-10:] if len(curve) >= 10 else curve
    return float(statistics.fmean(tail))


def fig3_dqn_three_panel(phase, cells, out_path):
    """3-panel for DQN: ECDF of p99, energy bars, final-reward bars."""
    algo = "dqn"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel A: ECDF of mean_p99_step_ms across 3 seeds for each wrapper
    axA = axes[0]
    plotted_a = 0
    all_vals_a = []
    for wrapper in WRAPPERS:
        cell = cells.get((algo, wrapper))
        if cell is None or cell.get("skipped") or not cell.get("completed"):
            continue
        vals = sorted(
            float(d["mean_p99_step_ms"])
            for d in cell["completed"]
            if d.get("mean_p99_step_ms") is not None
        )
        if not vals:
            continue
        v = np.asarray(vals, dtype=float)
        all_vals_a.extend(vals)
        x, y = ecdf(v)
        axA.step(x, y, where="post", label=WRAPPER_DISPLAY[wrapper],
                 color=WRAPPER_COLOR[wrapper], linewidth=1.8, marker="o", markersize=4)
        plotted_a += 1
    if plotted_a == 0:
        axA.text(0.5, 0.5, "(no DQN data)", transform=axA.transAxes, ha="center", va="center", color="0.5")
    if all_vals_a and min(all_vals_a) > 0 and max(all_vals_a) / min(all_vals_a) > 100:
        axA.set_xscale("log")
    axA.set_title("(A) DQN: p99 step latency ECDF")
    axA.set_xlabel("p99 step latency (ms)")
    axA.set_ylabel("CDF")
    axA.set_ylim(0, 1.02)
    axA.grid(True, alpha=0.3, linestyle=":")
    axA.legend(loc="lower right", fontsize=8)

    # Panel B: bar chart of mean energy/step (mJ) per wrapper for DQN
    axB = axes[1]
    labels_b, means_b, stds_b, colors_b = [], [], [], []
    for wrapper in WRAPPERS:
        cell = cells.get((algo, wrapper))
        if cell is None or cell.get("skipped") or not cell.get("completed"):
            continue
        vals = [float(d["mean_energy_j"]) * 1000.0 for d in cell["completed"] if d.get("mean_energy_j") is not None]
        if not vals:
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals)) if len(vals) > 1 else 0.0
        labels_b.append(WRAPPER_DISPLAY[wrapper])
        means_b.append(m)
        stds_b.append(s)
        colors_b.append(WRAPPER_COLOR[wrapper])
    if labels_b:
        x_pos = np.arange(len(labels_b))
        axB.bar(x_pos, means_b, yerr=stds_b, color=colors_b, capsize=4, edgecolor="black", linewidth=0.5)
        axB.set_xticks(x_pos)
        axB.set_xticklabels(labels_b, rotation=20, ha="right", fontsize=9)
    else:
        axB.text(0.5, 0.5, "(no DQN data)", transform=axB.transAxes, ha="center", va="center", color="0.5")
    axB.set_title("(B) DQN: mean energy per step")
    axB.set_ylabel("Energy/step (mJ)")
    axB.grid(True, axis="y", alpha=0.3, linestyle=":")

    # Panel C: bar chart of final reward (mean of last 10 episodes; mean +/- std over 3 seeds)
    axC = axes[2]
    labels_c, means_c, stds_c, colors_c = [], [], [], []
    for wrapper in WRAPPERS:
        cell = cells.get((algo, wrapper))
        if cell is None or cell.get("skipped") or not cell.get("completed"):
            continue
        vals = [_final_reward(d.get("cumulative_reward_curve")) for d in cell["completed"]]
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        if not vals:
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals)) if len(vals) > 1 else 0.0
        labels_c.append(WRAPPER_DISPLAY[wrapper])
        means_c.append(m)
        stds_c.append(s)
        colors_c.append(WRAPPER_COLOR[wrapper])
    if labels_c:
        x_pos = np.arange(len(labels_c))
        axC.bar(x_pos, means_c, yerr=stds_c, color=colors_c, capsize=4, edgecolor="black", linewidth=0.5)
        axC.set_xticks(x_pos)
        axC.set_xticklabels(labels_c, rotation=20, ha="right", fontsize=9)
    else:
        axC.text(0.5, 0.5, "(no DQN data)", transform=axC.transAxes, ha="center", va="center", color="0.5")
    axC.set_title("(C) DQN: final reward (last 10 ep)")
    axC.set_ylabel("Reward")
    axC.grid(True, axis="y", alpha=0.3, linestyle=":")

    fig.suptitle(f"Phase {phase['n']}: {phase['label']} - DQN per-wrapper view", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"OK: wrote {out_path}")


def main():
    for phase in PHASES:
        cells = load_phase(phase)
        n = phase["n"]
        n_complete = sum(len(c["completed"]) for c in cells.values())
        n_skipped = sum(1 for c in cells.values() if c["skipped"] and not c["completed"])
        print(f"Phase {n}: {len(cells)} cells, {n_complete} COMPLETED, {n_skipped} SKIPPED pairs")
        fig1_cumulative_reward(phase, cells, FIGS_DIR / f"p15_phase{n}_cumulative_reward.pdf")
        fig2_latency_cdf(phase, cells, FIGS_DIR / f"p15_phase{n}_latency_cdf.pdf")
        fig3_dqn_three_panel(phase, cells, FIGS_DIR / f"p15_phase{n}_r3_3panel.pdf")


if __name__ == "__main__":
    main()
