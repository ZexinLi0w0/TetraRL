#!/usr/bin/env python3
"""Week 11 Nano-GRPO response-curves figure.

Reads ``per_step_with_critic.csv`` and ``per_step_without_critic.csv`` from
``--in-dir`` and renders a 2x2 panel comparing the two passes along the
system-level response axes:

  * (top-left)  Per-step total latency CDF (ms)
  * (top-right) Cumulative GPU energy over training (J)
  * (bottom-left) RAM used over training (MB; tegrastats RAM_USED_MB)
  * (bottom-right) GPU utilization over training (%)

Output: ``<in-dir>/response_curves.{png,pdf,svg}``
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _read_csv(path: Path) -> Dict[str, np.ndarray]:
    rows: List[dict] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    cols: Dict[str, np.ndarray] = {}
    if not rows:
        return cols
    for k in rows[0].keys():
        try:
            cols[k] = np.array([float(row[k]) for row in rows])
        except ValueError:
            cols[k] = np.array([row[k] for row in rows])
    return cols


def _cumulative_energy_j(
    total_ms: np.ndarray, gpu_mw: np.ndarray
) -> np.ndarray:
    """Approximate cumulative GPU energy (J) from per-step latency and
    the tegrastats GPU power snapshot at the end of each step.

    Each step contributes (P_gpu_W * step_seconds) to total energy.
    """
    p_w = gpu_mw / 1000.0
    dt_s = total_ms / 1000.0
    return np.cumsum(p_w * dt_s)


def _smooth(x: np.ndarray, window: int = 11) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in-dir",
        required=True,
        help="Directory containing per_step_with_critic.csv and per_step_without_critic.csv",
    )
    p.add_argument(
        "--out-name",
        default="response_curves",
        help="Output basename (no extension); written under --in-dir.",
    )
    p.add_argument("--smooth-window", type=int, default=11)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    csv_w = in_dir / "per_step_with_critic.csv"
    csv_wo = in_dir / "per_step_without_critic.csv"
    if not csv_w.exists() or not csv_wo.exists():
        raise SystemExit(
            f"missing CSV(s) in {in_dir}: "
            f"{csv_w.exists()=} {csv_wo.exists()=}"
        )

    data = {
        "with_critic": _read_csv(csv_w),
        "without_critic": _read_csv(csv_wo),
    }
    colors = {"with_critic": "#1f77b4", "without_critic": "#d62728"}
    labels = {
        "with_critic": "with critic (PPO-style)",
        "without_critic": "without critic (GRPO)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # --- (TL) latency CDF ---
    ax = axes[0, 0]
    for name, d in data.items():
        x = np.sort(d["total_ms"])
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=colors[name], label=labels[name], lw=2)
    ax.set_xlabel("Per-step total latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("(a) Per-step latency CDF")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)

    # --- (TR) cumulative GPU energy ---
    ax = axes[0, 1]
    for name, d in data.items():
        cume = _cumulative_energy_j(d["total_ms"], d["vdd_gpu_mw"])
        ax.plot(d["step"], cume, color=colors[name], label=labels[name], lw=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cumulative GPU energy (J)")
    ax.set_title("(b) Cumulative energy")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False)

    # --- (BL) RAM used over training ---
    ax = axes[1, 0]
    for name, d in data.items():
        ax.plot(
            d["step"],
            _smooth(d["ram_used_mb"], args.smooth_window),
            color=colors[name],
            label=labels[name],
            lw=2,
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("RAM used (MB)")
    ax.set_title("(c) RAM (tegrastats RAM_USED_MB)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)

    # --- (BR) GPU utilization over training ---
    ax = axes[1, 1]
    for name, d in data.items():
        ax.plot(
            d["step"],
            _smooth(d["gpu_util_pct"], args.smooth_window),
            color=colors[name],
            label=labels[name],
            lw=2,
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("GPU utilization (%)")
    ax.set_title("(d) GPU utilization (GR3D_FREQ)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)

    fig.suptitle(
        "Nano-GRPO on Orin AGX: with-critic vs without-critic response curves",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    for ext in ("png", "pdf", "svg"):
        out = in_dir / f"{args.out_name}.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
