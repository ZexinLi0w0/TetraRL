#!/usr/bin/env python3
"""Plot hypervolume convergence from PD-MORL training progress JSON."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_hv_convergence(
    progress: list[dict[str, Any]],
    out: str | Path,
    reference: float | None = 229.0,
    dpi: int = 150,
) -> Path:
    """Plot HV vs environment steps and save to *out*. Returns the output path."""
    out = Path(out)
    frames = np.array([r["frames"] for r in progress])
    hvs = np.array([r["hv"] for r in progress])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frames, hvs, linewidth=2, color="#1f77b4", label="PD-MORL")
    if reference is not None:
        ax.axhline(
            reference,
            linestyle="--",
            color="#d62728",
            linewidth=1.2,
            label=f"PD-MORL paper reference ({reference:.0f})",
        )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Hypervolume")
    ax.set_title("PD-MORL on Deep Sea Treasure \u2014 Hypervolume convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HV convergence curve")
    parser.add_argument(
        "--progress-json", default="progress.json", help="Path to progress.json"
    )
    parser.add_argument(
        "--out", default="hv_convergence.png", help="Output PNG path"
    )
    parser.add_argument(
        "--reference", type=float, default=229.0, help="Reference HV value"
    )
    parser.add_argument(
        "--no-reference", action="store_true", help="Omit reference line"
    )
    args = parser.parse_args()

    with open(args.progress_json) as f:
        progress = json.load(f)

    ref = None if args.no_reference else args.reference
    out = plot_hv_convergence(progress, args.out, reference=ref)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
