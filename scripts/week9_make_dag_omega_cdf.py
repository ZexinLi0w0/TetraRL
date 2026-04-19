"""Week 9 Task D: per-omega CDF plotter for the 4-D DAG sweep.

Reads per-omega ``trace.jsonl`` files produced by
``scripts/week9_nano_dag_sweep.py`` and emits a single matplotlib figure
with one empirical-CDF curve per preference vector omega. Reuses the
empirical-CDF + percentile + style-cycle helpers from
``scripts.week7_make_cdf`` so the rendering matches the Week 7 figures.

The trace lines have the form::

    {"episode": int, "step": int, "action": int,
     "reward_vec": [float, float, float, float],
     "scalarised_reward": float, "latency_ms": float,
     "energy_step": float, "memory_step": float, "omega": [float, ...]}

The plotter pulls ``latency_ms`` from each line and emits one CDF curve
per ``--omegas`` keyword.

Example::

    python3 scripts/week9_make_dag_omega_cdf.py \\
        --in-dir runs/w9_nano_dag \\
        --omegas energy_corner,memory_corner,center \\
        --out-png runs/w9_nano_dag/omega_cdf.png \\
        --out-svg runs/w9_nano_dag/omega_cdf.svg

A Markdown summary of p50 / p95 / p99 per omega is also printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

# Force a non-interactive backend; safe for headless CI / Orin / Nano runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow ``matplotlib.use``)
import numpy as np  # noqa: E402

from scripts.week7_make_cdf import (  # noqa: E402
    _STYLE_CYCLE,
    _empirical_cdf,
    _percentiles,
)


def _load_trace_latencies(path: Path) -> np.ndarray:
    """Return the ``latency_ms`` column from a Week 9 DAG sweep trace.jsonl."""
    samples: list[float] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "latency_ms" in row:
                samples.append(float(row["latency_ms"]))
    return np.asarray(samples, dtype=np.float64)


def _parse_omegas(s: str) -> list[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--omegas must list at least one keyword")
    return parts


def _format_summary_table(
    rows: list[tuple[str, int, dict[str, float]]],
) -> str:
    header = "| omega | n | p50_ms | p95_ms | p99_ms |"
    sep = "|-------|---|--------|--------|--------|"
    out = [header, sep]
    for omega, n, pcts in rows:
        out.append(
            f"| {omega} | {n} | "
            f"{pcts['p50']:.3f} | {pcts['p95']:.3f} | {pcts['p99']:.3f} |"
        )
    return "\n".join(out) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-dir",
        required=True,
        help="Sweep root containing one <omega>/trace.jsonl subdir per omega.",
    )
    parser.add_argument(
        "--omegas",
        default="energy_corner,memory_corner,center",
        help="Comma-separated omega keywords (default: energy_corner,memory_corner,center).",
    )
    parser.add_argument(
        "--out-png",
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--out-svg",
        default=None,
        help="Optional second output path written in SVG format.",
    )
    parser.add_argument(
        "--xscale",
        choices=("linear", "log"),
        default="linear",
        help="X-axis scale (default: linear).",
    )
    parser.add_argument(
        "--title",
        default="Week 9: Nano DAG step-latency CDF per preference vector omega",
        help="Figure title.",
    )
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        print(
            f"error: --in-dir {str(in_dir)!r} does not exist or is not a directory",
            file=sys.stderr,
        )
        return 1

    omegas = _parse_omegas(args.omegas)
    out_png = Path(args.out_png)
    out_svg = Path(args.out_svg) if args.out_svg is not None else None

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))
    plotted: list[tuple[str, int, dict[str, float]]] = []

    for idx, omega in enumerate(omegas):
        jsonl_path = in_dir / omega / "trace.jsonl"
        if not jsonl_path.exists():
            print(
                f"warning: missing trace.jsonl for omega={omega}: {jsonl_path}",
                file=sys.stderr,
            )
            continue
        latencies = _load_trace_latencies(jsonl_path)
        if latencies.size == 0:
            print(
                f"warning: no latency samples in {jsonl_path}",
                file=sys.stderr,
            )
            continue
        x, y = _empirical_cdf(latencies)
        color, linestyle = _STYLE_CYCLE[idx % len(_STYLE_CYCLE)]
        ax.plot(
            x,
            y,
            label=f"{omega} (n={x.size})",
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )
        plotted.append((omega, int(x.size), _percentiles(latencies)))

    if not plotted:
        print(
            "error: no omegas had any latency data; nothing to plot",
            file=sys.stderr,
        )
        plt.close(fig)
        return 1

    ax.set_xlabel("Step latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(args.title)
    ax.set_xscale(args.xscale)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg)
    plt.close(fig)

    print(f"Saved per-omega CDF plot: {out_png}")
    if out_svg is not None:
        print(f"Saved per-omega CDF plot: {out_svg}")

    print()
    print(_format_summary_table(plotted), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
