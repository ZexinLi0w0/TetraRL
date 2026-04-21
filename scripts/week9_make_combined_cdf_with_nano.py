"""Week 9 P1: 2-panel combined CDF (Orin + Nano) per preference vector omega.

Renders a single matplotlib figure with two side-by-side panels (left = Orin
AGX, right = Orin Nano), each showing one empirical-CDF curve per preference
vector omega. The Orin panel reads the per-omega FFmpeg co-runner JSONL files
(``<omega>/<condition>.jsonl`` from ``scripts/week9_ffmpeg_corunner.py``);
``--orin-condition`` selects which condition to compare against the Nano runs
(default: ``none``, i.e. no FFmpeg co-runner, which is the apples-to-apples
match for the Nano DAG sweep). The Nano panel reads ``<omega>/trace.jsonl``
files produced by ``scripts/week9_nano_dag_sweep.py``.

Per-omega colours are shared with ``scripts.week9_make_expanded_cdf`` so the
ω identity is visually consistent across all Week 9 figures.

Example::

    python3 scripts/week9_make_combined_cdf_with_nano.py \\
        --orin-dir runs/w9_ffmpeg_orin_per_omega \\
        --nano-dir runs/w9_nano_dag \\
        --omegas energy_corner,memory_corner,center \\
        --orin-condition none \\
        --out-png runs/w9_combined_cdf_with_nano.png \\
        --out-svg runs/w9_combined_cdf_with_nano.svg

Writes a Markdown summary table to stdout AND to a sidecar file at
``<out-png>.summary.md`` so the figure is self-documenting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib

# Force a non-interactive backend; safe for headless CI / Orin / Nano runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow ``matplotlib.use``)

from scripts.week7_make_cdf import (  # noqa: E402
    _STYLE_CYCLE,
    _empirical_cdf,
    _load_latencies,
    _percentiles,
)
from scripts.week9_make_dag_omega_cdf import _load_trace_latencies  # noqa: E402
from scripts.week9_make_expanded_cdf import _color_for_omega  # noqa: E402


# A summary row recorded per (platform, ω) actually plotted.
_SummaryRow = tuple[str, str, int, dict[str, float]]


def _parse_omegas(raw: str) -> list[str]:
    """Split a comma-separated ``--omegas`` string into trimmed names."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--omegas must list at least one preference vector name")
    return parts


def _format_summary_table(rows: list[_SummaryRow]) -> str:
    """Render a Markdown summary table with platform + omega leading columns."""
    header = "| platform | omega | n | p50_ms | p95_ms | p99_ms |"
    sep = "|----------|-------|---|--------|--------|--------|"
    out = [header, sep]
    for platform, omega, n, pcts in rows:
        out.append(
            f"| {platform} | {omega} | {n} | "
            f"{pcts['p50']:.3f} | {pcts['p95']:.3f} | {pcts['p99']:.3f} |"
        )
    return "\n".join(out) + "\n"


def _plot_orin_panel(
    ax,
    orin_dir: Path,
    omegas: list[str],
    condition: str,
) -> list[_SummaryRow]:
    """Render Orin per-ω CDF curves at a fixed FFmpeg condition into ``ax``."""
    plotted: list[_SummaryRow] = []
    for oi, omega in enumerate(omegas):
        jsonl_path = orin_dir / omega / f"{condition}.jsonl"
        if not jsonl_path.exists():
            print(
                f"warning: missing Orin JSONL for {omega}/{condition}: {jsonl_path}",
                file=sys.stderr,
            )
            continue
        latencies = _load_latencies(jsonl_path)
        if latencies.size == 0:
            print(
                f"warning: no latency samples in {jsonl_path}",
                file=sys.stderr,
            )
            continue
        x, y = _empirical_cdf(latencies)
        color = _color_for_omega(omega, oi)
        _, linestyle = _STYLE_CYCLE[oi % len(_STYLE_CYCLE)]
        ax.plot(
            x,
            y,
            label=f"{omega} (n={x.size})",
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )
        plotted.append(("orin", omega, int(x.size), _percentiles(latencies)))
    return plotted


def _plot_nano_panel(
    ax,
    nano_dir: Path,
    omegas: list[str],
) -> list[_SummaryRow]:
    """Render Nano per-ω CDF curves from ``<omega>/trace.jsonl`` into ``ax``."""
    plotted: list[_SummaryRow] = []
    for oi, omega in enumerate(omegas):
        jsonl_path = nano_dir / omega / "trace.jsonl"
        if not jsonl_path.exists():
            print(
                f"warning: missing Nano trace.jsonl for {omega}: {jsonl_path}",
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
        color = _color_for_omega(omega, oi)
        _, linestyle = _STYLE_CYCLE[oi % len(_STYLE_CYCLE)]
        ax.plot(
            x,
            y,
            label=f"{omega} (n={x.size})",
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )
        plotted.append(("nano", omega, int(x.size), _percentiles(latencies)))
    return plotted


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orin-dir",
        required=True,
        help="Directory containing Orin <omega>/<condition>.jsonl files.",
    )
    parser.add_argument(
        "--nano-dir",
        required=True,
        help="Directory containing Nano <omega>/trace.jsonl files.",
    )
    parser.add_argument(
        "--omegas",
        default="energy_corner,memory_corner,center",
        help=(
            "Comma-separated preference-vector names "
            "(default: energy_corner,memory_corner,center)."
        ),
    )
    parser.add_argument(
        "--orin-condition",
        default="none",
        help=(
            "Which Orin FFmpeg co-runner condition to plot per omega "
            "(default: none, i.e. no co-runner — apples-to-apples with the "
            "Nano DAG sweep, which has no co-runner)."
        ),
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
        default="log",
        help=(
            "X-axis scale (default: log — Orin and Nano span ~2 orders of "
            "magnitude in latency)."
        ),
    )
    parser.add_argument(
        "--orin-title",
        default=None,
        help=(
            "Title for the left (Orin) panel (default: includes the chosen "
            "--orin-condition)."
        ),
    )
    parser.add_argument(
        "--nano-title",
        default="Orin Nano: per-ω DAG sweep step latency",
        help="Title for the right (Nano) panel.",
    )
    parser.add_argument(
        "--suptitle",
        default="Week 9: training-step latency CDF per ω (Orin AGX vs Orin Nano)",
        help="Overall figure title.",
    )
    args = parser.parse_args(argv)

    orin_dir = Path(args.orin_dir)
    nano_dir = Path(args.nano_dir)
    out_png = Path(args.out_png)
    out_svg = Path(args.out_svg) if args.out_svg is not None else None

    # Hard-fail if either root directory is missing; this is a configuration
    # error, distinct from "directory exists but has no usable per-ω data".
    if not orin_dir.is_dir():
        print(
            f"error: --orin-dir {str(orin_dir)!r} does not exist or is not a directory",
            file=sys.stderr,
        )
        return 1
    if not nano_dir.is_dir():
        print(
            f"error: --nano-dir {str(nano_dir)!r} does not exist or is not a directory",
            file=sys.stderr,
        )
        return 1

    omegas = _parse_omegas(args.omegas)
    orin_title = args.orin_title or (
        f"Orin AGX: per-ω step latency (FFmpeg condition: {args.orin_condition})"
    )

    fig, (ax_orin, ax_nano) = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)

    plotted_orin = _plot_orin_panel(ax_orin, orin_dir, omegas, args.orin_condition)
    plotted_nano = _plot_nano_panel(ax_nano, nano_dir, omegas)

    if not plotted_orin and not plotted_nano:
        print(
            "error: no omegas had data on either device; nothing to plot",
            file=sys.stderr,
        )
        plt.close(fig)
        return 1

    for ax, title, plotted in (
        (ax_orin, orin_title, plotted_orin),
        (ax_nano, args.nano_title, plotted_nano),
    ):
        ax.set_xlabel("Step latency (ms)")
        ax.set_title(title)
        ax.set_xscale(args.xscale)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="both", alpha=0.3)
        if plotted:
            ax.legend(loc="lower right")
    ax_orin.set_ylabel("CDF")

    fig.suptitle(args.suptitle)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg)
    plt.close(fig)

    print(f"Saved combined CDF plot: {out_png}")
    if out_svg is not None:
        print(f"Saved combined CDF plot: {out_svg}")

    table = _format_summary_table(plotted_orin + plotted_nano)
    print()
    print(table, end="")

    summary_md = out_png.with_suffix(out_png.suffix + ".summary.md")
    summary_md.write_text(table, encoding="utf-8")
    print(f"Saved combined CDF summary: {summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
