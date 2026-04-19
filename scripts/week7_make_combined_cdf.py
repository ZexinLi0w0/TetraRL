"""Week 7 cleanup Task D: 2-panel (Orin + Nano) combined CDF plotter.

Reads per-condition JSONL files from two run directories (one per device)
produced by ``scripts/week7_ffmpeg_corunner.py`` and emits a single
matplotlib figure with two side-by-side panels (left = Orin, right =
Nano), each showing one empirical-CDF curve per requested condition.

The recorder writes lines of the form ``{"sample_ms": float, "idx": int}``;
this script also accepts ``latency_ms`` / ``step`` keys for forward
compatibility, so it can plot output from either schema interchangeably.
Helpers are imported from ``scripts.week7_make_cdf`` to avoid duplicating
the JSONL loader, the empirical-CDF helper, and the percentile helper.

Example::

    python3 scripts/week7_make_combined_cdf.py \\
        --orin-dir runs/w7_ffmpeg_orin \\
        --nano-dir runs/w7_ffmpeg_nano \\
        --conditions none,720p,1080p \\
        --out-png runs/w7_combined_cdf.png \\
        --out-svg runs/w7_combined_cdf.svg

The script also prints a Markdown table of p50 / p95 / p99 to stdout for
both panels so the results are self-documenting in CI logs.
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
    _parse_conditions,
    _percentiles,
)


def _format_combined_summary_table(
    rows: list[tuple[str, str, int, dict[str, float]]],
) -> str:
    """Render a Markdown summary table with a ``platform`` leading column."""
    header = "| platform | condition | n | p50_ms | p95_ms | p99_ms |"
    sep = "|----------|-----------|---|--------|--------|--------|"
    out = [header, sep]
    for platform, cond, n, pcts in rows:
        out.append(
            f"| {platform} | {cond} | {n} | "
            f"{pcts['p50']:.3f} | {pcts['p95']:.3f} | {pcts['p99']:.3f} |"
        )
    return "\n".join(out) + "\n"


def _plot_one_panel(
    ax,
    in_dir: Path,
    conditions: list[str],
    panel_label: str,
) -> list[tuple[str, str, int, dict[str, float]]]:
    """Render one device's CDF curves into ``ax`` and return summary rows.

    Returns one summary tuple ``(platform, condition, n, percentiles)`` per
    condition that actually had latency samples.
    """
    plotted: list[tuple[str, str, int, dict[str, float]]] = []
    for idx, cond in enumerate(conditions):
        jsonl_path = in_dir / f"{cond}.jsonl"
        if not jsonl_path.exists():
            print(
                f"warning: missing JSONL for {panel_label}/{cond}: {jsonl_path}",
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
        color, linestyle = _STYLE_CYCLE[idx % len(_STYLE_CYCLE)]
        ax.plot(
            x,
            y,
            label=f"{cond} (n={x.size})",
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )
        plotted.append((panel_label, cond, int(x.size), _percentiles(latencies)))
    return plotted


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orin-dir",
        required=True,
        help="Directory containing Orin <condition>.jsonl files.",
    )
    parser.add_argument(
        "--nano-dir",
        required=True,
        help="Directory containing Nano <condition>.jsonl files.",
    )
    parser.add_argument(
        "--conditions",
        default="none,720p,1080p",
        help="Comma-separated condition names (default: none,720p,1080p).",
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
        "--orin-title",
        default="Orin AGX: FFmpeg CPU-decode co-runner",
        help="Title for the left (Orin) panel.",
    )
    parser.add_argument(
        "--nano-title",
        default="Orin Nano: FFmpeg CPU-decode co-runner",
        help="Title for the right (Nano) panel.",
    )
    parser.add_argument(
        "--suptitle",
        default="Week 7: training-step latency CDF under FFmpeg co-runner",
        help="Overall figure title.",
    )
    args = parser.parse_args(argv)

    orin_dir = Path(args.orin_dir)
    nano_dir = Path(args.nano_dir)
    out_png = Path(args.out_png)
    out_svg = Path(args.out_svg) if args.out_svg is not None else None

    # Hard-fail if either directory does not exist; this is a configuration
    # error, distinct from "directory exists but has no usable JSONLs".
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

    conditions = _parse_conditions(args.conditions)

    fig, (ax_orin, ax_nano) = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)

    plotted_orin = _plot_one_panel(ax_orin, orin_dir, conditions, "orin")
    plotted_nano = _plot_one_panel(ax_nano, nano_dir, conditions, "nano")

    if not plotted_orin and not plotted_nano:
        print(
            "error: no conditions had any latency data on either device; nothing to plot",
            file=sys.stderr,
        )
        plt.close(fig)
        return 1

    for ax, title, plotted in (
        (ax_orin, args.orin_title, plotted_orin),
        (ax_nano, args.nano_title, plotted_nano),
    ):
        ax.set_xlabel("Latency (ms)")
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

    table = _format_combined_summary_table(plotted_orin + plotted_nano)
    print()
    print(table, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
