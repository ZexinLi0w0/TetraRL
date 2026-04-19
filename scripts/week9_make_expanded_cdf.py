"""Week 9 cross-platform expanded CDF: per-ω 12-curve 2-panel plotter.

Extends the Week 7 combined CDF figure (see
``scripts/week7_make_combined_cdf.py``) by sweeping over preference
vectors (ω) in addition to FFmpeg co-runner conditions. Each panel
(left = Orin, right = Nano) renders up to ``len(omegas) * len(conditions)``
empirical-CDF curves (default 3 × 4 = 12 curves per panel).

Per-ω inputs live at ``<root>/<omega>/<condition>.jsonl`` on each side; the
recorder schema (``{"sample_ms": float, "idx": int}``) is reused unchanged
via :func:`scripts.week7_make_cdf._load_latencies`.

Skipping rules:
  * Missing ``<omega>/`` directory  -> warn on stderr, skip the whole ω
    on that panel only.
  * Missing single ``<omega>/<condition>.jsonl`` -> warn on stderr, skip
    that single curve.
  * Empty JSONL (zero usable samples) -> warn on stderr, skip that curve.

Hard-fails (exit 1) only when ``--orin-dir`` or ``--nano-dir`` itself is
missing, or when no curves were plotted on either panel.

Example::

    python3 scripts/week9_make_expanded_cdf.py \\
        --orin-dir runs/w9_ffmpeg_orin_per_omega \\
        --nano-dir runs/w9_ffmpeg_nano_per_omega \\
        --omegas energy_corner,memory_corner,center \\
        --conditions none,720p,1080p,2K \\
        --out-png runs/w9_expanded_cdf.png \\
        --out-svg runs/w9_expanded_cdf.svg

A Markdown summary table is written to stdout AND to a sidecar file at
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
    _parse_conditions,
    _percentiles,
)

# Stable per-ω colour family. Falls back to ``_STYLE_CYCLE`` for unknown ωs.
_OMEGA_COLORS: dict[str, str] = {
    "energy_corner": "tab:blue",
    "memory_corner": "tab:orange",
    "center": "tab:green",
}

# Stable per-condition linestyle. Falls back to "-" for unknown conditions.
_CONDITION_LINESTYLES: dict[str, str] = {
    "none": "-",
    "720p": "--",
    "1080p": ":",
    "2K": "-.",
}


# A summary row recorded per (platform, ω, condition) actually plotted.
_SummaryRow = tuple[str, str, str, int, dict[str, float]]


def _parse_omegas(raw: str) -> list[str]:
    """Split a comma-separated ``--omegas`` string into trimmed names.

    No whitelist is enforced; any user-supplied ω string passes through so
    that future preference vectors can be added without code changes.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--omegas must list at least one preference vector name")
    return parts


def _color_for_omega(omega: str, omega_index: int) -> str:
    """Return the stable colour for ``omega`` or fall back to ``_STYLE_CYCLE``."""
    if omega in _OMEGA_COLORS:
        return _OMEGA_COLORS[omega]
    color, _ = _STYLE_CYCLE[omega_index % len(_STYLE_CYCLE)]
    return color


def _linestyle_for_condition(condition: str) -> str:
    """Return the stable linestyle for ``condition`` or fall back to solid."""
    return _CONDITION_LINESTYLES.get(condition, "-")


def _format_expanded_summary_table(rows: list[_SummaryRow]) -> str:
    """Render a Markdown summary table including platform, omega, condition."""
    header = "| platform | omega | condition | n | p50_ms | p95_ms | p99_ms |"
    sep = "|----------|-------|-----------|---|--------|--------|--------|"
    out = [header, sep]
    for platform, omega, cond, n, pcts in rows:
        out.append(
            f"| {platform} | {omega} | {cond} | {n} | "
            f"{pcts['p50']:.3f} | {pcts['p95']:.3f} | {pcts['p99']:.3f} |"
        )
    return "\n".join(out) + "\n"


def _plot_one_panel_per_omega(
    ax,
    root_dir: Path,
    omegas: list[str],
    conditions: list[str],
    panel_label: str,
) -> list[_SummaryRow]:
    """Render one device's per-ω CDF curves into ``ax`` and return summary rows.

    Sweeps the (ω, condition) grid in row-major order. Each ω contributes a
    colour family (see :data:`_OMEGA_COLORS`); each condition contributes a
    distinct linestyle (see :data:`_CONDITION_LINESTYLES`) so the legend is
    decodable into the two underlying axes.
    """
    plotted: list[_SummaryRow] = []
    for oi, omega in enumerate(omegas):
        omega_dir = root_dir / omega
        if not omega_dir.is_dir():
            print(
                f"warning: missing omega dir for {panel_label}/{omega}: {omega_dir}",
                file=sys.stderr,
            )
            continue
        color = _color_for_omega(omega, oi)
        for cond in conditions:
            jsonl_path = omega_dir / f"{cond}.jsonl"
            if not jsonl_path.exists():
                print(
                    f"warning: missing JSONL for {panel_label}/{omega}/{cond}: {jsonl_path}",
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
            ax.plot(
                x,
                y,
                label=f"{omega} | {cond} (n={x.size})",
                color=color,
                linestyle=_linestyle_for_condition(cond),
                linewidth=1.4,
                alpha=0.85,
            )
            plotted.append(
                (panel_label, omega, cond, int(x.size), _percentiles(latencies))
            )
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
        help="Directory containing Nano <omega>/<condition>.jsonl files.",
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
        "--conditions",
        default="none,720p,1080p,2K",
        help="Comma-separated condition names (default: none,720p,1080p,2K).",
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
        default="Orin AGX: per-ω FFmpeg co-runner",
        help="Title for the left (Orin) panel.",
    )
    parser.add_argument(
        "--nano-title",
        default="Orin Nano: per-ω FFmpeg co-runner",
        help="Title for the right (Nano) panel.",
    )
    parser.add_argument(
        "--suptitle",
        default="Week 9: training-step latency CDF (per-ω × FFmpeg co-runner)",
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
    conditions = _parse_conditions(args.conditions)

    fig, (ax_orin, ax_nano) = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)

    plotted_orin = _plot_one_panel_per_omega(
        ax_orin, orin_dir, omegas, conditions, "orin"
    )
    plotted_nano = _plot_one_panel_per_omega(
        ax_nano, nano_dir, omegas, conditions, "nano"
    )

    if not plotted_orin and not plotted_nano:
        print(
            "error: no (omega, condition) pairs had data on either device; "
            "nothing to plot",
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
            ax.legend(loc="lower right", fontsize="x-small", ncol=1)
    ax_orin.set_ylabel("CDF")

    fig.suptitle(args.suptitle)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg)
    plt.close(fig)

    print(f"Saved expanded CDF plot: {out_png}")
    if out_svg is not None:
        print(f"Saved expanded CDF plot: {out_svg}")

    table = _format_expanded_summary_table(plotted_orin + plotted_nano)
    print()
    print(table, end="")

    summary_md = out_png.with_suffix(out_png.suffix + ".summary.md")
    summary_md.write_text(table, encoding="utf-8")
    print(f"Saved expanded CDF summary: {summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
