"""Week 7 Task 8a deliverable: CDF plotter for FFmpeg co-runner JSONL output.

Reads per-condition JSONL files produced by ``scripts/week7_ffmpeg_corunner.py``
(see :class:`tetrarl.eval.ffmpeg_interference.LatencyRecorder`) and emits an
empirical-CDF plot of per-step training latency, one curve per condition.

The recorder writes lines of the form ``{"sample_ms": float, "idx": int}``;
this script also accepts ``latency_ms`` / ``step`` keys for forward
compatibility, so it can plot output from either schema interchangeably.

Example::

    python3 scripts/week7_make_cdf.py \\
        --in-dir runs/week7_ffmpeg_1700000000 \\
        --conditions none,720p,1080p,2K \\
        --out-png runs/week7_ffmpeg_1700000000/cdf.png

The script also prints a Markdown table of p50 / p95 / p99 to stdout so the
results are self-documenting in CI logs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

# Force a non-interactive backend; safe for headless CI / Orin runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow ``matplotlib.use``)
import numpy as np  # noqa: E402


# Latency value can live under either of these keys; we accept both so the
# plotter works against the recorder's native schema (``sample_ms``) and the
# spec's forward-looking schema (``latency_ms``).
_LATENCY_KEYS: tuple[str, ...] = ("latency_ms", "sample_ms")

# Distinct (color, linestyle) pairs so curves stay readable in B&W printouts.
_STYLE_CYCLE: tuple[tuple[str, str], ...] = (
    ("#1f77b4", "-"),   # blue, solid
    ("#d62728", "--"),  # red, dashed
    ("#2ca02c", "-."),  # green, dash-dot
    ("#9467bd", ":"),   # purple, dotted
    ("#ff7f0e", "-"),   # orange, solid
    ("#17becf", "--"),  # teal, dashed
)


def _parse_conditions(raw: str) -> list[str]:
    """Split a comma-separated ``--conditions`` string into trimmed names."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--conditions must list at least one condition name")
    return parts


def _parse_xlim(raw: Optional[str]) -> Optional[tuple[float, float]]:
    """Parse ``--xlim-ms`` of the form ``min,max`` into a (lo, hi) tuple."""
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit(
            f"--xlim-ms expects 'min,max', got {raw!r}"
        )
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise SystemExit(f"--xlim-ms values must be numeric: {exc}") from exc
    if not (lo < hi):
        raise SystemExit(
            f"--xlim-ms requires min < max, got min={lo} max={hi}"
        )
    return (lo, hi)


def _load_latencies(path: Path) -> np.ndarray:
    """Return a 1D float array of latencies (ms) parsed from a JSONL file.

    Lines that lack any known latency key (or fail to parse as JSON) are
    skipped silently; surfacing them per-line would be too noisy in a
    summary plotter and the recorder schema is well-defined.
    """
    samples: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            for key in _LATENCY_KEYS:
                if key in rec:
                    try:
                        samples.append(float(rec[key]))
                    except (TypeError, ValueError):
                        pass
                    break
    return np.asarray(samples, dtype=np.float64)


def _empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_sorted, y_cdf)`` for the empirical CDF of ``values``.

    Uses the standard ``y = i/n`` form (i in 1..n) so the curve reaches
    1.0 at the largest sample.
    """
    if values.size == 0:
        raise ValueError("empirical_cdf requires at least one sample")
    x = np.sort(values)
    n = x.size
    y = np.arange(1, n + 1, dtype=np.float64) / float(n)
    return x, y


def _percentiles(values: np.ndarray) -> dict[str, float]:
    """Return p50/p95/p99 over ``values`` (linear interpolation, numpy default)."""
    p50, p95, p99 = np.percentile(values, [50.0, 95.0, 99.0])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99)}


def _format_summary_table(rows: list[tuple[str, int, dict[str, float]]]) -> str:
    """Render a Markdown summary table for the CI/console log."""
    header = "| condition | n | p50_ms | p95_ms | p99_ms |"
    sep = "|-----------|---|--------|--------|--------|"
    out = [header, sep]
    for cond, n, pcts in rows:
        out.append(
            f"| {cond} | {n} | {pcts['p50']:.3f} | "
            f"{pcts['p95']:.3f} | {pcts['p99']:.3f} |"
        )
    return "\n".join(out) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-dir",
        required=True,
        help="Directory containing <condition>.jsonl files.",
    )
    parser.add_argument(
        "--conditions",
        default="none,720p,1080p,2K",
        help="Comma-separated condition names (default: none,720p,1080p,2K).",
    )
    parser.add_argument(
        "--out-png",
        default=None,
        help="Output PNG path (default: <in-dir>/cdf.png).",
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
        "--xlim-ms",
        default=None,
        help="Optional X-axis range as 'min,max' in ms.",
    )
    parser.add_argument(
        "--title",
        default="FFmpeg co-runner: training step latency CDF",
        help="Plot title.",
    )
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        print(
            f"error: --in-dir {str(in_dir)!r} does not exist or is not a directory",
            file=sys.stderr,
        )
        return 1

    conditions = _parse_conditions(args.conditions)
    out_png = Path(args.out_png) if args.out_png is not None else in_dir / "cdf.png"
    out_svg = Path(args.out_svg) if args.out_svg is not None else None
    xlim = _parse_xlim(args.xlim_ms)

    plotted: list[tuple[str, int, dict[str, float]]] = []

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for idx, cond in enumerate(conditions):
        jsonl_path = in_dir / f"{cond}.jsonl"
        if not jsonl_path.exists():
            print(
                f"warning: missing JSONL for condition {cond!r}: {jsonl_path}",
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
        plotted.append((cond, int(x.size), _percentiles(latencies)))

    if not plotted:
        print(
            "error: no conditions had any latency data; nothing to plot",
            file=sys.stderr,
        )
        plt.close(fig)
        return 1

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(args.title)
    ax.set_xscale(args.xscale)
    if xlim is not None:
        ax.set_xlim(xlim)
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

    print(f"Saved CDF plot: {out_png}")
    if out_svg is not None:
        print(f"Saved CDF plot: {out_svg}")

    table = _format_summary_table(plotted)
    print()
    print(table, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
