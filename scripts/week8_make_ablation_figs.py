"""Generate the two Week 8 ablation paper figures from ``summary.csv``.

Reads ``runs/w8_ablation_orin/summary.csv`` (default; override with
``--runs-dir``), groups rows by ``ablation``, and writes two PNGs
side-by-side with the Markdown Table 4:

  * ``violation_rate_bar.png`` — mean override-fire rate (fires / step)
    per ablation arm with std error bars.
  * ``hv_comparison.png`` — mean cumulative ``n_steps`` per arm with std
    error bars (HV proxy on CartPole, where per-step reward is constant).

Both figures use a colour-blind-friendly palette (Wong 2011) with the
``none`` baseline rendered in a distinct shade, and a footer noting that
the data is from the Mac substitute run while the Orin re-run is pending
(override the footer text with ``--footer "..."``, e.g. for the W8
real-Orin re-run that closes PR #24's deviation).
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

# Wong 2011 colour-blind-safe palette.
_PALETTE: dict[str, str] = {
    "none": "#000000",                # baseline = black for distinction
    "preference_plane": "#E69F00",    # orange
    "resource_manager": "#56B4E9",    # sky blue
    "rl_arbiter": "#009E73",          # bluish green
    "override_layer": "#CC79A7",      # reddish purple
}

_ABLATION_ORDER: tuple[str, ...] = (
    "none",
    "preference_plane",
    "resource_manager",
    "rl_arbiter",
    "override_layer",
)

_FOOTER = "Mac substitute run; Orin re-run pending — see progress.md"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_summary(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _group_metrics(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, list[float]]]:
    """Return ``{ablation: {metric: [values per seed]}}``.

    Computes the per-row violation rate (override_fire_count / n_steps)
    and keeps the raw n_steps for the HV-proxy chart.
    """
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        arm = str(row.get("ablation", "")).strip()
        if not arm:
            continue
        try:
            n_steps = float(row["n_steps"])
            override = float(row["override_fire_count"])
        except (KeyError, ValueError):
            continue
        rate = override / n_steps if n_steps > 0 else 0.0
        bucket = grouped.setdefault(arm, {"violation_rate": [], "n_steps": []})
        bucket["violation_rate"].append(rate)
        bucket["n_steps"].append(n_steps)
    return grouped


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return float(mean), float(std)


def _ordered_arms(grouped: dict[str, dict[str, list[float]]]) -> list[str]:
    return [arm for arm in _ABLATION_ORDER if arm in grouped]


def _make_bar_chart(
    arms: list[str],
    means: list[float],
    stds: list[float],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    footer: str = _FOOTER,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    colours = [_PALETTE.get(arm, "#999999") for arm in arms]
    x = list(range(len(arms)))
    ax.bar(
        x,
        means,
        yerr=stds,
        color=colours,
        edgecolor="black",
        capsize=4,
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.figtext(
        0.99,
        0.01,
        footer,
        ha="right",
        va="bottom",
        fontsize=7,
        style="italic",
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the two Week 8 ablation paper figures.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing summary.csv; the two PNGs are written here "
            "too. Defaults to <repo>/runs/w8_ablation_orin/."
        ),
    )
    parser.add_argument(
        "--footer",
        type=str,
        default=_FOOTER,
        help="Footer text rendered at the bottom-right of each PNG.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    runs_dir = args.runs_dir if args.runs_dir is not None else _REPO_ROOT / "runs" / "w8_ablation_orin"
    csv_path = runs_dir / "summary.csv"
    out_dir = runs_dir

    rows = _read_summary(csv_path)
    grouped = _group_metrics(rows)
    arms = _ordered_arms(grouped)

    rate_stats = [_mean_std(grouped[a]["violation_rate"]) for a in arms]
    rate_means = [m for m, _ in rate_stats]
    rate_stds = [s for _, s in rate_stats]

    step_stats = [_mean_std(grouped[a]["n_steps"]) for a in arms]
    step_means = [m for m, _ in step_stats]
    step_stds = [s for _, s in step_stats]

    _make_bar_chart(
        arms,
        rate_means,
        rate_stds,
        title="Override fire rate per ablation (CartPole, n=3 seeds)",
        ylabel="override_fire_count / n_steps (mean ± std)",
        out_path=out_dir / "violation_rate_bar.png",
        footer=args.footer,
    )

    _make_bar_chart(
        arms,
        step_means,
        step_stds,
        title=(
            "Cumulative episode steps per ablation "
            "(CartPole HV proxy; reward saturates at 1.0/step)"
        ),
        ylabel="n_steps (mean ± std)",
        out_path=out_dir / "hv_comparison.png",
        footer=args.footer,
    )

    print(f"[figs] wrote {out_dir / 'violation_rate_bar.png'}")
    print(f"[figs] wrote {out_dir / 'hv_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
