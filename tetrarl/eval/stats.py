"""Aggregation + significance tests for Week 8 ablation sweeps.

Reads the eval runner's ``summary.csv`` schema, groups rows by the
``ablation`` column, and reports per-group means/stds along with Welch's
two-sample t-test of each non-baseline arm vs the ``none`` baseline on
``n_steps``, ``override_fire_count``, ``mean_reward``, and
``tail_p99_ms``. A small CLI emits the paper-ready Markdown table
(Table 4) and copies the input CSV under the spec-mandated
``ablation_summary.csv`` name.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Sequence

from scipy import stats as _scipy_stats

ABLATION_ORDER: tuple[str, ...] = (
    "none",
    "preference_plane",
    "resource_manager",
    "rl_arbiter",
    "override_layer",
)

_NUMERIC_COLUMNS: tuple[str, ...] = (
    "n_steps",
    "mean_reward",
    "tail_p99_ms",
    "override_fire_count",
    "mean_energy_j",
)


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Welch's two-sample t-test (unequal variance), two-sided.

    Returns ``(t_statistic, p_value)``. Raises :class:`ValueError` if
    either sample has fewer than two elements (the test is undefined).
    """
    if len(a) < 2 or len(b) < 2:
        raise ValueError(
            f"welch_t_test requires >=2 samples per group, got len(a)={len(a)}, len(b)={len(b)}"
        )
    res = _scipy_stats.ttest_ind(list(a), list(b), equal_var=False)
    return float(res.statistic), float(res.pvalue)


def _safe_mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, std) with std=0.0 for n<2 (avoids stdev raising)."""
    if not values:
        return (float("nan"), float("nan"))
    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if len(values) >= 2 else 0.0
    return (mean, std)


def _read_summary_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _coerce_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def aggregate_ablation(csv_path: str | Path) -> list[dict]:
    """Aggregate a sweep ``summary.csv`` into one row per ablation arm.

    Each returned dict has the columns:
      - ``ablation``: arm name
      - ``n_seeds``: number of rows for this arm
      - ``{metric}_mean`` and ``{metric}_std`` for each metric in
        ``n_steps``, ``mean_reward``, ``tail_p99_ms``,
        ``override_fire_count``, ``mean_energy_j``
      - ``t_stat_n_steps_vs_baseline`` / ``p_n_steps_vs_baseline``
      - ``t_stat_override_fire_count_vs_baseline`` /
        ``p_override_fire_count_vs_baseline``
      - ``t_stat_mean_reward_vs_baseline`` / ``p_mean_reward_vs_baseline``
      - ``t_stat_tail_p99_vs_baseline`` / ``p_tail_p99_vs_baseline``

    The baseline row (``ablation == 'none'``) has ``None`` for all eight
    t-stat / p-value columns. Rows are emitted in :data:`ABLATION_ORDER`;
    arms missing from the CSV are skipped.
    """
    rows = _read_summary_rows(Path(csv_path))

    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        arm = str(row.get("ablation", "")).strip()
        if not arm:
            continue
        bucket = grouped.setdefault(arm, {col: [] for col in _NUMERIC_COLUMNS})
        for col in _NUMERIC_COLUMNS:
            bucket[col].append(_coerce_float(row.get(col, "nan")))

    baseline_n_steps = grouped.get("none", {}).get("n_steps", [])
    baseline_override = grouped.get("none", {}).get("override_fire_count", [])
    baseline_rewards = grouped.get("none", {}).get("mean_reward", [])
    baseline_tail = grouped.get("none", {}).get("tail_p99_ms", [])

    out: list[dict] = []
    for arm in ABLATION_ORDER:
        if arm not in grouped:
            continue
        bucket = grouped[arm]
        record: dict = {
            "ablation": arm,
            "n_seeds": len(bucket["mean_reward"]),
        }
        for col in _NUMERIC_COLUMNS:
            mean, std = _safe_mean_std(bucket[col])
            record[f"{col}_mean"] = mean
            record[f"{col}_std"] = std

        if arm == "none":
            record["t_stat_n_steps_vs_baseline"] = None
            record["p_n_steps_vs_baseline"] = None
            record["t_stat_override_fire_count_vs_baseline"] = None
            record["p_override_fire_count_vs_baseline"] = None
            record["t_stat_mean_reward_vs_baseline"] = None
            record["p_mean_reward_vs_baseline"] = None
            record["t_stat_tail_p99_vs_baseline"] = None
            record["p_tail_p99_vs_baseline"] = None
        else:
            try:
                t_n, p_n = welch_t_test(bucket["n_steps"], baseline_n_steps)
            except ValueError:
                t_n, p_n = float("nan"), float("nan")
            try:
                t_o, p_o = welch_t_test(
                    bucket["override_fire_count"], baseline_override
                )
            except ValueError:
                t_o, p_o = float("nan"), float("nan")
            try:
                t_r, p_r = welch_t_test(bucket["mean_reward"], baseline_rewards)
            except ValueError:
                t_r, p_r = float("nan"), float("nan")
            try:
                t_t, p_t = welch_t_test(bucket["tail_p99_ms"], baseline_tail)
            except ValueError:
                t_t, p_t = float("nan"), float("nan")
            record["t_stat_n_steps_vs_baseline"] = t_n
            record["p_n_steps_vs_baseline"] = p_n
            record["t_stat_override_fire_count_vs_baseline"] = t_o
            record["p_override_fire_count_vs_baseline"] = p_o
            record["t_stat_mean_reward_vs_baseline"] = t_r
            record["p_mean_reward_vs_baseline"] = p_r
            record["t_stat_tail_p99_vs_baseline"] = t_t
            record["p_tail_p99_vs_baseline"] = p_t

        out.append(record)

    return out


def _fmt_mean_std(mean: float, std: float) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "—"
    return f"{mean:.3f} ± {std:.3f}"


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    if isinstance(p, float) and math.isnan(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _significance_marker(
    p_n_steps: float | None,
    p_override: float | None,
    p_reward: float | None,
    p_tail: float | None,
) -> str:
    candidates = [
        p
        for p in (p_n_steps, p_override, p_reward, p_tail)
        if p is not None and not math.isnan(p)
    ]
    if not candidates:
        return ""
    p_min = min(candidates)
    if p_min < 0.001:
        return "***"
    if p_min < 0.01:
        return "**"
    if p_min < 0.05:
        return "*"
    return "ns"


def format_paper_table(rows: list[dict]) -> str:
    """Render aggregate rows as a paper-ready Markdown table (Table 4)."""
    header = (
        "| Ablation | n_steps (μ ± σ) | Override fires (μ ± σ) | "
        "Tail p99 ms (μ ± σ) | Mean energy J (μ ± σ) | "
        "Mean Reward (μ ± σ) | p (n_steps) | p (override) | "
        "p (tail p99) | p (reward) | sig |"
    )
    align = "|---|---|---|---|---|---|---|---|---|---|---|"

    body_lines: list[str] = []
    for r in rows:
        is_baseline = r["ablation"] == "none"
        p_n_steps = r.get("p_n_steps_vs_baseline")
        p_override = r.get("p_override_fire_count_vs_baseline")
        p_reward = r.get("p_mean_reward_vs_baseline")
        p_tail = r.get("p_tail_p99_vs_baseline")
        marker = (
            ""
            if is_baseline
            else _significance_marker(p_n_steps, p_override, p_reward, p_tail)
        )
        body_lines.append(
            "| {abl} | {ns} | {ov} | {tp} | {en} | {mr} | {pn} | {po} | {pt} | {pr} | {sig} |".format(
                abl=r["ablation"],
                ns=_fmt_mean_std(r["n_steps_mean"], r["n_steps_std"]),
                ov=_fmt_mean_std(r["override_fire_count_mean"], r["override_fire_count_std"]),
                tp=_fmt_mean_std(r["tail_p99_ms_mean"], r["tail_p99_ms_std"]),
                en=_fmt_mean_std(r["mean_energy_j_mean"], r["mean_energy_j_std"]),
                mr=_fmt_mean_std(r["mean_reward_mean"], r["mean_reward_std"]),
                pn="—" if is_baseline else _fmt_p(p_n_steps),
                po="—" if is_baseline else _fmt_p(p_override),
                pt="—" if is_baseline else _fmt_p(p_tail),
                pr="—" if is_baseline else _fmt_p(p_reward),
                sig=marker,
            )
        )

    title = (
        "### Table 4. Ablation study (CartPole-v1, n=3 seeds, 200 episodes; "
        "reward saturates at 1.0/step on CartPole — see n_steps for the action signal)"
    )
    return "\n".join([title, "", header, align, *body_lines])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tetrarl.eval.stats",
        description="Aggregate ablation sweep summary.csv into Table 4 + p-values.",
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to runner summary.csv")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument(
        "--baseline-ablation",
        type=str,
        default="none",
        help="Ablation name treated as the baseline (default: none)",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_ablation != "none":
        # Future-proof hook: aggregate_ablation currently hard-codes 'none'
        # as the baseline. Surface a clear message rather than silently
        # producing wrong stats if the operator overrides this.
        print(
            f"[stats] WARNING: --baseline-ablation={args.baseline_ablation!r} ignored; "
            "aggregate_ablation uses 'none' as baseline.",
            file=sys.stderr,
        )

    rows = aggregate_ablation(csv_path)
    table_md = format_paper_table(rows)

    table_path = out_dir / "ablation_table.md"
    table_path.write_text(table_md + "\n", encoding="utf-8")

    summary_copy = out_dir / "ablation_summary.csv"
    if csv_path.resolve() != summary_copy.resolve():
        shutil.copyfile(csv_path, summary_copy)

    print(table_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
