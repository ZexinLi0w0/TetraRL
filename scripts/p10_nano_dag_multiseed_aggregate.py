#!/usr/bin/env python3
"""Aggregate Item #3b Nano DAG multi-seed sweep into per-omega tables.

Reads ``runs/p10_nano_dag_multiseed/seed{0..4}/summary.csv`` (each a 3-row
CSV emitted by ``scripts/week9_nano_dag_sweep.py``) and produces:

  * ``nano_dag_long.csv``     - flat 5x3 = 15-row CSV (seed x omega).
  * ``nano_dag_summary.csv``  - per-omega mean/std across seeds.
  * ``nano_dag_table.md``     - markdown table for embedding in notes.
  * ``nano_dag_table.tex``    - LaTeX booktabs ``tabular`` block.

Robust to partial sweeps: missing ``seed{N}/summary.csv`` files are skipped
with a warning.

Stdlib + numpy only (no pandas).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Seeds we expect from the sweep script.
EXPECTED_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

# Canonical omega ordering for emitted tables.
OMEGA_ORDER: Tuple[str, ...] = ("energy_corner", "memory_corner", "center")

# Numeric columns we aggregate.
METRIC_COLS: Tuple[str, ...] = (
    "mean_scalarised_reward",
    "tail_p99_ms",
    "mean_energy_step",
    "mean_memory_delta",
    "wall_time_s",
)

# Long-form CSV column ordering.
LONG_COLS: Tuple[str, ...] = (
    "seed",
    "omega_name",
    "n_episodes",
    "n_steps",
    "mean_scalarised_reward",
    "tail_p99_ms",
    "mean_energy_step",
    "mean_memory_delta",
    "wall_time_s",
)

# Summary CSV column ordering.
SUMMARY_COLS: Tuple[str, ...] = (
    "omega_name",
    "n",
    "mean_scal_reward",
    "std_scal_reward",
    "mean_p99_ms",
    "std_p99_ms",
    "mean_energy",
    "std_energy",
    "mean_memory",
    "std_memory",
    "mean_wall_s",
    "std_wall_s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs/p10_nano_dag_multiseed"),
        help="Directory containing seed{N}/summary.csv subdirectories.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for aggregated artifacts (default: same as --runs-dir).",
    )
    return p.parse_args()


def load_seed_summary(path: Path) -> List[Dict[str, str]]:
    """Load one seed's summary.csv as a list of row dicts."""
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_all(runs_dir: Path) -> Dict[int, List[Dict[str, str]]]:
    """Load all available seed summaries; warn on missing seeds."""
    out: Dict[int, List[Dict[str, str]]] = {}
    for seed in EXPECTED_SEEDS:
        f = runs_dir / f"seed{seed}" / "summary.csv"
        if not f.exists():
            print(f"[warn] missing {f}; skipping seed {seed}", file=sys.stderr)
            continue
        try:
            out[seed] = load_seed_summary(f)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] failed to read {f}: {exc}; skipping seed {seed}",
                  file=sys.stderr)
    return out


def write_long_csv(per_seed: Dict[int, List[Dict[str, str]]], out_path: Path) -> None:
    """Write the flat seed x omega long-form CSV."""
    rows: List[Dict[str, object]] = []
    for seed in sorted(per_seed.keys()):
        for row in per_seed[seed]:
            rows.append({
                "seed": seed,
                "omega_name": row["omega_name"],
                "n_episodes": row["n_episodes"],
                "n_steps": row["n_steps"],
                "mean_scalarised_reward": row["mean_scalarised_reward"],
                "tail_p99_ms": row["tail_p99_ms"],
                "mean_energy_step": row["mean_energy_step"],
                "mean_memory_delta": row["mean_memory_delta"],
                "wall_time_s": row["wall_time_s"],
            })
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(LONG_COLS))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(per_seed: Dict[int, List[Dict[str, str]]]) -> Dict[str, Dict[str, float]]:
    """Group rows by omega and compute mean/std for each metric."""
    by_omega: Dict[str, Dict[str, List[float]]] = {
        omega: {metric: [] for metric in METRIC_COLS}
        for omega in OMEGA_ORDER
    }
    for seed_rows in per_seed.values():
        for row in seed_rows:
            omega = row["omega_name"]
            if omega not in by_omega:
                # Tolerate unexpected omegas by registering on the fly.
                by_omega[omega] = {metric: [] for metric in METRIC_COLS}
            for metric in METRIC_COLS:
                by_omega[omega][metric].append(float(row[metric]))

    summary: Dict[str, Dict[str, float]] = {}
    for omega, metrics in by_omega.items():
        n = len(metrics[METRIC_COLS[0]])
        if n == 0:
            continue
        entry: Dict[str, float] = {"n": float(n)}
        for metric in METRIC_COLS:
            arr = np.asarray(metrics[metric], dtype=float)
            entry[f"mean_{metric}"] = float(np.mean(arr))
            # ddof=1 sample std; if n==1 numpy returns nan, which is fine.
            entry[f"std_{metric}"] = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
        summary[omega] = entry
    return summary


def write_summary_csv(summary: Dict[str, Dict[str, float]], out_path: Path) -> None:
    """Write per-omega mean/std table as CSV."""
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_COLS))
        writer.writeheader()
        for omega in OMEGA_ORDER:
            if omega not in summary:
                continue
            s = summary[omega]
            writer.writerow({
                "omega_name": omega,
                "n": int(s["n"]),
                "mean_scal_reward": f"{s['mean_mean_scalarised_reward']:.6f}",
                "std_scal_reward": f"{s['std_mean_scalarised_reward']:.6f}",
                "mean_p99_ms": f"{s['mean_tail_p99_ms']:.6f}",
                "std_p99_ms": f"{s['std_tail_p99_ms']:.6f}",
                "mean_energy": f"{s['mean_mean_energy_step']:.6f}",
                "std_energy": f"{s['std_mean_energy_step']:.6f}",
                "mean_memory": f"{s['mean_mean_memory_delta']:.6f}",
                "std_memory": f"{s['std_mean_memory_delta']:.6f}",
                "mean_wall_s": f"{s['mean_wall_time_s']:.6f}",
                "std_wall_s": f"{s['std_wall_time_s']:.6f}",
            })
        # Append any unexpected omegas so we never silently drop data.
        for omega, s in summary.items():
            if omega in OMEGA_ORDER:
                continue
            writer.writerow({
                "omega_name": omega,
                "n": int(s["n"]),
                "mean_scal_reward": f"{s['mean_mean_scalarised_reward']:.6f}",
                "std_scal_reward": f"{s['std_mean_scalarised_reward']:.6f}",
                "mean_p99_ms": f"{s['mean_tail_p99_ms']:.6f}",
                "std_p99_ms": f"{s['std_tail_p99_ms']:.6f}",
                "mean_energy": f"{s['mean_mean_energy_step']:.6f}",
                "std_energy": f"{s['std_mean_energy_step']:.6f}",
                "mean_memory": f"{s['mean_mean_memory_delta']:.6f}",
                "std_memory": f"{s['std_mean_memory_delta']:.6f}",
                "mean_wall_s": f"{s['mean_wall_time_s']:.6f}",
                "std_wall_s": f"{s['std_wall_time_s']:.6f}",
            })


def _fmt(mean: float, std: float, decimals: int) -> str:
    """Format ``mean +/- std`` with the requested precision (plain ASCII)."""
    if np.isnan(std):
        return f"{mean:.{decimals}f} +/- nan"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def _fmt_md(mean: float, std: float, decimals: int) -> str:
    """Format mean/std for markdown using a Unicode +/- glyph."""
    if np.isnan(std):
        return f"{mean:.{decimals}f} ± nan"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _fmt_tex(mean: float, std: float, decimals: int) -> str:
    """Format mean/std for LaTeX math mode with $\\pm$."""
    if np.isnan(std):
        return f"${mean:.{decimals}f} \\pm \\mathrm{{nan}}$"
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def _ordered_omegas(summary: Dict[str, Dict[str, float]]) -> List[str]:
    """Canonical omega order, then any extras."""
    ordered = [o for o in OMEGA_ORDER if o in summary]
    extras = [o for o in summary if o not in OMEGA_ORDER]
    return ordered + extras


def render_markdown(summary: Dict[str, Dict[str, float]]) -> str:
    """Render the markdown table string."""
    header = (
        "| Preference ω "
        "| Mean scal. reward (mean ± std) "
        "| Tail p99 (ms) (mean ± std) "
        "| Mean energy/step (mean ± std) "
        "| Mean memory delta (mean ± std) "
        "| Wall (s) (mean ± std) |"
    )
    sep = "| --- | --- | --- | --- | --- | --- |"
    lines: List[str] = [header, sep]
    for omega in _ordered_omegas(summary):
        s = summary[omega]
        lines.append(
            "| `" + omega + "` "
            f"| {_fmt_md(s['mean_mean_scalarised_reward'], s['std_mean_scalarised_reward'], 3)} "
            f"| {_fmt_md(s['mean_tail_p99_ms'], s['std_tail_p99_ms'], 3)} "
            f"| {_fmt_md(s['mean_mean_energy_step'], s['std_mean_energy_step'], 4)} "
            f"| {_fmt_md(s['mean_mean_memory_delta'], s['std_mean_memory_delta'], 4)} "
            f"| {_fmt_md(s['mean_wall_time_s'], s['std_wall_time_s'], 2)} |"
        )
    return "\n".join(lines) + "\n"


def render_latex(summary: Dict[str, Dict[str, float]]) -> str:
    """Render a LaTeX booktabs ``tabular`` block."""
    lines: List[str] = []
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append(
        "Preference $\\omega$ "
        "& Mean scal. reward "
        "& Tail p99 (ms) "
        "& Mean energy/step "
        "& Mean memory $\\Delta$ "
        "& Wall (s) \\\\"
    )
    lines.append("\\midrule")
    for omega in _ordered_omegas(summary):
        s = summary[omega]
        # Escape underscores for \texttt{...}.
        omega_tt = "\\texttt{" + omega.replace("_", "\\_") + "}"
        lines.append(
            f"{omega_tt} "
            f"& {_fmt_tex(s['mean_mean_scalarised_reward'], s['std_mean_scalarised_reward'], 3)} "
            f"& {_fmt_tex(s['mean_tail_p99_ms'], s['std_tail_p99_ms'], 3)} "
            f"& {_fmt_tex(s['mean_mean_energy_step'], s['std_mean_energy_step'], 4)} "
            f"& {_fmt_tex(s['mean_mean_memory_delta'], s['std_mean_memory_delta'], 4)} "
            f"& {_fmt_tex(s['mean_wall_time_s'], s['std_wall_time_s'], 2)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    runs_dir: Path = args.runs_dir
    out_dir: Path = args.out_dir if args.out_dir is not None else runs_dir

    if not runs_dir.exists():
        print(f"[error] runs dir does not exist: {runs_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = load_all(runs_dir)
    if not per_seed:
        print(f"[error] no seed{{N}}/summary.csv files found under {runs_dir}",
              file=sys.stderr)
        return 1

    long_path = out_dir / "nano_dag_long.csv"
    summary_csv_path = out_dir / "nano_dag_summary.csv"
    md_path = out_dir / "nano_dag_table.md"
    tex_path = out_dir / "nano_dag_table.tex"

    write_long_csv(per_seed, long_path)
    summary = aggregate(per_seed)
    write_summary_csv(summary, summary_csv_path)

    md = render_markdown(summary)
    tex = render_latex(summary)
    md_path.write_text(md)
    tex_path.write_text(tex)

    # Emit on stdout so callers can pipe / inspect.
    print(md, end="")

    print(f"[ok] wrote {long_path}", file=sys.stderr)
    print(f"[ok] wrote {summary_csv_path}", file=sys.stderr)
    print(f"[ok] wrote {md_path}", file=sys.stderr)
    print(f"[ok] wrote {tex_path}", file=sys.stderr)
    print(f"[ok] seeds aggregated: {sorted(per_seed.keys())}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
