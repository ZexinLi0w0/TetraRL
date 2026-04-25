#!/usr/bin/env python3
"""Aggregate Item #3a DST multi-seed eval results into a Table IV row.

Reads ``runs/p10_dst_multiseed_orin/seed{0..4}/eval.json`` (each emitted by
``scripts/eval_pd_morl_dst.py``) plus optional ``train.log`` files for
wall-clock training time, and produces:

  * ``dst_long.csv``        - one row per seed (achieved_hv, pareto_unique, train_wall_s).
  * ``dst_summary.csv``     - one row of cross-seed aggregates.
  * ``dst_table_row.md``    - one Markdown row for paper Table IV.
  * ``dst_table_row.tex``   - one LaTeX row matching paper Table IV columns.

Robust to partial sweeps: missing per-seed ``eval.json`` files are skipped
with a warning. Missing ``train.log`` files leave wall_clock blank.

Stdlib + numpy only (no pandas).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Seeds we expect from the sweep.
EXPECTED_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

# Long-form CSV column ordering.
LONG_COLS: Tuple[str, ...] = (
    "seed",
    "achieved_hv",
    "pareto_unique_count",
    "train_wall_s",
)

# Summary CSV column ordering.
SUMMARY_COLS: Tuple[str, ...] = (
    "n",
    "mean_hv",
    "std_hv",
    "min_hv",
    "max_hv",
    "mean_pareto_unique",
    "mean_train_wall_s",
)

# Regex for the final wall-clock line in train.log:
#   [ 50000/50000] HV=... t=NNNs
# We anchor on ``t=`` and an integer count of seconds before ``s``.
WALL_RE = re.compile(r"t=(\d+)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs/p10_dst_multiseed_orin"),
        help="Directory containing seed{N}/eval.json subdirectories.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for aggregated artifacts (default: same as --runs-dir).",
    )
    return p.parse_args()


def load_eval(path: Path) -> Dict[str, object]:
    """Load one seed's eval.json as a dict."""
    with path.open("r") as f:
        return json.load(f)


def parse_train_wall_s(log_path: Path) -> Optional[int]:
    """Scan a train.log file for the LAST ``t=NNNs`` token and return N.

    Returns ``None`` if the file is missing or no match is found.
    """
    if not log_path.exists():
        return None
    last: Optional[int] = None
    try:
        with log_path.open("r", errors="replace") as f:
            for line in f:
                m = WALL_RE.search(line)
                if m:
                    try:
                        last = int(m.group(1))
                    except ValueError:
                        continue
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] failed to parse {log_path}: {exc}", file=sys.stderr)
        return None
    return last


def pareto_unique_count(pareto_front: object) -> int:
    """Count unique (treasure, time) tuples in a pareto_front list."""
    if not isinstance(pareto_front, list):
        return 0
    uniq = set()
    for p in pareto_front:
        # Tolerate both list and tuple entries; require length >= 2.
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            uniq.add((p[0], p[1]))
    return len(uniq)


def load_all(
    runs_dir: Path,
) -> Dict[int, Dict[str, object]]:
    """Load all available per-seed records.

    Each record contains ``achieved_hv``, ``pareto_unique_count``, and
    ``train_wall_s`` (the latter possibly ``None``). Seeds with missing
    or malformed ``eval.json`` are skipped with a warning.
    """
    out: Dict[int, Dict[str, object]] = {}
    for seed in EXPECTED_SEEDS:
        seed_dir = runs_dir / f"seed{seed}"
        eval_path = seed_dir / "eval.json"
        if not eval_path.exists():
            print(f"[warn] missing {eval_path}; skipping seed {seed}",
                  file=sys.stderr)
            continue
        try:
            data = load_eval(eval_path)
        except Exception as exc:
            print(f"[warn] failed to read {eval_path}: {exc}; skipping seed {seed}",
                  file=sys.stderr)
            continue

        if "achieved_hv" not in data:
            print(f"[warn] {eval_path} missing 'achieved_hv'; skipping seed {seed}",
                  file=sys.stderr)
            continue

        try:
            hv = float(data["achieved_hv"])
        except (TypeError, ValueError) as exc:
            print(f"[warn] {eval_path} achieved_hv unparseable ({exc}); "
                  f"skipping seed {seed}", file=sys.stderr)
            continue

        unique = pareto_unique_count(data.get("pareto_front", []))
        wall = parse_train_wall_s(seed_dir / "train.log")

        out[seed] = {
            "achieved_hv": hv,
            "pareto_unique_count": unique,
            "train_wall_s": wall,
        }
    return out


def write_long_csv(per_seed: Dict[int, Dict[str, object]], out_path: Path) -> None:
    """Write the per-seed long-form CSV."""
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(LONG_COLS))
        writer.writeheader()
        for seed in sorted(per_seed.keys()):
            rec = per_seed[seed]
            wall = rec["train_wall_s"]
            writer.writerow({
                "seed": seed,
                "achieved_hv": f"{float(rec['achieved_hv']):.6f}",
                "pareto_unique_count": int(rec["pareto_unique_count"]),
                "train_wall_s": "" if wall is None else int(wall),
            })


def aggregate(per_seed: Dict[int, Dict[str, object]]) -> Dict[str, float]:
    """Compute cross-seed mean/std/min/max for the headline metric and
    mean of the auxiliary columns. Sample std uses ddof=1.
    """
    hvs = np.asarray([float(r["achieved_hv"]) for r in per_seed.values()],
                     dtype=float)
    pareto = np.asarray([int(r["pareto_unique_count"]) for r in per_seed.values()],
                        dtype=float)

    walls = [r["train_wall_s"] for r in per_seed.values() if r["train_wall_s"] is not None]
    walls_arr = np.asarray(walls, dtype=float) if walls else np.asarray([], dtype=float)

    n = int(hvs.size)
    summary: Dict[str, float] = {
        "n": float(n),
        "mean_hv": float(np.mean(hvs)) if n else float("nan"),
        # Sample std with ddof=1; numpy returns nan for n<=1, which matches our intent.
        "std_hv": float(np.std(hvs, ddof=1)) if n > 1 else float("nan"),
        "min_hv": float(np.min(hvs)) if n else float("nan"),
        "max_hv": float(np.max(hvs)) if n else float("nan"),
        "mean_pareto_unique": float(np.mean(pareto)) if n else float("nan"),
        "mean_train_wall_s": float(np.mean(walls_arr)) if walls_arr.size else float("nan"),
    }
    return summary


def write_summary_csv(summary: Dict[str, float], out_path: Path) -> None:
    """Write the single-row summary CSV."""
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_COLS))
        writer.writeheader()

        def _fmt(v: float, decimals: int = 6) -> str:
            if np.isnan(v):
                return "nan"
            return f"{v:.{decimals}f}"

        writer.writerow({
            "n": int(summary["n"]),
            "mean_hv": _fmt(summary["mean_hv"]),
            "std_hv": _fmt(summary["std_hv"]),
            "min_hv": _fmt(summary["min_hv"]),
            "max_hv": _fmt(summary["max_hv"]),
            "mean_pareto_unique": _fmt(summary["mean_pareto_unique"], 4),
            "mean_train_wall_s": _fmt(summary["mean_train_wall_s"], 2),
        })


def _fmt_md(mean: float, std: float, decimals: int) -> str:
    """Format ``mean ± std`` for Markdown."""
    if np.isnan(mean):
        return "n/a"
    if np.isnan(std):
        return f"{mean:.{decimals}f} ± nan"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _fmt_tex(mean: float, std: float, decimals: int) -> str:
    """Format ``mean \\pm std`` for LaTeX math mode."""
    if np.isnan(mean):
        return "n/a"
    if np.isnan(std):
        return f"${mean:.{decimals}f} \\pm \\mathrm{{nan}}$"
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def _fmt_int_or_blank(v: float, decimals: int = 0) -> str:
    """Render a numeric value as an int-ish string or '---' if NaN."""
    if np.isnan(v):
        return "---"
    if decimals == 0:
        return f"{int(round(v))}"
    return f"{v:.{decimals}f}"


def render_md_row(summary: Dict[str, float]) -> str:
    """Render the one-line Markdown row for Table IV."""
    n = int(summary["n"])
    mean_hv = summary["mean_hv"]
    std_hv = summary["std_hv"]
    mean_pareto = summary["mean_pareto_unique"]
    mean_wall = summary["mean_train_wall_s"]

    hv_cell = _fmt_md(mean_hv, std_hv, 1)
    pareto_cell = _fmt_int_or_blank(mean_pareto, 1)
    wall_cell = _fmt_int_or_blank(mean_wall, 0)

    label = f"TetraRL Orin AGX, {n}-seed train (50k frames)"
    return (
        f"| {label} "
        f"| {hv_cell} "
        f"| {pareto_cell} "
        f"| {wall_cell} |"
    )


def render_tex_row(summary: Dict[str, float]) -> str:
    """Render the one-line LaTeX row for Table IV."""
    n = int(summary["n"])
    mean_hv = summary["mean_hv"]
    std_hv = summary["std_hv"]
    mean_pareto = summary["mean_pareto_unique"]
    mean_wall = summary["mean_train_wall_s"]

    hv_cell = _fmt_tex(mean_hv, std_hv, 2)
    pareto_cell = _fmt_int_or_blank(mean_pareto, 1)
    wall_cell = _fmt_int_or_blank(mean_wall, 0)

    label = f"\\TetraRL Orin AGX, {n}-seed train"
    return (
        f"{label} "
        f"& --- "
        f"& {hv_cell} "
        f"& {pareto_cell} "
        f"& {wall_cell} \\\\"
    )


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
        print(f"[error] no usable seed{{N}}/eval.json files found under {runs_dir}",
              file=sys.stderr)
        return 1

    long_path = out_dir / "dst_long.csv"
    summary_csv_path = out_dir / "dst_summary.csv"
    md_path = out_dir / "dst_table_row.md"
    tex_path = out_dir / "dst_table_row.tex"

    write_long_csv(per_seed, long_path)
    summary = aggregate(per_seed)
    write_summary_csv(summary, summary_csv_path)

    md_row = render_md_row(summary) + "\n"
    tex_row = render_tex_row(summary) + "\n"
    md_path.write_text(md_row)
    tex_path.write_text(tex_row)

    # Print the markdown row + a short summary to stdout.
    print(md_row, end="")
    print()
    n = int(summary["n"])
    print(f"seeds aggregated : {sorted(per_seed.keys())}  (n={n})")
    print(f"achieved_hv      : mean={summary['mean_hv']:.4f}  "
          f"std={summary['std_hv']:.4f}  "
          f"min={summary['min_hv']:.4f}  "
          f"max={summary['max_hv']:.4f}")
    print(f"pareto_unique    : mean={summary['mean_pareto_unique']:.4f}")
    if not np.isnan(summary["mean_train_wall_s"]):
        print(f"train_wall_s     : mean={summary['mean_train_wall_s']:.2f}")
    else:
        print("train_wall_s     : n/a (no train.log files found)")

    print(f"[ok] wrote {long_path}", file=sys.stderr)
    print(f"[ok] wrote {summary_csv_path}", file=sys.stderr)
    print(f"[ok] wrote {md_path}", file=sys.stderr)
    print(f"[ok] wrote {tex_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
