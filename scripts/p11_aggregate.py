#!/usr/bin/env python3
"""Aggregate P11 head-to-head DST eval results across 4 baselines x 3 seeds.

Reads ``tetrarl/eval/configs/p11_dst_headtohead.yaml`` for the (baseline, seed)
manifest, then for each cell loads either ``<runs-dir>/<baseline>_seed<N>/eval.json``
(HV-baselines: tetrarl_pref_ppo, pd_morl, duojoule -- schema matches
``scripts/eval_pd_morl_dst.py``) or ``<runs-dir>/r3_seed<N>/r3_native_metrics.json``
(R3 only).

R3 does NOT target Pareto / hypervolume. Its row is emitted with empty HV columns
and a populated ``native_metric_value`` (framework_overhead_pct). The aggregator
keeps R3 in the manifest so the comparison table is complete, but never tries to
compute or compare HV against R3.

Outputs (all under ``--out-dir``):

  * ``p11_long.csv``    - one row per (baseline, seed) cell.
  * ``hv_summary.csv``  - one row per baseline with mean/std/min/max HV plus
                          R3's native metric.
  * ``p11_table.md``    - Markdown table for the paper SVII.B.
  * ``p11_table.tex``   - LaTeX tabular block matching paper Table IV style.

Robust to partial sweeps: missing per-seed files are skipped with a warning,
not a crash.

Stdlib + numpy + PyYAML only. No pandas.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


DEFAULT_CONFIG = Path("tetrarl/eval/configs/p11_dst_headtohead.yaml")

HV_BASELINES: Tuple[str, ...] = ("tetrarl_pref_ppo", "pd_morl", "duojoule")
NATIVE_METRIC_BASELINES: Tuple[str, ...] = ("r3",)

LONG_COLS: Tuple[str, ...] = (
    "baseline",
    "seed",
    "achieved_hv",
    "n_pareto",
    "framework_overhead_pct",
    "mean_deadline_miss_rate",
    "status",
)

SUMMARY_COLS: Tuple[str, ...] = (
    "baseline",
    "n_seeds_ok",
    "mean_hv",
    "std_hv",
    "min_hv",
    "max_hv",
    "mean_n_pareto",
    "native_metric_value",
    "native_metric_name",
    "notes",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs/p11_dst_headtohead"),
        help="Directory containing <baseline>_seed<N>/ subdirectories.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for aggregated artifacts (default: same as --runs-dir).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="P11 sweep manifest YAML.",
    )
    return p.parse_args()


def load_manifest(path: Path) -> Dict[str, object]:
    """Load the P11 sweep manifest YAML."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def pareto_unique_count(pareto_front: object) -> int:
    """Count unique (treasure, time) tuples in a pareto_front list."""
    if not isinstance(pareto_front, list):
        return 0
    uniq = set()
    for p in pareto_front:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            uniq.add((p[0], p[1]))
    return len(uniq)


def load_hv_cell(path: Path) -> Tuple[Optional[Dict[str, float]], str]:
    """Load one HV-baseline seed's eval.json.

    Returns ``(record, status)`` where ``record`` is None if the file is
    missing or unparseable.
    """
    if not path.exists():
        return None, "missing"
    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[warn] failed to read {path}: {exc}", file=sys.stderr)
        return None, "error"

    if "achieved_hv" not in data:
        print(f"[warn] {path} missing 'achieved_hv'", file=sys.stderr)
        return None, "error"
    try:
        hv = float(data["achieved_hv"])
    except (TypeError, ValueError) as exc:
        print(f"[warn] {path} achieved_hv unparseable ({exc})", file=sys.stderr)
        return None, "error"

    n_pareto = pareto_unique_count(data.get("pareto_front", []))
    return {"achieved_hv": hv, "n_pareto": n_pareto}, "ok"


def load_r3_cell(path: Path) -> Tuple[Optional[Dict[str, float]], str]:
    """Load one R3 seed's native-metrics JSON."""
    if not path.exists():
        return None, "missing"
    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[warn] failed to read {path}: {exc}", file=sys.stderr)
        return None, "error"

    try:
        overhead = float(data["framework_overhead_pct"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"[warn] {path} framework_overhead_pct unparseable ({exc})",
              file=sys.stderr)
        return None, "error"

    miss = data.get("mean_deadline_miss_rate")
    try:
        miss_val = float(miss) if miss is not None else None
    except (TypeError, ValueError):
        miss_val = None

    return {
        "framework_overhead_pct": overhead,
        "mean_deadline_miss_rate": miss_val,
    }, "ok"


def collect_cells(
    runs_dir: Path,
    baselines: List[Dict[str, object]],
    seeds: List[int],
) -> List[Dict[str, object]]:
    """Walk every (baseline, seed) cell and produce a long-form record list."""
    rows: List[Dict[str, object]] = []
    for b in baselines:
        name = str(b["name"])
        for seed in seeds:
            cell_dir = runs_dir / f"{name}_seed{seed}"
            row: Dict[str, object] = {
                "baseline": name,
                "seed": int(seed),
                "achieved_hv": "",
                "n_pareto": "",
                "framework_overhead_pct": "",
                "mean_deadline_miss_rate": "",
                "status": "missing",
            }
            if name in HV_BASELINES:
                rec, status = load_hv_cell(cell_dir / "eval.json")
                row["status"] = status
                if rec is not None:
                    row["achieved_hv"] = float(rec["achieved_hv"])
                    row["n_pareto"] = int(rec["n_pareto"])
                else:
                    print(f"[warn] missing or invalid HV cell: "
                          f"{cell_dir/'eval.json'}", file=sys.stderr)
            elif name in NATIVE_METRIC_BASELINES:
                rec, status = load_r3_cell(cell_dir / "r3_native_metrics.json")
                row["status"] = status
                if rec is not None:
                    row["framework_overhead_pct"] = float(rec["framework_overhead_pct"])
                    if rec["mean_deadline_miss_rate"] is not None:
                        row["mean_deadline_miss_rate"] = float(rec["mean_deadline_miss_rate"])
                else:
                    print(f"[warn] missing or invalid R3 cell: "
                          f"{cell_dir/'r3_native_metrics.json'}", file=sys.stderr)
            else:
                print(f"[warn] unknown baseline '{name}' (not in HV or native lists); "
                      f"skipping", file=sys.stderr)
                continue
            rows.append(row)
    return rows


def write_long_csv(rows: List[Dict[str, object]], path: Path) -> None:
    """Write the per-cell long-form CSV."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(LONG_COLS))
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in LONG_COLS}
            if isinstance(out["achieved_hv"], float):
                out["achieved_hv"] = f"{out['achieved_hv']:.6f}"
            if isinstance(out["framework_overhead_pct"], float):
                out["framework_overhead_pct"] = f"{out['framework_overhead_pct']:.4f}"
            if isinstance(out["mean_deadline_miss_rate"], float):
                out["mean_deadline_miss_rate"] = f"{out['mean_deadline_miss_rate']:.4f}"
            writer.writerow(out)


def aggregate_hv(rows: List[Dict[str, object]], baseline: str) -> Dict[str, float]:
    """Compute mean/std/min/max HV plus mean n_pareto for one HV baseline."""
    hvs: List[float] = []
    pareto: List[int] = []
    for r in rows:
        if r["baseline"] != baseline or r["status"] != "ok":
            continue
        if isinstance(r["achieved_hv"], (int, float)):
            hvs.append(float(r["achieved_hv"]))
        if isinstance(r["n_pareto"], int):
            pareto.append(int(r["n_pareto"]))

    n = len(hvs)
    return {
        "n_seeds_ok": n,
        "mean_hv": float(np.mean(hvs)) if n else float("nan"),
        "std_hv": float(np.std(hvs, ddof=1)) if n > 1 else float("nan"),
        "min_hv": float(np.min(hvs)) if n else float("nan"),
        "max_hv": float(np.max(hvs)) if n else float("nan"),
        "mean_n_pareto": float(np.mean(pareto)) if pareto else float("nan"),
    }


def aggregate_r3(rows: List[Dict[str, object]], baseline: str) -> Dict[str, float]:
    """Compute mean framework_overhead_pct over R3 seeds."""
    vals: List[float] = []
    for r in rows:
        if r["baseline"] != baseline or r["status"] != "ok":
            continue
        if isinstance(r["framework_overhead_pct"], (int, float)):
            vals.append(float(r["framework_overhead_pct"]))
    n = len(vals)
    return {
        "n_seeds_ok": n,
        "native_metric_value": float(np.mean(vals)) if n else float("nan"),
    }


def write_summary_csv(
    baselines: List[Dict[str, object]],
    rows: List[Dict[str, object]],
    path: Path,
) -> List[Dict[str, object]]:
    """Write per-baseline summary CSV; return the rows for downstream rendering."""
    summary_rows: List[Dict[str, object]] = []
    for b in baselines:
        name = str(b["name"])
        notes = str(b.get("notes", ""))
        if name in HV_BASELINES:
            agg = aggregate_hv(rows, name)
            summary_rows.append({
                "baseline": name,
                "n_seeds_ok": agg["n_seeds_ok"],
                "mean_hv": agg["mean_hv"],
                "std_hv": agg["std_hv"],
                "min_hv": agg["min_hv"],
                "max_hv": agg["max_hv"],
                "mean_n_pareto": agg["mean_n_pareto"],
                "native_metric_value": float("nan"),
                "native_metric_name": "",
                "notes": notes,
            })
        elif name in NATIVE_METRIC_BASELINES:
            agg = aggregate_r3(rows, name)
            summary_rows.append({
                "baseline": name,
                "n_seeds_ok": agg["n_seeds_ok"],
                "mean_hv": float("nan"),
                "std_hv": float("nan"),
                "min_hv": float("nan"),
                "max_hv": float("nan"),
                "mean_n_pareto": float("nan"),
                "native_metric_value": agg["native_metric_value"],
                "native_metric_name": "framework_overhead_pct",
                "notes": notes,
            })

    def _fmt(v: float, decimals: int = 4) -> str:
        if isinstance(v, float) and np.isnan(v):
            return ""
        return f"{v:.{decimals}f}"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_COLS))
        writer.writeheader()
        for s in summary_rows:
            writer.writerow({
                "baseline": s["baseline"],
                "n_seeds_ok": int(s["n_seeds_ok"]),
                "mean_hv": _fmt(s["mean_hv"]),
                "std_hv": _fmt(s["std_hv"]),
                "min_hv": _fmt(s["min_hv"]),
                "max_hv": _fmt(s["max_hv"]),
                "mean_n_pareto": _fmt(s["mean_n_pareto"], 2),
                "native_metric_value": _fmt(s["native_metric_value"], 4),
                "native_metric_name": s["native_metric_name"],
                "notes": s["notes"],
            })
    return summary_rows


def _md_hv_cell(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "N/A"
    if np.isnan(std):
        return f"{mean:.2f} +/- nan"
    return f"{mean:.2f} +/- {std:.2f}"


def _md_pareto_cell(mean_p: float) -> str:
    if np.isnan(mean_p):
        return "N/A"
    return f"{mean_p:.1f}"


def _md_native_cell(val: float, name: str) -> str:
    if not name:
        return "---"
    if np.isnan(val):
        return "N/A"
    return f"{val:.2f} ({name})"


def render_md_table(summary_rows: List[Dict[str, object]]) -> str:
    """Render the Markdown comparison table."""
    lines = [
        "| Baseline | n_seeds_ok | HV (mean +/- std) | mean |Pareto| | Native metric |",
        "|---|---:|---:|---:|---|",
    ]
    for s in summary_rows:
        lines.append(
            f"| {s['baseline']} "
            f"| {int(s['n_seeds_ok'])} "
            f"| {_md_hv_cell(s['mean_hv'], s['std_hv'])} "
            f"| {_md_pareto_cell(s['mean_n_pareto'])} "
            f"| {_md_native_cell(s['native_metric_value'], s['native_metric_name'])} |"
        )
    return "\n".join(lines) + "\n"


def _tex_hv_cell(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "N/A"
    if np.isnan(std):
        return f"${mean:.2f} \\pm \\mathrm{{nan}}$"
    return f"${mean:.2f} \\pm {std:.2f}$"


def _tex_pareto_cell(mean_p: float) -> str:
    if np.isnan(mean_p):
        return "N/A"
    return f"{mean_p:.1f}"


def _tex_native_cell(val: float, name: str) -> str:
    if not name:
        return "---"
    if np.isnan(val):
        return "N/A"
    safe_name = name.replace("_", "\\_")
    return f"${val:.2f}$ ({safe_name})"


def render_tex_table(summary_rows: List[Dict[str, object]]) -> str:
    """Render the LaTeX tabular block matching paper Table IV style."""
    lines = [
        "\\begin{tabular}{lrrrl}",
        "\\toprule",
        "Baseline & $n$ & HV (mean $\\pm$ std) & $|\\mathcal{P}|$ & Native metric \\\\",
        "\\midrule",
    ]
    for s in summary_rows:
        baseline_tex = s["baseline"].replace("_", "\\_")
        lines.append(
            f"{baseline_tex} & {int(s['n_seeds_ok'])} "
            f"& {_tex_hv_cell(s['mean_hv'], s['std_hv'])} "
            f"& {_tex_pareto_cell(s['mean_n_pareto'])} "
            f"& {_tex_native_cell(s['native_metric_value'], s['native_metric_name'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    runs_dir: Path = args.runs_dir
    out_dir: Path = args.out_dir if args.out_dir is not None else runs_dir
    config_path: Path = args.config

    if not config_path.exists():
        print(f"[error] manifest not found: {config_path}", file=sys.stderr)
        return 2

    manifest = load_manifest(config_path)
    baselines = list(manifest.get("baselines", []))
    seeds = list(manifest.get("seeds", []))
    if not baselines or not seeds:
        print(f"[error] manifest missing baselines or seeds: {config_path}",
              file=sys.stderr)
        return 2

    if not runs_dir.exists():
        print(f"[error] runs dir does not exist: {runs_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_cells(runs_dir, baselines, seeds)
    if not rows:
        print(f"[error] no cells produced from {runs_dir}", file=sys.stderr)
        return 1

    long_path = out_dir / "p11_long.csv"
    summary_path = out_dir / "hv_summary.csv"
    md_path = out_dir / "p11_table.md"
    tex_path = out_dir / "p11_table.tex"

    write_long_csv(rows, long_path)
    summary_rows = write_summary_csv(baselines, rows, summary_path)

    md_table = render_md_table(summary_rows)
    tex_table = render_tex_table(summary_rows)
    md_path.write_text(md_table)
    tex_path.write_text(tex_table)

    print(md_table, end="")
    print()
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_total = len(rows)
    print(f"cells ok / total : {n_ok} / {n_total}")
    print(f"[ok] wrote {long_path}", file=sys.stderr)
    print(f"[ok] wrote {summary_path}", file=sys.stderr)
    print(f"[ok] wrote {md_path}", file=sys.stderr)
    print(f"[ok] wrote {tex_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
