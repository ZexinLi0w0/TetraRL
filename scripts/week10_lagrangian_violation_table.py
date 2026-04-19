"""Week 10 Task 5: Lagrangian-only vs Lagrangian+override violation table.

TetraRL combines a Lagrangian dual update (``tetrarl/morl/native/lagrangian.py``)
with a hardware override layer (``tetrarl/morl/native/override.py``). The W10
spec asks for a head-to-head comparison: how often does each variant violate
the per-step latency / memory / cumulative-energy thresholds?

This script consumes a matrix YAML (the same shape produced by
``scripts/week10_make_matrix_yaml.py``) plus a directory of pre-recorded
JSONL eval runs, partitions runs into the two variants by their ``ablation``
field, computes per-run violation rates, and emits both a Markdown summary
table and a CSV companion.

Variant mapping (matches the runner's ablation taxonomy):
    * ``ablation == "override_layer"`` -> override layer ablated -> ``override_off``
      (rendered as "Lagrangian only (no override)" in the Markdown table).
    * any other ablation tag (typically ``"none"``) -> ``override_on``
      (rendered as "Lagrangian + override").

Per-step violation booleans:
    * latency: ``latency_ms > latency_threshold_ms``
    * memory:  ``memory_util > memory_threshold``
    * energy:  ``cumsum(energy_j) > energy_budget_j`` (cumulative budget rule)

Per-run violation rate = mean over steps of (any of the three booleans).
Per-variant aggregates are mean / std / n_runs across runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable, Optional

import yaml

VARIANT_OFF = "override_off"
VARIANT_ON = "override_on"

_MD_LABEL = {
    VARIANT_OFF: "Lagrangian only (no override)",
    VARIANT_ON: "Lagrangian + override",
}

# Render order: off first (the "bad" variant), then on (the "good" variant).
_VARIANT_ORDER: tuple[str, ...] = (VARIANT_OFF, VARIANT_ON)


def _variant_for_ablation(ablation: str) -> str:
    """Map an ablation tag to its violation-table variant key."""
    return VARIANT_OFF if str(ablation).strip() == "override_layer" else VARIANT_ON


def _jsonl_name_for(entry: dict) -> str:
    """Return the JSONL filename associated with a matrix entry.

    Honours an explicit ``extra.jsonl_name`` override when present so that
    callers can pin a specific file; otherwise falls back to the canonical
    ``<ablation>__<agent>__seed<N>.jsonl`` naming convention used by the
    eval runner / the W10 matrix sweep.
    """
    extra = entry.get("extra") or {}
    if isinstance(extra, dict) and extra.get("jsonl_name"):
        return str(extra["jsonl_name"])
    ablation = str(entry.get("ablation", "none"))
    agent = str(entry.get("agent_type", "unknown"))
    seed = int(entry.get("seed", 0))
    return f"{ablation}__{agent}__seed{seed}.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    """Read a per-step JSONL file into a list of dict records."""
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def compute_run_violation_rate(
    records: Iterable[dict],
    *,
    latency_threshold_ms: float,
    memory_threshold: float,
    energy_budget_j: float,
) -> float:
    """Return the fraction of steps that violate at least one threshold."""
    recs = list(records)
    if not recs:
        return 0.0
    cum_energy = 0.0
    n_violations = 0
    for rec in recs:
        latency = float(rec.get("latency_ms", 0.0))
        memory = float(rec.get("memory_util", 0.0))
        energy = float(rec.get("energy_j", 0.0))
        cum_energy += energy
        lat_v = latency > latency_threshold_ms
        mem_v = memory > memory_threshold
        eng_v = cum_energy > energy_budget_j
        if lat_v or mem_v or eng_v:
            n_violations += 1
    return n_violations / len(recs)


def _load_matrix(matrix_yaml: Path) -> list[dict]:
    """Load the matrix YAML and return its ``configs`` list."""
    with open(matrix_yaml, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    configs = doc.get("configs", [])
    if not isinstance(configs, list):
        raise ValueError(f"matrix YAML at {matrix_yaml} has no list under 'configs'")
    return [dict(c) for c in configs]


def _aggregate(rates: list[float]) -> tuple[float, float, int]:
    """Return (mean, std, n) over a list of per-run violation rates."""
    n = len(rates)
    if n == 0:
        return 0.0, 0.0, 0
    mean = statistics.fmean(rates)
    std = statistics.stdev(rates) if n >= 2 else 0.0
    return float(mean), float(std), n


def build_violation_table(
    *,
    matrix_yaml: Path,
    runs_dir: Path,
    latency_threshold_ms: float,
    memory_threshold: float,
    energy_budget_j: float,
) -> dict[str, tuple[float, float, int]]:
    """Return ``{variant: (mean_rate, std_rate, n_runs)}`` for both variants."""
    entries = _load_matrix(matrix_yaml)
    per_variant_rates: dict[str, list[float]] = {VARIANT_OFF: [], VARIANT_ON: []}
    for entry in entries:
        variant = _variant_for_ablation(entry.get("ablation", "none"))
        jsonl_path = Path(runs_dir) / _jsonl_name_for(entry)
        if not jsonl_path.exists():
            # Skip silently — partial sweeps are common during incremental
            # data collection. Aggregates honour whatever runs are present.
            continue
        records = _read_jsonl(jsonl_path)
        rate = compute_run_violation_rate(
            records,
            latency_threshold_ms=latency_threshold_ms,
            memory_threshold=memory_threshold,
            energy_budget_j=energy_budget_j,
        )
        per_variant_rates[variant].append(rate)
    return {v: _aggregate(per_variant_rates[v]) for v in _VARIANT_ORDER}


def _format_md(stats: dict[str, tuple[float, float, int]]) -> str:
    """Render the aggregated stats as the Markdown summary table."""
    lines = [
        "# Lagrangian Violation Table",
        "",
        "| Variant                              | Violation Rate | Std    | N Runs |",
        "| ------------------------------------ | -------------- | ------ | ------ |",
    ]
    for v in _VARIANT_ORDER:
        mean, std, n = stats[v]
        label = _MD_LABEL[v]
        lines.append(
            f"| {label:<36} | {mean:>14.3f} | {std:>6.3f} | {n:>6d} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_csv(stats: dict[str, tuple[float, float, int]], out_csv: Path) -> None:
    """Write the long-form CSV (one row per variant)."""
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "violation_rate", "std", "n_runs"])
        for v in _VARIANT_ORDER:
            mean, std, n = stats[v]
            writer.writerow([v, f"{mean:.6f}", f"{std:.6f}", str(n)])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the W10 Lagrangian-vs-override violation table.",
    )
    parser.add_argument(
        "--matrix-yaml",
        type=Path,
        required=True,
        help="Path to the matrix YAML manifest (configs: list).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing per-run JSONL files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where the .md and .csv outputs are written.",
    )
    parser.add_argument(
        "--latency-threshold-ms",
        type=float,
        required=True,
        help="Per-step latency cap; latency_ms > threshold counts as a violation.",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        required=True,
        help="Per-step memory cap (memory_util in [0, 1]).",
    )
    parser.add_argument(
        "--energy-budget-j",
        type=float,
        required=True,
        help="Cumulative energy budget; cumsum(energy_j) > budget counts as a violation.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = build_violation_table(
        matrix_yaml=args.matrix_yaml,
        runs_dir=args.runs_dir,
        latency_threshold_ms=float(args.latency_threshold_ms),
        memory_threshold=float(args.memory_threshold),
        energy_budget_j=float(args.energy_budget_j),
    )

    md_path = out_dir / "lagrangian_violation_table.md"
    csv_path = out_dir / "lagrangian_violation_table.csv"
    md_path.write_text(_format_md(stats), encoding="utf-8")
    _write_csv(stats, csv_path)

    print(f"[lagrangian-table] wrote {md_path}")
    print(f"[lagrangian-table] wrote {csv_path}")
    for v in _VARIANT_ORDER:
        mean, std, n = stats[v]
        print(f"  {v}: mean={mean:.4f} std={std:.4f} n_runs={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
