#!/usr/bin/env python3
"""Aggregate P15 Phase 1 results (CartPole x Orin AGX, 75 cells).

Reads all summary.json files under runs/p15_phase1_orin_agx_cartpole/, then emits:
  - _phase1_summary.csv  : long-form, one row per cell (75 rows)
  - _phase1_summary.md   : 21-row x 5-metric table (active (algo,wrapper) pairs),
                           plus a SKIPPED-rows section for the 4 SKIPPED pairs.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs" / "p15_phase1_orin_agx_cartpole"
CSV_PATH = RUNS_DIR / "_phase1_summary.csv"
MD_PATH = RUNS_DIR / "_phase1_summary.md"

CSV_COLUMNS = [
    "algo",
    "wrapper",
    "seed",
    "status",
    "wall_time_s",
    "n_episodes",
    "framework_overhead_pct",
    "mean_p99_step_ms",
    "mean_energy_j",
    "peak_gpu_memory_mb",
    "peak_memory_mb",
    "final_reward",
    "time_to_converge_steps",
    "reason",
]

MD_METRICS = [
    "framework_overhead_pct",
    "mean_p99_step_ms",
    "mean_energy_j",
    "peak_memory_mb",
    "final_reward",
]


def _final_reward(curve: list[float] | None) -> float:
    if not curve:
        return 0.0
    tail = curve[-10:] if len(curve) >= 10 else curve
    return float(statistics.fmean(tail))


def _mean_std(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return float(clean[0]), 0.0
    return float(statistics.fmean(clean)), float(statistics.pstdev(clean))


def _fmt(mean: float, std: float, prec: int = 3) -> str:
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.{prec}f} ± {std:.{prec}f}"


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for sub in sorted(RUNS_DIR.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        sj = sub / "summary.json"
        if not sj.is_file():
            print(f"WARN: missing summary.json in {sub.name}")
            continue
        try:
            data = json.loads(sj.read_text())
        except Exception as e:
            print(f"ERROR parsing {sj}: {e}")
            continue
        status = data.get("status", "UNKNOWN")
        peak_gpu = data.get("peak_gpu_memory_mb")
        row = {
            "algo": data.get("algo", ""),
            "wrapper": data.get("wrapper", ""),
            "seed": data.get("seed", ""),
            "status": status,
            "wall_time_s": data.get("wall_time_s", ""),
            "n_episodes": data.get("n_episodes", ""),
            "framework_overhead_pct": data.get("framework_overhead_pct", ""),
            "mean_p99_step_ms": data.get("mean_p99_step_ms", ""),
            "mean_energy_j": data.get("mean_energy_j", ""),
            "peak_gpu_memory_mb": peak_gpu if peak_gpu is not None else "",
            # peak_memory_mb is an alias since real peak_memory isn't separately recorded
            "peak_memory_mb": peak_gpu if peak_gpu is not None else "",
            "final_reward": (
                _final_reward(data.get("cumulative_reward_curve"))
                if status == "COMPLETED"
                else 0.0
            ),
            "time_to_converge_steps": data.get("time_to_converge_steps", ""),
            "reason": data.get("reason", ""),
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: (r["algo"], r["wrapper"], r["seed"]))
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)


def aggregate_pairs(rows: list[dict]) -> tuple[list[tuple], list[tuple]]:
    pairs: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        pairs.setdefault((r["algo"], r["wrapper"]), []).append(r)

    active: list[tuple] = []  # (algo, wrapper, agg_dict)
    skipped: list[tuple] = []  # (algo, wrapper, reason)

    for (algo, wrapper), group in sorted(pairs.items()):
        statuses = {r["status"] for r in group}
        if statuses == {"SKIPPED"}:
            reasons = {r.get("reason", "") for r in group}
            skipped.append((algo, wrapper, "; ".join(sorted(r for r in reasons if r))))
            continue
        completed = [r for r in group if r["status"] == "COMPLETED"]
        if not completed:
            continue
        agg = {}
        # framework_overhead_pct, mean_p99_step_ms, mean_energy_j use raw fields
        for metric in ("framework_overhead_pct", "mean_p99_step_ms", "mean_energy_j"):
            vals = [float(r[metric]) for r in completed if r.get(metric) not in (None, "")]
            agg[metric] = _mean_std(vals)
        # peak_memory_mb (alias for peak_gpu_memory_mb)
        vals = [float(r["peak_memory_mb"]) for r in completed if r.get("peak_memory_mb") not in (None, "")]
        agg["peak_memory_mb"] = _mean_std(vals)
        # final_reward
        vals = [float(r["final_reward"]) for r in completed]
        agg["final_reward"] = _mean_std(vals)
        active.append((algo, wrapper, agg, len(completed)))
    return active, skipped


def write_md(active: list[tuple], skipped: list[tuple]) -> None:
    lines: list[str] = []
    lines.append("# P15 Phase 1 — CartPole × Orin AGX summary")
    lines.append("")
    lines.append(
        "Active (algo, wrapper) pairs: 21. Each cell shows mean ± std across 3 seeds. "
        "`peak_memory_mb` is an alias of `peak_gpu_memory_mb` (no separate host-RAM peak is recorded)."
    )
    lines.append("")
    header = ["algo", "wrapper", "n", *MD_METRICS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    precs = {
        "framework_overhead_pct": 2,
        "mean_p99_step_ms": 3,
        "mean_energy_j": 6,
        "peak_memory_mb": 2,
        "final_reward": 2,
    }
    for algo, wrapper, agg, n in active:
        cells = [algo, wrapper, str(n)]
        for m in MD_METRICS:
            mean, std = agg[m]
            cells.append(_fmt(mean, std, precs[m]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## SKIPPED rows (compatibility gate fired before training)")
    lines.append("")
    lines.append("| algo | wrapper | reason |")
    lines.append("|---|---|---|")
    for algo, wrapper, reason in skipped:
        lines.append(f"| {algo} | {wrapper} | {reason} |")
    lines.append("")
    MD_PATH.write_text("\n".join(lines))


def main() -> None:
    rows = load_rows()
    write_csv(rows)
    active, skipped = aggregate_pairs(rows)
    write_md(active, skipped)
    print(
        f"OK: wrote _phase1_summary.csv ({len(rows)} rows) and _phase1_summary.md"
    )


if __name__ == "__main__":
    main()
