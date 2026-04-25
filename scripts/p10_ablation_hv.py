"""P10 Item #2 - 4-row component-ablation HV table from Orin AGX sweep.

Consumes runs/p10_orin_ablation/ (60 JSONLs from
tetrarl/eval/configs/p10_orin_ablation.yaml: 4 ablations x 5 omegas x 3 seeds)
and emits:
- ablation_summary.csv: 4 rows (one per ablation) x metrics
  {n, mean_hv, std_hv, mean_step_lat_ms, mean_p99_lat_ms,
   mean_peak_memory, mean_energy_j, mean_n_steps, mean_n_episodes}
- ablation_long.csv: 60 rows (one per cell) for transparency
- ablation_table.md: human-readable Markdown rendering
- ablation_table.tex: LaTeX `tabular` block, ready to drop into the paper

HV is computed via tetrarl.eval.hv.compute_run_hv with the same 4-D
reference point used by W10 Nano (-0.1, -1.0, -0.15, -0.01).

Per-cell metrics:
- HV (compute_run_hv on the JSONL)
- mean step latency_ms (across all steps in the JSONL)
- p99 step latency_ms
- peak memory_util (max across all steps)
- mean energy_j per step
- n_steps total in the JSONL

Aggregation per ablation row: mean +/- std across the 15 cells (5 omega x 3 seed).

Usage:
    python scripts/p10_ablation_hv.py \\
        --runs-dir runs/p10_orin_ablation \\
        --out-dir runs/p10_orin_ablation
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tetrarl.eval.hv import compute_run_hv  # noqa: E402

DEFAULT_REF_POINT = np.array([-0.1, -1.0, -0.15, -0.01], dtype=np.float64)

ABLATIONS: list[str] = ["none", "preference_plane", "resource_manager", "override_layer"]
OMEGA_NAMES: list[str] = ["reward", "time", "memory", "energy", "center"]
SEEDS: list[int] = [0, 1, 2]

ABLATION_DISPLAY: dict[str, str] = {
    "none": "Full (TetraRL)",
    "preference_plane": "no-PP",
    "resource_manager": "no-RM",
    "override_layer": "no-OL",
}


def _jsonl_filename(ablation: str, omega_name: str, seed: int, omega_idx: int) -> str:
    return (
        f"{ablation}__preference_ppo__CartPole__{omega_name}"
        f"__seed{seed}__o{omega_idx}.jsonl"
    )


def _per_cell_metrics(jsonl_path: Path) -> dict[str, float]:
    """Compute per-cell metrics from a single JSONL: lat means/p99, peak mem, energy, n_steps."""
    latencies: list[float] = []
    mems: list[float] = []
    energies: list[float] = []
    episodes: set[int] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            latencies.append(float(rec["latency_ms"]))
            mems.append(float(rec["memory_util"]))
            energies.append(float(rec["energy_j"]))
            episodes.add(int(rec["episode"]))
    n_steps = len(latencies)
    if n_steps == 0:
        return {
            "mean_lat_ms": 0.0,
            "p99_lat_ms": 0.0,
            "peak_mem_util": 0.0,
            "mean_energy_j": 0.0,
            "n_steps": 0.0,
            "n_episodes": 0.0,
        }
    lat_arr = np.asarray(latencies, dtype=np.float64)
    return {
        "mean_lat_ms": float(np.mean(lat_arr)),
        "p99_lat_ms": float(np.percentile(lat_arr, 99.0)),
        "peak_mem_util": float(np.max(mems)),
        "mean_energy_j": float(np.mean(energies)),
        "n_steps": float(n_steps),
        "n_episodes": float(len(episodes)),
    }


def _collect_long_rows(runs_dir: Path, ref_point: np.ndarray) -> list[dict]:
    """Iterate the 4 x 5 x 3 grid and assemble per-cell records.

    Skips cells whose JSONL does not exist (warns to stderr) so the script
    is robust to partial sweeps.
    """
    rows: list[dict] = []
    for ablation in ABLATIONS:
        for omega_idx, omega_name in enumerate(OMEGA_NAMES):
            for seed in SEEDS:
                fname = _jsonl_filename(ablation, omega_name, seed, omega_idx)
                path = runs_dir / fname
                if not path.exists():
                    print(f"[ablation_hv] missing JSONL: {path}", file=sys.stderr)
                    continue
                hv = compute_run_hv(path, ref_point=ref_point)
                metrics = _per_cell_metrics(path)
                rows.append(
                    {
                        "ablation": ablation,
                        "omega_name": omega_name,
                        "omega_idx": int(omega_idx),
                        "seed": int(seed),
                        "hv": float(hv),
                        "mean_lat_ms": float(metrics["mean_lat_ms"]),
                        "p99_lat_ms": float(metrics["p99_lat_ms"]),
                        "peak_mem_util": float(metrics["peak_mem_util"]),
                        "mean_energy_j": float(metrics["mean_energy_j"]),
                        "n_steps": int(metrics["n_steps"]),
                        "n_episodes": int(metrics["n_episodes"]),
                    }
                )
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return float(mean), float(std)


def _aggregate_summary(rows: list[dict]) -> list[dict]:
    """Aggregate per-cell rows by ablation: mean +/- std across (omega x seed)."""
    by_abl: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_abl[r["ablation"]].append(r)

    out: list[dict] = []
    for ablation in ABLATIONS:
        cells = by_abl.get(ablation, [])
        hv_vals = [float(c["hv"]) for c in cells]
        lat_vals = [float(c["mean_lat_ms"]) for c in cells]
        p99_vals = [float(c["p99_lat_ms"]) for c in cells]
        mem_vals = [float(c["peak_mem_util"]) for c in cells]
        en_vals = [float(c["mean_energy_j"]) for c in cells]
        steps_vals = [float(c["n_steps"]) for c in cells]
        eps_vals = [float(c["n_episodes"]) for c in cells]

        mean_hv, std_hv = _mean_std(hv_vals)
        mean_lat, std_lat = _mean_std(lat_vals)
        mean_p99, std_p99 = _mean_std(p99_vals)
        mean_mem, std_mem = _mean_std(mem_vals)
        mean_en, std_en = _mean_std(en_vals)
        mean_steps, _ = _mean_std(steps_vals)
        mean_eps, _ = _mean_std(eps_vals)

        out.append(
            {
                "ablation": ablation,
                "n": int(len(cells)),
                "mean_hv": mean_hv,
                "std_hv": std_hv,
                "mean_step_lat_ms": mean_lat,
                "std_step_lat_ms": std_lat,
                "mean_p99_lat_ms": mean_p99,
                "std_p99_lat_ms": std_p99,
                "mean_peak_memory": mean_mem,
                "std_peak_memory": std_mem,
                "mean_energy_j": mean_en,
                "std_energy_j": std_en,
                "mean_n_steps": mean_steps,
                "mean_n_episodes": mean_eps,
            }
        )
    return out


def _write_long_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "ablation",
        "omega_name",
        "omega_idx",
        "seed",
        "hv",
        "mean_lat_ms",
        "p99_lat_ms",
        "peak_mem_util",
        "mean_energy_j",
        "n_steps",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def _write_summary_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "ablation",
        "n",
        "mean_hv",
        "std_hv",
        "mean_step_lat_ms",
        "std_step_lat_ms",
        "mean_p99_lat_ms",
        "std_p99_lat_ms",
        "mean_peak_memory",
        "std_peak_memory",
        "mean_energy_j",
        "std_energy_j",
        "mean_n_steps",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def _format_markdown_table(summary: list[dict]) -> str:
    """Render the 4-row ablation table as Markdown with bolded header."""
    header = (
        "| **Ablation** | **HV (mean +/- std)** | **Mean step latency (ms)** "
        "| **p99 latency (ms)** | **Peak memory (frac)** "
        "| **Mean energy/step (mJ)** |"
    )
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in summary:
        display = ABLATION_DISPLAY.get(r["ablation"], r["ablation"])
        # energy_j -> mJ for readability
        mean_en_mj = r["mean_energy_j"] * 1000.0
        std_en_mj = r["std_energy_j"] * 1000.0
        lines.append(
            f"| {display} "
            f"| {r['mean_hv']:.4e} +/- {r['std_hv']:.2e} "
            f"| {r['mean_step_lat_ms']:.3f} +/- {r['std_step_lat_ms']:.3f} "
            f"| {r['mean_p99_lat_ms']:.3f} +/- {r['std_p99_lat_ms']:.3f} "
            f"| {r['mean_peak_memory']:.4f} +/- {r['std_peak_memory']:.4f} "
            f"| {mean_en_mj:.3f} +/- {std_en_mj:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _format_latex_table(summary: list[dict]) -> str:
    """Render the 4-row ablation table as a booktabs LaTeX `tabular` block."""
    lines: list[str] = []
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append(
        "Ablation & HV (mean $\\pm$ std) & Mean step lat. (ms) "
        "& p99 lat. (ms) & Peak mem. (frac) & Mean energy/step (mJ) \\\\"
    )
    lines.append("\\midrule")
    for r in summary:
        display = ABLATION_DISPLAY.get(r["ablation"], r["ablation"])
        mean_en_mj = r["mean_energy_j"] * 1000.0
        std_en_mj = r["std_energy_j"] * 1000.0
        lines.append(
            f"{display} "
            f"& ${r['mean_hv']:.4e} \\pm {r['std_hv']:.2e}$ "
            f"& ${r['mean_step_lat_ms']:.3f} \\pm {r['std_step_lat_ms']:.3f}$ "
            f"& ${r['mean_p99_lat_ms']:.3f} \\pm {r['std_p99_lat_ms']:.3f}$ "
            f"& ${r['mean_peak_memory']:.4f} \\pm {r['std_peak_memory']:.4f}$ "
            f"& ${mean_en_mj:.3f} \\pm {std_en_mj:.3f}$ \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs/p10_orin_ablation"),
        help="Directory containing the 60 ablation JSONLs.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/p10_orin_ablation"),
        help="Directory for ablation_long.csv / ablation_summary.csv / "
        "ablation_table.md / ablation_table.tex.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_long_rows(runs_dir, DEFAULT_REF_POINT)
    if not rows:
        print(
            f"[ablation_hv] no JSONLs found under {runs_dir}; "
            "did the sweep run yet?",
            file=sys.stderr,
        )
        return 1

    summary = _aggregate_summary(rows)

    long_csv = out_dir / "ablation_long.csv"
    summary_csv = out_dir / "ablation_summary.csv"
    md_path = out_dir / "ablation_table.md"
    tex_path = out_dir / "ablation_table.tex"

    _write_long_csv(rows, long_csv)
    _write_summary_csv(summary, summary_csv)

    md = _format_markdown_table(summary)
    tex = _format_latex_table(summary)
    md_path.write_text(md, encoding="utf-8")
    tex_path.write_text(tex, encoding="utf-8")

    print(f"[ablation_hv] wrote {long_csv}")
    print(f"[ablation_hv] wrote {summary_csv}")
    print(f"[ablation_hv] wrote {md_path}")
    print(f"[ablation_hv] wrote {tex_path}")
    print()
    print(md, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
