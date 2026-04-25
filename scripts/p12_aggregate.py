#!/usr/bin/env python3
"""Aggregate multi-tenant Nano-GRPO results into table + plots.

Reads ``runs/p12_multitenant_grpo/{condition}_seed{N}/`` directories and emits
a summary CSV, Markdown comparison table, and 2x2 contention figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SUMMARY_COLS: Tuple[str, ...] = (
    "condition", "seed",
    "grpo_elapsed_s", "grpo_energy_total_j", "grpo_energy_per_step_j",
    "grpo_step_total_mean_ms", "grpo_step_total_p50_ms", "grpo_step_total_p99_ms",
    "grpo_mem_peak_ram_mb", "grpo_torch_peak_alloc_mb", "grpo_gpu_util_mean_pct",
    "grpo_final_reward_mean", "grpo_oom",
    "override_fire_count", "override_fire_rate",
    "llm_inf_latency_mean_ms", "llm_inf_latency_p99_ms", "llm_inf_n_prompts",
    "perception_fps_actual", "perception_latency_mean_ms",
    "perception_latency_p99_ms", "perception_n_frames",
    "n_tegrastats_samples",
)

COMPARISON_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("GRPO step latency (mean ms)",      "grpo_step_total_mean_ms",    "%.3g"),
    ("GRPO step latency (p99 ms)",       "grpo_step_total_p99_ms",     "%.3g"),
    ("GRPO energy/step (J)",             "grpo_energy_per_step_j",     "%.3g"),
    ("GRPO energy total (J)",            "grpo_energy_total_j",        "%.3g"),
    ("GRPO mem peak (MB)",               "grpo_mem_peak_ram_mb",       "%.3g"),
    ("GRPO torch peak alloc (MB)",       "grpo_torch_peak_alloc_mb",   "%.3g"),
    ("GRPO GPU util mean (%)",           "grpo_gpu_util_mean_pct",     "%.3g"),
    ("Override fire count",              "override_fire_count",        "%.3g"),
    ("Override fire rate (/step)",       "override_fire_rate",         "%.3g"),
    ("LLM inference latency mean (ms)",  "llm_inf_latency_mean_ms",    "%.3g"),
    ("LLM inference latency p99 (ms)",   "llm_inf_latency_p99_ms",     "%.3g"),
    ("Perception fps actual",            "perception_fps_actual",      "%.3g"),
    ("Perception latency mean (ms)",     "perception_latency_mean_ms", "%.3g"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--conditions", type=str, default="with_critic,without_critic")
    return p.parse_args()


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except (pd.errors.EmptyDataError, ValueError):
        return None
    if df.empty:
        return None
    return df


def load_run(run_dir: Path) -> Optional[Dict[str, object]]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r") as f:
        summary = json.load(f)
    return {
        "dir": run_dir,
        "summary": summary,
        "perception": _safe_read_csv(run_dir / "perception.csv"),
        "llm_inf": _safe_read_csv(run_dir / "llm_inf.csv"),
        "per_step": _safe_read_csv(run_dir / "per_step_grpo.csv"),
    }


def _perception_fps(df: Optional[pd.DataFrame]) -> float:
    if df is None or len(df) < 2:
        return float("nan") if df is None else 0.0
    ts = df["timestamp_s"].to_numpy(dtype=float)
    span = float(ts.max()) - float(ts.min())
    return (len(df) - 1) / span if span > 0 else 0.0


def _stat(series: Optional[pd.Series], fn) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    return float(fn(series))


def summarize_run(run: Dict[str, object]) -> Dict[str, object]:
    s = run["summary"]
    grpo = s.get("grpo", {}) or {}
    override = s.get("override", {}) or {}
    step_total = grpo.get("step_total", {}) or {}

    perception = run["perception"]
    llm_inf = run["llm_inf"]
    perception_lat = perception["latency_ms"] if perception is not None else None
    llm_lat = llm_inf["latency_ms"] if llm_inf is not None else None

    return {
        "condition": s.get("condition", ""),
        "seed": int(s.get("seed", -1)),
        "grpo_elapsed_s": float(grpo.get("elapsed_s", float("nan"))),
        "grpo_energy_total_j": float(grpo.get("energy_total_j", float("nan"))),
        "grpo_energy_per_step_j": float(grpo.get("energy_per_step_j", float("nan"))),
        "grpo_step_total_mean_ms": float(step_total.get("mean_ms", float("nan"))),
        "grpo_step_total_p50_ms": float(step_total.get("p50_ms", float("nan"))),
        "grpo_step_total_p99_ms": float(step_total.get("p99_ms", float("nan"))),
        "grpo_mem_peak_ram_mb": float(grpo.get("mem_peak_ram_mb", float("nan"))),
        "grpo_torch_peak_alloc_mb": float(grpo.get("torch_peak_alloc_mb", float("nan"))),
        "grpo_gpu_util_mean_pct": float(grpo.get("gpu_util_mean_pct", float("nan"))),
        "grpo_final_reward_mean": float(grpo.get("final_reward_mean", float("nan"))),
        "grpo_oom": 1 if grpo.get("oom", False) else 0,
        "override_fire_count": int(override.get("fire_count", 0)),
        "override_fire_rate": float(override.get("fire_rate", 0.0)),
        "llm_inf_latency_mean_ms": _stat(llm_lat, np.mean),
        "llm_inf_latency_p99_ms": _stat(llm_lat, lambda x: np.percentile(x, 99)),
        "llm_inf_n_prompts": 0 if llm_inf is None else len(llm_inf),
        "perception_fps_actual": _perception_fps(perception),
        "perception_latency_mean_ms": _stat(perception_lat, np.mean),
        "perception_latency_p99_ms": _stat(perception_lat, lambda x: np.percentile(x, 99)),
        "perception_n_frames": 0 if perception is None else len(perception),
        "n_tegrastats_samples": int(s.get("n_tegrastats_samples", 0)),
    }


def build_summary_df(runs: List[Dict[str, object]]) -> pd.DataFrame:
    rows = [summarize_run(r) for r in runs]
    return pd.DataFrame(rows, columns=list(SUMMARY_COLS))


def format_cell(values: List[float], fmt: str = "%.3g") -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return "n/a"
    mean = float(np.mean(arr))
    if n == 1:
        return f"{fmt % mean} (n=1)"
    std = float(np.std(arr, ddof=1))
    return f"{fmt % mean} ± {fmt % std} (n={n})"


def make_comparison_table(
    df: pd.DataFrame, header_info: Dict[str, object], oom_dirs: List[str]
) -> str:
    lines: List[str] = []
    if oom_dirs:
        lines.append(f"**⚠ OOM detected in: {', '.join(oom_dirs)}**")
        lines.append("")
    lines.append("# Multi-tenant Nano-GRPO comparison")
    lines.append("")
    lines.append(f"- n_seeds: {header_info['n_seeds']}")
    lines.append(f"- n_steps: {header_info['n_steps']}")
    lines.append(f"- model: {header_info['model']}")
    lines.append(f"- host: {header_info['host']}")
    lines.append("")
    lines.append(
        "| Metric | with-critic (PPO) | without-critic (GRPO) "
        "| Δ (without − with) | Δ % |"
    )
    lines.append("|---|---|---|---|---|")

    with_df = df[df["condition"] == "with_critic"]
    wo_df = df[df["condition"] == "without_critic"]

    for label, key, fmt in COMPARISON_ROWS:
        with_vals = with_df[key].to_numpy(dtype=float).tolist()
        wo_vals = wo_df[key].to_numpy(dtype=float).tolist()
        with_str = format_cell(with_vals, fmt)
        wo_str = format_cell(wo_vals, fmt)

        with_mean = float(np.nanmean(with_vals)) if len(with_vals) else float("nan")
        wo_mean = float(np.nanmean(wo_vals)) if len(wo_vals) else float("nan")
        if np.isnan(with_mean) or np.isnan(wo_mean):
            delta_str = "n/a"
            pct_str = "n/a"
        else:
            delta = wo_mean - with_mean
            delta_str = fmt % delta
            pct_str = "n/a" if with_mean == 0 else f"{(delta / with_mean) * 100:+.3g}"
        lines.append(f"| {label} | {with_str} | {wo_str} | {delta_str} | {pct_str} |")

    lines.append("")
    lines.append(
        "Note: Δ%>0 means without-critic is HIGHER than with-critic; "
        "for latency/energy/memory, NEGATIVE Δ% favors GRPO (without-critic)."
    )
    return "\n".join(lines) + "\n"


def _per_step_mean_std(
    per_step_dfs: Dict[str, List[Optional[pd.DataFrame]]], col: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for cond, dfs in per_step_dfs.items():
        usable = [d for d in dfs if d is not None and col in d.columns and len(d) > 0]
        if not usable:
            continue
        n = min(len(d) for d in usable)
        stacked = np.stack([d[col].to_numpy(dtype=float)[:n] for d in usable], axis=0)
        steps = np.arange(n)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0, ddof=0) if stacked.shape[0] > 1 else np.zeros(n)
        out[cond] = (steps, mean, std)
    return out


def _cum_energy(
    per_step_dfs: Dict[str, List[Optional[pd.DataFrame]]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    need = {"total_ms", "vdd_gpu_mw", "vdd_cpu_mw"}
    for cond, dfs in per_step_dfs.items():
        cum_runs: List[np.ndarray] = []
        for d in dfs:
            if d is None or len(d) == 0 or not need.issubset(d.columns):
                continue
            tot_s = d["total_ms"].to_numpy(dtype=float) / 1000.0
            pwr_w = (d["vdd_gpu_mw"].to_numpy(dtype=float)
                     + d["vdd_cpu_mw"].to_numpy(dtype=float)) / 1000.0
            cum_runs.append(np.cumsum(tot_s * pwr_w))
        if not cum_runs:
            continue
        n = min(len(r) for r in cum_runs)
        stacked = np.stack([r[:n] for r in cum_runs], axis=0)
        steps = np.arange(n)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0, ddof=0) if stacked.shape[0] > 1 else np.zeros(n)
        out[cond] = (steps, mean, std)
    return out


def _empty(ax: plt.Axes) -> None:
    ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)


def plot_contention_curves(
    per_step_by_cond: Dict[str, List[Optional[pd.DataFrame]]],
    llm_inf_by_cond: Dict[str, List[Optional[pd.DataFrame]]],
    perception_by_cond: Dict[str, List[Optional[pd.DataFrame]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    colors = {"with_critic": "tab:blue", "without_critic": "tab:orange"}

    ax = axes[0, 0]
    lat = _per_step_mean_std(per_step_by_cond, "total_ms")
    if not lat:
        _empty(ax)
    else:
        for cond, (x, m, s) in lat.items():
            c = colors.get(cond)
            ax.plot(x, m, label=cond, color=c)
            ax.fill_between(x, m - s, m + s, alpha=0.2, color=c)
        ax.legend()
    ax.set_title("GRPO step latency under contention")
    ax.set_xlabel("step")
    ax.set_ylabel("ms")

    ax = axes[0, 1]
    energy = _cum_energy(per_step_by_cond)
    if not energy:
        _empty(ax)
    else:
        for cond, (x, m, s) in energy.items():
            c = colors.get(cond)
            ax.plot(x, m, label=cond, color=c)
            ax.fill_between(x, m - s, m + s, alpha=0.2, color=c)
        ax.legend()
    ax.set_title("GRPO cumulative energy")
    ax.set_xlabel("step")
    ax.set_ylabel("J (approx, GPU+CPU)")

    ax = axes[1, 0]
    plotted = False
    for cond, dfs in llm_inf_by_cond.items():
        usable = [
            d for d in dfs
            if d is not None and {"timestamp_s", "latency_ms"}.issubset(d.columns) and len(d) > 0
        ]
        if not usable:
            continue
        n = min(len(d) for d in usable)
        ts = np.median(
            np.stack([d["timestamp_s"].to_numpy(dtype=float)[:n] for d in usable], axis=0), axis=0,
        )
        lt = np.median(
            np.stack([d["latency_ms"].to_numpy(dtype=float)[:n] for d in usable], axis=0), axis=0,
        )
        ts = ts - ts[0]
        ax.plot(ts, lt, label=cond, color=colors.get(cond))
        plotted = True
    if not plotted:
        _empty(ax)
    else:
        ax.legend()
    ax.set_title("LLM inference latency under contention")
    ax.set_xlabel("elapsed s")
    ax.set_ylabel("ms")

    ax = axes[1, 1]
    plotted = False
    for cond, dfs in perception_by_cond.items():
        chunks = [d["latency_ms"].to_numpy(dtype=float)
                  for d in dfs if d is not None and "latency_ms" in d.columns and len(d) > 0]
        if not chunks:
            continue
        cat = np.sort(np.concatenate(chunks))
        cdf = np.arange(1, len(cat) + 1) / len(cat)
        ax.plot(cat, cdf, label=cond, color=colors.get(cond))
        plotted = True
    if not plotted:
        _empty(ax)
    else:
        ax.legend()
    ax.set_title("Perception latency CDF")
    ax.set_xlabel("ms")
    ax.set_ylabel("CDF")

    fig.tight_layout()
    fig.savefig(str(out_path) + ".png")
    fig.savefig(str(out_path) + ".pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir or in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    runs: List[Dict[str, object]] = []
    per_step_by_cond: Dict[str, List[Optional[pd.DataFrame]]] = {c: [] for c in conditions}
    llm_inf_by_cond: Dict[str, List[Optional[pd.DataFrame]]] = {c: [] for c in conditions}
    perception_by_cond: Dict[str, List[Optional[pd.DataFrame]]] = {c: [] for c in conditions}

    header_info: Dict[str, object] = {
        "n_seeds": 0, "n_steps": "?", "model": "?", "host": "?",
    }
    oom_dirs: List[str] = []
    seen_meta = False

    for cond in conditions:
        for seed in seeds:
            run_dir = in_dir / f"{cond}_seed{seed}"
            run = load_run(run_dir)
            if run is None:
                print(f"[aggregate] WARN: skipping missing run {run_dir}")
                continue
            runs.append(run)
            per_step_by_cond[cond].append(run["per_step"])
            llm_inf_by_cond[cond].append(run["llm_inf"])
            perception_by_cond[cond].append(run["perception"])
            if run["summary"].get("grpo", {}).get("oom", False):
                oom_dirs.append(run_dir.name)
            if not seen_meta:
                s = run["summary"]
                header_info = {
                    "n_seeds": len(seeds),
                    "n_steps": s.get("n_steps", "?"),
                    "model": s.get("model", "?"),
                    "host": s.get("host", "?"),
                }
                seen_meta = True

    df = build_summary_df(runs)
    summary_csv = out_dir / "summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"[aggregate] wrote {summary_csv}")

    table_md = make_comparison_table(df, header_info, oom_dirs)
    table_path = out_dir / "comparison_table.md"
    table_path.write_text(table_md)
    print(f"[aggregate] wrote {table_path}")

    fig_stem = out_dir / "contention_curves"
    plot_contention_curves(per_step_by_cond, llm_inf_by_cond, perception_by_cond, fig_stem)
    print(f"[aggregate] wrote {fig_stem}.png")
    print(f"[aggregate] wrote {fig_stem}.pdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
