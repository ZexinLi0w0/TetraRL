"""Week 10 Nano: HV summary + bar chart + Pareto scatter.

Consumes the runs directory produced by ``tetrarl.eval.runner`` against
``tetrarl/eval/configs/w10_nano_matrix.yaml`` (60 JSONLs: 2 agents x 2 envs
x 5 omegas x 3 seeds), computes the per-run hypervolume scalar via
:func:`tetrarl.eval.hv.compute_run_hv`, and emits three artefacts:

* ``hv_summary.csv`` -- per (agent, env) row plus per-run long-form
  ``hv_long.csv`` (one row per (agent, env, omega_idx, seed)).
* ``hv_bar.png`` / ``.svg`` -- grouped bar chart, x = agent, hue = env, y = mean
  HV across (omega x seed). Error bar = std.
* ``pareto_scatter.png`` / ``.svg`` -- per-run scatter on the
  (mean_latency_ms, mean_energy_j) plane, one point per
  (agent, env, omega_idx, seed), coloured by agent and shaped by env.

Reference point matches the W10 Orin convention (4-D, after the
"all higher = better" sign-flip on latency / memory / energy):
``(-0.1, -1.0, -0.15, -0.01)``. See ``docs/week10_full_eval_orin.md`` Section 3.
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

import matplotlib

matplotlib.use("Agg")  # noqa: E402  non-interactive backend before pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tetrarl.eval.hv import compute_run_hv  # noqa: E402
from tetrarl.eval.hypervolume import pareto_filter  # noqa: E402

DEFAULT_REF_POINT = "-0.1,-1.0,-0.15,-0.01"


def _parse_ref_point(s: str) -> np.ndarray:
    parts = [tok.strip() for tok in s.split(",") if tok.strip()]
    if len(parts) != 4:
        raise ValueError(
            f"--ref-point expected 4 comma-separated floats, got {len(parts)}: {s!r}"
        )
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def _load_matrix(yaml_path: Path) -> list[dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    cfgs = doc.get("configs") or []
    if not isinstance(cfgs, list):
        raise ValueError(
            f"matrix YAML {yaml_path} must contain a 'configs' list "
            f"(got {type(cfgs).__name__})"
        )
    return list(cfgs)


def _per_run_records(
    configs: list[dict], runs_dir: Path, ref_point: np.ndarray
) -> list[dict]:
    rows: list[dict] = []
    for cfg in configs:
        extra = dict(cfg.get("extra") or {})
        fname = extra.get("jsonl_name")
        if not fname:
            continue
        path = runs_dir / fname
        if not path.exists():
            continue
        hv = compute_run_hv(path, ref_point=ref_point)
        n_pareto = _count_pareto_points(path)
        row = {
            "agent": str(cfg["agent_type"]),
            "env": str(cfg["env_name"]),
            "seed": int(cfg["seed"]),
            "omega_idx": int(extra.get("omega_idx", 0)),
            "omega_name": str(extra.get("omega_name", "")),
            "hv": float(hv),
            "n_pareto_points": int(n_pareto),
        }
        rows.append(row)
    return rows


def _count_pareto_points(jsonl_path: Path) -> int:
    """Count Pareto-front points used in HV computation for a single run."""
    per_episode: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            per_episode[int(rec["episode"])].append(
                (
                    float(rec["reward"]),
                    float(rec["latency_ms"]),
                    float(rec["memory_util"]),
                    float(rec["energy_j"]),
                )
            )
    if not per_episode:
        return 0
    points: list[list[float]] = []
    for ep in sorted(per_episode.keys()):
        steps = np.asarray(per_episode[ep], dtype=np.float64)
        points.append(
            [
                float(steps[:, 0].mean()),
                -float(steps[:, 1].mean()),
                -float(steps[:, 2].mean()),
                -float(steps[:, 3].mean()),
            ]
        )
    return int(len(pareto_filter(np.asarray(points, dtype=np.float64))))


def _per_run_means(jsonl_path: Path) -> tuple[float, float, float, float]:
    """Average reward / latency_ms / memory_util / energy_j across all steps."""
    rewards: list[float] = []
    latencies: list[float] = []
    mems: list[float] = []
    energies: list[float] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rewards.append(float(rec["reward"]))
            latencies.append(float(rec["latency_ms"]))
            mems.append(float(rec["memory_util"]))
            energies.append(float(rec["energy_j"]))
    if not rewards:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(rewards)),
        float(np.mean(latencies)),
        float(np.mean(mems)),
        float(np.mean(energies)),
    )


def _aggregate_summary(rows: list[dict]) -> list[dict]:
    """Aggregate per-run rows by (agent, env): mean HV +/- std across (omega x seed)."""
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_cell_pareto: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        key = (r["agent"], r["env"])
        by_cell[key].append(float(r["hv"]))
        by_cell_pareto[key].append(int(r["n_pareto_points"]))
    out: list[dict] = []
    for (agent, env), vals in sorted(by_cell.items()):
        mean = statistics.fmean(vals) if vals else 0.0
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        n = len(vals)
        out.append(
            {
                "agent": agent,
                "env": env,
                "n": int(n),
                "mean_hv": float(mean),
                "std_hv": float(std),
                "mean_n_pareto_points": float(
                    statistics.fmean(by_cell_pareto[(agent, env)])
                    if by_cell_pareto[(agent, env)]
                    else 0.0
                ),
            }
        )
    return out


def _write_long_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "agent",
        "env",
        "seed",
        "omega_idx",
        "omega_name",
        "hv",
        "n_pareto_points",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def _write_summary_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "agent",
        "env",
        "n",
        "mean_hv",
        "std_hv",
        "mean_n_pareto_points",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def _write_bar_chart(summary: list[dict], out_dir: Path) -> tuple[Path, Path]:
    """Grouped bar chart, x = agent, hue = env, y = mean HV +/- std."""
    agents = sorted({r["agent"] for r in summary})
    envs = sorted({r["env"] for r in summary})
    n_agents = len(agents)
    n_envs = len(envs)
    width = 0.8 / max(1, n_envs)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for j, env in enumerate(envs):
        means: list[float] = []
        stds: list[float] = []
        for ag in agents:
            cell = next(
                (r for r in summary if r["agent"] == ag and r["env"] == env), None
            )
            means.append(float(cell["mean_hv"]) if cell else 0.0)
            stds.append(float(cell["std_hv"]) if cell else 0.0)
        x = [i + (j - (n_envs - 1) / 2.0) * width for i in range(n_agents)]
        ax.bar(
            x,
            means,
            width=width,
            yerr=stds,
            label=env,
            color=colours[j % len(colours)],
            edgecolor="black",
            capsize=4,
            linewidth=0.8,
        )
    ax.set_xticks(list(range(n_agents)))
    ax.set_xticklabels(agents, rotation=10, ha="right")
    ax.set_ylabel("Hypervolume (mean +/- std across 5 omega x 3 seeds, n=15)")
    ax.set_title("W10 Nano HV: agent x env (60-cell matrix)")
    ax.legend(title="env", loc="best", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    png = out_dir / "hv_bar.png"
    svg = out_dir / "hv_bar.svg"
    fig.savefig(png, dpi=150)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def _write_pareto_scatter(
    rows: list[dict], runs_dir: Path, configs: list[dict], out_dir: Path
) -> tuple[Path, Path]:
    """Per-run scatter on (mean_latency_ms, mean_energy_j); colour=agent, marker=env."""
    points: list[tuple[str, str, float, float]] = []  # (agent, env, lat, en)
    for cfg in configs:
        extra = dict(cfg.get("extra") or {})
        fname = extra.get("jsonl_name")
        if not fname:
            continue
        path = runs_dir / fname
        if not path.exists():
            continue
        _, mean_lat, _, mean_en = _per_run_means(path)
        points.append((str(cfg["agent_type"]), str(cfg["env_name"]), mean_lat, mean_en))

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    if not points:
        ax.text(0.5, 0.5, "no runs found", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        png = out_dir / "pareto_scatter.png"
        svg = out_dir / "pareto_scatter.svg"
        fig.savefig(png, dpi=150)
        fig.savefig(svg)
        plt.close(fig)
        return png, svg

    agents = sorted({p[0] for p in points})
    envs = sorted({p[1] for p in points})
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D", "v"]

    for ai, ag in enumerate(agents):
        for ei, env in enumerate(envs):
            xs = [p[2] for p in points if p[0] == ag and p[1] == env]
            ys = [p[3] for p in points if p[0] == ag and p[1] == env]
            if not xs:
                continue
            ax.scatter(
                xs,
                ys,
                color=palette[ai % len(palette)],
                marker=markers[ei % len(markers)],
                label=f"{ag} | {env}",
                edgecolors="black",
                linewidths=0.5,
                s=42,
                alpha=0.8,
            )
    ax.set_xlabel("mean latency_ms (per-run avg over 200 episodes)")
    ax.set_ylabel("mean energy_j (per-run avg over 200 episodes)")
    ax.set_title("W10 Nano per-run latency vs energy (60 runs)")
    ax.grid(linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    png = out_dir / "pareto_scatter.png"
    svg = out_dir / "pareto_scatter.svg"
    fig.savefig(png, dpi=150)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix-yaml", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ref-point", type=str, default=DEFAULT_REF_POINT)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    ref = _parse_ref_point(str(args.ref_point))
    matrix = _load_matrix(Path(args.matrix_yaml))
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _per_run_records(matrix, runs_dir, ref)
    if not rows:
        print(
            f"[hv] no JSONL runs found under {runs_dir}; "
            "did the matrix run yet?",
            file=sys.stderr,
        )
        return 1

    summary = _aggregate_summary(rows)
    long_csv = out_dir / "hv_long.csv"
    summary_csv = out_dir / "hv_summary.csv"
    _write_long_csv(rows, long_csv)
    _write_summary_csv(summary, summary_csv)
    bar_png, bar_svg = _write_bar_chart(summary, out_dir)
    sc_png, sc_svg = _write_pareto_scatter(rows, runs_dir, matrix, out_dir)

    print(f"[hv] wrote {long_csv}")
    print(f"[hv] wrote {summary_csv}")
    print(f"[hv] wrote {bar_png}")
    print(f"[hv] wrote {bar_svg}")
    print(f"[hv] wrote {sc_png}")
    print(f"[hv] wrote {sc_svg}")
    for r in summary:
        print(
            f"[hv] agent={r['agent']:<22} env={r['env']:<22} "
            f"n={r['n']:>2} mean_hv={r['mean_hv']:.6e} std={r['std_hv']:.3e}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
