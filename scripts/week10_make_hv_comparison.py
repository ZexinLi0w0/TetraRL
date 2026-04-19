"""Week 10 Task 2c: HV bar-chart + Welch t-test comparison across agents.

Consumes a directory of ``<ablation>__<agent>__seedN[__oI].jsonl`` files
(produced by :class:`tetrarl.eval.runner.EvalRunner`) plus the matrix
YAML used to drive them, computes the per-run hypervolume scalar via
:func:`tetrarl.eval.hv.compute_run_hv`, runs Welch two-sample t-tests
between TetraRL Native (``preference_ppo`` by default) and each other
baseline, and writes four artefacts to ``--out-dir``:

  * ``hv_comparison.png`` — matplotlib bar chart, x = agent_type, y =
    mean HV across (seed × omega), error bar = std. Saved at 150 dpi.
  * ``hv_comparison.svg`` — vector copy of the same chart.
  * ``hv_comparison.md`` — Markdown table with columns
    ``Method | mean HV | std | n | p-value vs TetraRL``. The TetraRL
    row gets ``-`` in the p-value column (no self-comparison).
  * ``hv_comparison.csv`` — long-form: ``agent, env, seed, omega_idx,
    hv``. One row per JSONL run.

The matrix YAML provides the manifest: each entry's
``extra.jsonl_name`` (preferred) or the canonical name
``<ablation>__<agent>__seed<N>.jsonl`` (with a ``__o<idx>`` suffix when
multiple omegas share an (agent, seed) pair) maps to a JSONL under
``--runs-dir``.
"""
from __future__ import annotations

import argparse
import csv
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

from tetrarl.eval.hv import compute_run_hv, welch_pvalue  # noqa: E402


def _parse_ref_point(s: str) -> np.ndarray:
    """Parse a 4-comma-separated string into a (4,) float64 array."""
    parts = [tok.strip() for tok in s.split(",") if tok.strip()]
    if len(parts) != 4:
        raise ValueError(
            f"--ref-point expected 4 comma-separated floats, got {len(parts)}: {s!r}"
        )
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def _load_matrix(yaml_path: Path) -> list[dict]:
    """Load the matrix YAML and return the ``configs`` list."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    configs = doc.get("configs") or []
    if not isinstance(configs, list):
        raise ValueError(
            f"matrix YAML {yaml_path} must contain a 'configs' list (got {type(configs).__name__})"
        )
    return list(configs)


def _canonical_name(ablation: str, agent: str, seed: int,
                    omega_idx: Optional[int] = None) -> str:
    """Reproduce the runner's per-run JSONL name with optional omega suffix."""
    base = f"{ablation}__{agent}__seed{int(seed)}"
    if omega_idx is not None:
        base = f"{base}__o{int(omega_idx)}"
    return base + ".jsonl"


def _resolve_jsonl_name(cfg: dict, multi_omega_keys: set[tuple[str, str, int]]) -> str:
    """Pick the JSONL filename for ``cfg``.

    Honour ``extra.jsonl_name`` when present; otherwise build the
    canonical name. When multiple omegas share the same
    (ablation, agent, seed) triple, append ``__o<idx>`` so each entry
    has a distinct filename.
    """
    extra = dict(cfg.get("extra") or {})
    explicit = extra.get("jsonl_name")
    if explicit:
        return str(explicit)
    ablation = str(cfg.get("ablation", "none"))
    agent = str(cfg["agent_type"])
    seed = int(cfg["seed"])
    key = (ablation, agent, seed)
    omega_idx = extra.get("omega_idx")
    if omega_idx is None and key in multi_omega_keys:
        # Caller had multi-omega entries but didn't supply an index; we
        # cannot disambiguate, so fall back to the canonical no-suffix
        # name (will likely collide — that's the runner's problem).
        return _canonical_name(ablation, agent, seed)
    if omega_idx is not None:
        return _canonical_name(ablation, agent, seed, omega_idx=int(omega_idx))
    return _canonical_name(ablation, agent, seed)


def _compute_records(configs: list[dict], runs_dir: Path,
                     ref_point: np.ndarray) -> list[dict]:
    """Return a long-form list of dicts: one per matrix entry with HV."""
    # Detect (ablation, agent, seed) triples that appear with multiple
    # omegas — used to decide whether the canonical name needs an
    # __o<idx> suffix.
    bucket: dict[tuple[str, str, int], int] = defaultdict(int)
    for cfg in configs:
        bucket[(str(cfg.get("ablation", "none")), str(cfg["agent_type"]),
                int(cfg["seed"]))] += 1
    multi_omega = {k for k, n in bucket.items() if n > 1}

    rows: list[dict] = []
    for cfg in configs:
        fname = _resolve_jsonl_name(cfg, multi_omega)
        path = runs_dir / fname
        if not path.exists():
            continue
        hv = compute_run_hv(path, ref_point=ref_point)
        extra = dict(cfg.get("extra") or {})
        omega_idx = extra.get("omega_idx", 0)
        rows.append({
            "agent": str(cfg["agent_type"]),
            "env": str(cfg["env_name"]),
            "seed": int(cfg["seed"]),
            "omega_idx": int(omega_idx),
            "hv": float(hv),
        })
    return rows


def _aggregate_per_agent(rows: list[dict]) -> dict[str, list[float]]:
    """Bucket per-run HV scalars by agent for downstream stats."""
    per_agent: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        per_agent[r["agent"]].append(float(r["hv"]))
    return dict(per_agent)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return float(mean), float(std)


def _write_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = ["agent", "env", "seed", "omega_idx", "hv"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})


def _ordered_agents(per_agent: dict[str, list[float]],
                    tetrarl_agent: str) -> list[str]:
    """TetraRL Native first, then remaining agents alphabetically."""
    others = sorted(a for a in per_agent.keys() if a != tetrarl_agent)
    if tetrarl_agent in per_agent:
        return [tetrarl_agent, *others]
    return others


def _write_markdown(per_agent: dict[str, list[float]],
                    tetrarl_agent: str, out_path: Path) -> None:
    """Emit the Method | mean HV | std | n | p-value vs TetraRL table."""
    agents = _ordered_agents(per_agent, tetrarl_agent)
    tetra_vals = np.asarray(per_agent.get(tetrarl_agent, []), dtype=np.float64)
    lines: list[str] = []
    lines.append(f"# HV comparison vs TetraRL ({tetrarl_agent})")
    lines.append("")
    lines.append(f"| Method | mean HV | std | n | p-value vs {tetrarl_agent} |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for ag in agents:
        vals = per_agent.get(ag, [])
        mean, std = _mean_std(vals)
        n = len(vals)
        if ag == tetrarl_agent:
            p_str = "-"
        elif tetra_vals.size == 0 or n == 0:
            p_str = "n/a"
        else:
            p = welch_pvalue(tetra_vals, np.asarray(vals, dtype=np.float64))
            p_str = f"{p:.4f}"
        lines.append(f"| {ag} | {mean:.6f} | {std:.6f} | {n} | {p_str} |")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_chart(per_agent: dict[str, list[float]], tetrarl_agent: str,
                 out_dir: Path) -> tuple[Path, Path]:
    agents = _ordered_agents(per_agent, tetrarl_agent)
    means: list[float] = []
    stds: list[float] = []
    for ag in agents:
        m, s = _mean_std(per_agent.get(ag, []))
        means.append(m)
        stds.append(s)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = list(range(len(agents)))
    colours = ["#000000" if a == tetrarl_agent else "#56B4E9" for a in agents]
    ax.bar(x, means, yerr=stds, color=colours, edgecolor="black",
           capsize=4, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")
    ax.set_ylabel("Hypervolume (mean ± std across seeds × omegas)")
    ax.set_title(f"HV comparison: {tetrarl_agent} vs baselines")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    png_path = out_dir / "hv_comparison.png"
    svg_path = out_dir / "hv_comparison.svg"
    fig.savefig(png_path, dpi=150)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def _print_summary(per_agent: dict[str, list[float]], tetrarl_agent: str) -> None:
    agents = _ordered_agents(per_agent, tetrarl_agent)
    print(f"[hv] TetraRL Native = {tetrarl_agent}")
    for ag in agents:
        vals = per_agent.get(ag, [])
        mean, _ = _mean_std(vals)
        marker = " (TetraRL)" if ag == tetrarl_agent else ""
        print(f"[hv] agent={ag}{marker} mean HV={mean:.6f} n={len(vals)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Week 10 HV bar chart + Welch comparison artefacts.",
    )
    parser.add_argument(
        "--matrix-yaml",
        type=Path,
        required=True,
        help="Matrix YAML used to drive tetrarl.eval.runner (the manifest).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing the per-run <ablation>__<agent>__seedN[__oI].jsonl files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination dir for hv_comparison.{png,svg,md,csv}.",
    )
    parser.add_argument(
        "--ref-point",
        type=str,
        required=True,
        help=(
            "Comma-separated 4 floats for the HV reference point in the order "
            "(reward, -latency, -memory, -energy)."
        ),
    )
    parser.add_argument(
        "--tetrarl-agent",
        type=str,
        default="preference_ppo",
        help="agent_type label treated as TetraRL Native (default: preference_ppo).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ref_point = _parse_ref_point(str(args.ref_point))
    matrix_yaml = Path(args.matrix_yaml)
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = _load_matrix(matrix_yaml)
    rows = _compute_records(configs, runs_dir, ref_point)
    if not rows:
        print(f"[hv] no runs matched matrix YAML {matrix_yaml} under {runs_dir}",
              file=sys.stderr)
        return 1

    per_agent = _aggregate_per_agent(rows)
    tetrarl_agent = str(args.tetrarl_agent)

    csv_path = out_dir / "hv_comparison.csv"
    md_path = out_dir / "hv_comparison.md"
    _write_csv(rows, csv_path)
    _write_markdown(per_agent, tetrarl_agent, md_path)
    png_path, svg_path = _write_chart(per_agent, tetrarl_agent, out_dir)

    _print_summary(per_agent, tetrarl_agent)
    print(f"[hv] wrote {csv_path}")
    print(f"[hv] wrote {md_path}")
    print(f"[hv] wrote {png_path}")
    print(f"[hv] wrote {svg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
