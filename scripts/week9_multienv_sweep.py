"""Week 9 deliverable: multi-env scaling sweep (3 n_envs × 2 dvfs_modes).

Drives the W9 multi-env scaling matrix through :class:`tetrarl.eval.runner.EvalRunner`:

    n_envs   ∈ {1, 2, 4}
    dvfs_mode ∈ {"fixed_max", "userspace_with_arbiter"}

= 6 configs total, all using ``agent_type="preference_ppo"``,
``ablation="none"``. The ``dvfs_mode`` field is NOT a first-class
:class:`EvalConfig` slot (it only matters on physical Orin where the
sysfs DVFS controller is wired up), so it lives in ``cfg.extra``.

After the sweep completes, prints a Markdown table with one row per
config to stdout AND writes it to ``{out_dir}/multienv_summary.md``.
The ratio column ``tail_p99_ratio_vs_nenvs1_same_dvfs`` is the W9-spec
validation knob: tail-p99 latency degradation should be < 3× compared
to the n_envs=1 baseline at the same dvfs_mode when DVFS is active.

Importable: callers (incl. tests) can use :func:`build_sweep_configs`
to construct the 6-config matrix without executing it, and
:func:`main` is the CLI entrypoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

N_ENVS_GRID: tuple[int, ...] = (1, 2, 4)
DVFS_MODES: tuple[str, ...] = ("fixed_max", "userspace_with_arbiter")


def build_sweep_configs(
    out_dir: Path,
    n_episodes: int,
    seed: int,
    platform: str,
    env_name: str,
):
    """Return the 6 :class:`EvalConfig` rows for the W9 multi-env matrix.

    Each combination of (n_envs, dvfs_mode) appears exactly once.
    ``dvfs_mode`` is encoded in ``cfg.extra["dvfs_mode"]``.
    """
    # Lazy import keeps this module importable without pulling gymnasium.
    from tetrarl.eval.runner import EvalConfig

    configs = []
    for n_envs in N_ENVS_GRID:
        for dvfs_mode in DVFS_MODES:
            cfg = EvalConfig(
                env_name=env_name,
                agent_type="preference_ppo",
                ablation="none",
                platform=platform,
                n_episodes=int(n_episodes),
                seed=int(seed),
                out_dir=Path(out_dir),
                extra={"dvfs_mode": dvfs_mode},
                n_envs=int(n_envs),
            )
            configs.append(cfg)
    return configs


def _format_summary_table(rows: list[dict]) -> str:
    """Return the Markdown summary table as a single string."""
    headers = [
        "n_envs",
        "dvfs_mode",
        "total_episodes",
        "tail_p99_ms",
        "wall_time_s",
        "tail_p99_ratio_vs_nenvs1_same_dvfs",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        ratio = r["tail_p99_ratio_vs_nenvs1_same_dvfs"]
        ratio_str = "—" if ratio is None else f"{ratio:.4f}"
        cells = [
            str(r["n_envs"]),
            str(r["dvfs_mode"]),
            str(r["total_episodes"]),
            f"{r['tail_p99_ms']:.4f}",
            f"{r['wall_time_s']:.4f}",
            ratio_str,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _build_summary_rows(configs, results) -> list[dict]:
    """Combine configs + results into the rows used by the summary table.

    Computes ``tail_p99_ratio_vs_nenvs1_same_dvfs`` per dvfs_mode using
    the n_envs=1 baseline at the same dvfs_mode as the denominator.
    """
    # Index baselines (n_envs=1) by dvfs_mode for the ratio computation.
    baselines: dict[str, float] = {}
    for cfg, res in zip(configs, results):
        if int(cfg.n_envs) == 1:
            mode = cfg.extra.get("dvfs_mode", "")
            baselines[mode] = float(res.tail_p99_ms)

    rows = []
    for cfg, res in zip(configs, results):
        mode = cfg.extra.get("dvfs_mode", "")
        baseline = baselines.get(mode)
        ratio: Optional[float]
        if baseline is None or baseline <= 0.0:
            ratio = None
        else:
            ratio = float(res.tail_p99_ms) / baseline
        rows.append(
            {
                "n_envs": int(cfg.n_envs),
                "dvfs_mode": mode,
                "total_episodes": int(res.n_episodes),
                "tail_p99_ms": float(res.tail_p99_ms),
                "wall_time_s": float(res.wall_time_s),
                "tail_p99_ratio_vs_nenvs1_same_dvfs": ratio,
            }
        )
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/w9_multienv_orin/",
        help="Output directory for JSONLs and summary.md (default: runs/w9_multienv_orin/).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Per-env episode count (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0).",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="orin_agx",
        help="Platform identifier passed through to EvalConfig (default: orin_agx).",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="CartPole-v1",
        help="Gymnasium env name (default: CartPole-v1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the 6 configs without executing them.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = build_sweep_configs(
        out_dir=out_dir,
        n_episodes=args.n_episodes,
        seed=args.seed,
        platform=args.platform,
        env_name=args.env_name,
    )

    if args.dry_run:
        print(f"[dry-run] would execute {len(configs)} configs in {out_dir}:")
        for i, c in enumerate(configs):
            print(
                f"  [{i}] n_envs={c.n_envs} dvfs_mode={c.extra.get('dvfs_mode')} "
                f"agent={c.agent_type} ablation={c.ablation} "
                f"n_episodes={c.n_episodes} platform={c.platform}"
            )
        return 0

    # Lazy import: keeps --dry-run / --help cheap.
    from tetrarl.eval.runner import EvalRunner

    print(f"Running {len(configs)} configs (out_dir={out_dir})")
    runner = EvalRunner()
    results = runner.run_sweep(configs)

    rows = _build_summary_rows(configs, results)
    table_md = _format_summary_table(rows)

    print()
    print(table_md, end="")
    print()

    summary_path = out_dir / "multienv_summary.md"
    summary_path.write_text(table_md)
    print(f"summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
