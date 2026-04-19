"""Week 10 deliverable: emit a sweep YAML covering the full eval matrix.

Generates a YAML doc consumable by :class:`tetrarl.eval.runner.EvalRunner`
spanning the Cartesian product:

    agents x envs x omegas x seeds

The five 4-D omegas are hardcoded to cover the Pareto-front corners (one-hot
on each of {reward, latency, memory, energy}) plus the uniform centroid.

Defaults follow the W10 spec line "= 90 runs":
    3 agents (preference_ppo + 2 baselines) * 2 envs * 5 omegas * 3 seeds.

The agent and env lists are exposed as CSV flags so the matrix can grow to
the full 8-baseline (240-run) configuration, and so the PyBullet env can be
dropped when unavailable.

Output format: a YAML doc with a top-level ``configs:`` list whose entries
match :meth:`EvalConfig.to_dict`. Pass the resulting file to the runner via
``python -m tetrarl.eval.runner --config <out.yaml>``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

from tetrarl.eval.runner import EvalConfig

# Five-omega 4-D Pareto-front cover: four one-hot corners + uniform centroid.
OMEGAS_4D: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.0],   # reward-only
    [0.0, 1.0, 0.0, 0.0],   # latency-only
    [0.0, 0.0, 1.0, 0.0],   # memory-only
    [0.0, 0.0, 0.0, 1.0],   # energy-only
    [0.25, 0.25, 0.25, 0.25],  # uniform
]

DEFAULT_AGENTS = "preference_ppo,envelope_morl,ppo_lagrangian"
DEFAULT_ENVS = "dag_scheduler_mo-v0,HalfCheetahBulletEnv-v0"
DEFAULT_SEEDS = "0,1,2"
DEFAULT_N_EPISODES = 100
DEFAULT_PLATFORM = "mac_stub"
DEFAULT_OUT_DIR = "runs/w10_orin_full"
DEFAULT_ABLATION = "none"


def _parse_csv_str(s: str) -> list[str]:
    """Split a CSV string into a list of stripped non-empty tokens."""
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _parse_csv_int(s: str) -> list[int]:
    """Split a CSV string into a list of ints."""
    return [int(tok) for tok in _parse_csv_str(s)]


def build_matrix_configs(
    *,
    agents: list[str],
    envs: list[str],
    seeds: list[int],
    n_episodes: int,
    platform: str,
    out_dir: Path,
    ablation: str,
) -> list[EvalConfig]:
    """Return the Cartesian product (agents x envs x omegas x seeds) as configs."""
    out_dir = Path(out_dir)
    cfgs: list[EvalConfig] = []
    for agent in agents:
        for env in envs:
            for omega in OMEGAS_4D:
                for seed in seeds:
                    cfg = EvalConfig(
                        env_name=env,
                        agent_type=agent,
                        ablation=ablation,
                        platform=platform,
                        n_episodes=int(n_episodes),
                        seed=int(seed),
                        out_dir=out_dir,
                        extra={"omega": [float(x) for x in omega]},
                    )
                    cfgs.append(cfg)
    return cfgs


def write_sweep_yaml(configs: list[EvalConfig], out_path: Path) -> None:
    """Dump ``configs`` to a YAML doc consumable by load_sweep_yaml."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"configs": [c.to_dict() for c in configs]}
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output YAML path.",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=DEFAULT_AGENTS,
        help=(
            "CSV of agent_type values. Default keeps the spec-equivalent trio "
            "(preference_ppo + 2 baselines) so the default matrix has 90 rows."
        ),
    )
    parser.add_argument(
        "--envs",
        type=str,
        default=DEFAULT_ENVS,
        help=(
            "CSV of Gymnasium env_name values. Drop the PyBullet entry when "
            "that backend is unavailable on the target host."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=DEFAULT_SEEDS,
        help="CSV of integer seeds.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=DEFAULT_N_EPISODES,
        help="Per-config episode count passed to each EvalConfig.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=DEFAULT_PLATFORM,
        help="Platform string forwarded into each EvalConfig.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Destination dir the runner will write per-run JSONLs into.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=DEFAULT_ABLATION,
        help="Ablation tag forwarded to each EvalConfig (default: none).",
    )
    args = parser.parse_args(argv)

    agents = _parse_csv_str(args.agents)
    envs = _parse_csv_str(args.envs)
    seeds = _parse_csv_int(args.seeds)

    cfgs = build_matrix_configs(
        agents=agents,
        envs=envs,
        seeds=seeds,
        n_episodes=int(args.n_episodes),
        platform=str(args.platform),
        out_dir=Path(args.out_dir),
        ablation=str(args.ablation),
    )
    out_path = Path(args.out)
    write_sweep_yaml(cfgs, out_path)
    print(
        f"wrote {len(cfgs)} configs to {out_path} "
        f"(agents={len(agents)} x envs={len(envs)} x omegas={len(OMEGAS_4D)} "
        f"x seeds={len(seeds)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
