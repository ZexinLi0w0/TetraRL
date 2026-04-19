"""Week 10 Task 4: emit a denser ω-sensitivity sweep YAML.

The first W10 sweep (see :mod:`scripts.week10_make_matrix_yaml`) covered the
five Pareto-front anchors (four one-hot corners + uniform centroid). This
script adds four INTERMEDIATE ω vectors that live closer to the training
distribution, so we can characterise where ``preference_ppo`` actually wins
and where it loses. The four additional ω indices are 5..8 so the emitted
JSONL filenames (``__o5``..``__o8``) do NOT collide with the existing
``__o0``..``__o4`` files in the run directory.

Defaults for Task 4:
    4 agents (``preference_ppo`` + 3 strong baselines) * 1 env (DAG only,
    PyBullet deferred) * 4 omegas * 3 seeds = 48 configs.

Output format mirrors :mod:`week10_make_matrix_yaml`: a YAML doc with a
top-level ``configs:`` list whose entries match :meth:`EvalConfig.to_dict`.
Pass the resulting file to the runner via
``python -m tetrarl.eval.runner --config <out.yaml>``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

from tetrarl.eval.runner import EvalConfig

# Four 4-D intermediate omegas (each row sums to 1.0). Indices 5..8 to
# avoid colliding with the existing five corner/centroid omegas (0..4)
# in the same run directory.
INTERMEDIATE_OMEGAS_4D: list[list[float]] = [
    [0.4, 0.3, 0.2, 0.1],   # reward-leaning, balanced
    [0.1, 0.4, 0.3, 0.2],   # latency-leaning
    [0.2, 0.2, 0.3, 0.3],   # memory+energy-leaning, balanced
    [0.3, 0.3, 0.2, 0.2],   # reward+latency-leaning
]

# Index offset so the emitted ``__o<idx>`` suffix slots in after the five
# anchors from the original sweep (which used indices 0..4).
INTERMEDIATE_OMEGA_INDEX_OFFSET = 5

DEFAULT_AGENTS = "preference_ppo,pcn,pd_morl,max_performance"
DEFAULT_ENVS = "dag_scheduler_mo-v0"
DEFAULT_SEEDS = "0,1,2"
DEFAULT_N_EPISODES = 100
DEFAULT_PLATFORM = "orin_agx"
DEFAULT_OUT_DIR = "runs/w10_orin_full_fixed"
DEFAULT_ABLATION = "none"


def _parse_csv_str(s: str) -> list[str]:
    """Split a CSV string into a list of stripped non-empty tokens."""
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _parse_csv_int(s: str) -> list[int]:
    """Split a CSV string into a list of ints."""
    return [int(tok) for tok in _parse_csv_str(s)]


def build_intermediate_omega_configs(
    *,
    agents: list[str],
    envs: list[str],
    seeds: list[int],
    n_episodes: int,
    platform: str,
    out_dir: Path,
    ablation: str,
) -> list[EvalConfig]:
    """Return the Cartesian product (agents x envs x INTERMEDIATE omegas x seeds)."""
    out_dir = Path(out_dir)
    cfgs: list[EvalConfig] = []
    for agent in agents:
        for env in envs:
            for local_idx, omega in enumerate(INTERMEDIATE_OMEGAS_4D):
                omega_idx = INTERMEDIATE_OMEGA_INDEX_OFFSET + local_idx
                for seed in seeds:
                    jsonl_name = (
                        f"{ablation}__{agent}__seed{int(seed)}__o{omega_idx}.jsonl"
                    )
                    cfg = EvalConfig(
                        env_name=env,
                        agent_type=agent,
                        ablation=ablation,
                        platform=platform,
                        n_episodes=int(n_episodes),
                        seed=int(seed),
                        out_dir=out_dir,
                        extra={
                            "omega": [float(x) for x in omega],
                            "omega_idx": int(omega_idx),
                            "jsonl_name": jsonl_name,
                        },
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
            "CSV of agent_type values. Default = preference_ppo + 3 strong "
            "baselines (pcn, pd_morl, max_performance) so the default matrix "
            "has 48 rows."
        ),
    )
    parser.add_argument(
        "--envs",
        type=str,
        default=DEFAULT_ENVS,
        help=(
            "CSV of Gymnasium env_name values. Default = DAG only "
            "(PyBullet env deferred for the W10 sensitivity sweep)."
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

    cfgs = build_intermediate_omega_configs(
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
        f"(agents={len(agents)} x envs={len(envs)} "
        f"x intermediate_omegas={len(INTERMEDIATE_OMEGAS_4D)} "
        f"x seeds={len(seeds)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
