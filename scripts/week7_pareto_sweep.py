#!/usr/bin/env python3
"""Week 7 Task 4 -- 4-D preference-corner sweep + Pareto visualization.

Runs preference-conditioned PPO at five preference vectors -- the four
one-hot corners of the 4-D simplex plus one uniform interior point --
and aggregates the final 4-D objective returns into a Pareto front, an
HV indicator, and 3 x 2-D projection plots.

If the chosen environment is unavailable on the host (e.g. ``mo-halfcheetah``
without MuJoCo, or the ``dag`` env's reward dim does not match
``--obj-num``), the script falls back to a synthetic 4-D point cloud so
the visualization pipeline can still be smoke-tested end-to-end. The
``summary.md`` writes "synthetic dev points" in that case.

Example::

    python scripts/week7_pareto_sweep.py \
        --env dag_scheduler_mo \
        --steps-per-omega 30000 \
        --out-dir runs/w7_pareto/
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from tetrarl.eval.pareto import (
    compute_hv,
    pareto_front,
    pareto_summary_table,
    plot_2d_projections,
)


DEFAULT_DIM_LABELS = ["Throughput", "Latency", "Energy", "Memory"]


def build_omegas(n_omegas: int, obj_num: int) -> np.ndarray:
    """Return the sweep's preference vectors.

    Always includes the ``obj_num`` one-hot corners. If ``n_omegas`` is
    greater than ``obj_num``, fills the remainder with a single uniform
    vector first, then with random samples from the simplex.
    """
    rng = np.random.default_rng(0)
    omegas: list[np.ndarray] = []
    for k in range(min(n_omegas, obj_num)):
        v = np.zeros(obj_num, dtype=np.float64)
        v[k] = 1.0
        omegas.append(v)
    if n_omegas > obj_num:
        omegas.append(np.full(obj_num, 1.0 / obj_num, dtype=np.float64))
    while len(omegas) < n_omegas:
        v = rng.random(obj_num)
        v = v / v.sum()
        omegas.append(v)
    return np.stack(omegas[:n_omegas], axis=0)


def _train_one_omega(
    env_name: str,
    omega: np.ndarray,
    steps: int,
    obj_num: int,
    seed: int,
    device: str,
    ref_point: list[float],
) -> np.ndarray:
    """Train + evaluate one omega; return the 4-D mean evaluation return.

    Raises ``RuntimeError`` on any failure so the caller can decide to
    fall back to synthetic points.
    """
    from tetrarl.morl.native.agent import TetraRLNativeAgent

    if env_name == "dag_scheduler_mo":
        agent = TetraRLNativeAgent(
            env_name="dag",
            obj_num=obj_num,
            ref_point=ref_point,
            total_timesteps=int(steps),
            num_steps=64,
            hidden_dim=32,
            seed=seed,
            eval_interval=max(1, int(steps) // 256),
            eval_episodes=1,
            n_eval_interior=2,
            device=device,
            use_gnn=True,
            n_tasks=6,
            density=0.3,
        )
    elif env_name == "halfcheetah_mo":
        try:
            agent = TetraRLNativeAgent(
                env_name="mo-halfcheetah-v4",
                obj_num=obj_num,
                ref_point=ref_point,
                total_timesteps=int(steps),
                num_steps=64,
                hidden_dim=32,
                seed=seed,
                eval_interval=max(1, int(steps) // 256),
                eval_episodes=1,
                n_eval_interior=2,
                device=device,
            )
        except Exception:
            print(
                "[warn] mo-halfcheetah-v4 unavailable -- falling back to "
                "mo-mountaincarcontinuous-v0",
            )
            agent = TetraRLNativeAgent(
                env_name="mo-mountaincarcontinuous-v0",
                obj_num=obj_num,
                ref_point=ref_point,
                total_timesteps=int(steps),
                num_steps=64,
                hidden_dim=32,
                seed=seed,
                eval_interval=max(1, int(steps) // 256),
                eval_episodes=1,
                n_eval_interior=2,
                device=device,
            )
    else:
        raise ValueError(f"unknown env: {env_name}")

    agent.train(verbose=False)
    return agent.evaluate(omega, n_episodes=3)


def _synthetic_points(seed: int) -> np.ndarray:
    """Deterministic 5x4 dev fallback point cloud."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(5, 4)).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Week 7 Task 4 -- 4-D Pareto sweep + visualization"
    )
    parser.add_argument(
        "--env",
        choices=["dag_scheduler_mo", "halfcheetah_mo"],
        default="dag_scheduler_mo",
    )
    parser.add_argument("--steps-per-omega", type=int, default=30_000)
    parser.add_argument("--n-omegas", type=int, default=5)
    parser.add_argument("--obj-num", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ref-point",
        type=str,
        default="0,-100,-100,-100",
        help="comma-separated 4 floats (assumes maximization on all dims)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="defaults to runs/w7_pareto_<unix_ts>/",
    )
    args = parser.parse_args()

    ref_point = [float(x) for x in args.ref_point.split(",") if x.strip()]
    if len(ref_point) != args.obj_num:
        parser.error(
            f"--ref-point has {len(ref_point)} entries but --obj-num={args.obj_num}"
        )

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"runs/w7_pareto_{int(time.time())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    omegas = build_omegas(args.n_omegas, args.obj_num)
    print(f"[w7] sweep omegas:\n{omegas}")
    print(f"[w7] out_dir: {out_dir.resolve()}")

    points: list[np.ndarray] = []
    synthetic = False
    fallback_reason: str | None = None

    for k, omega in enumerate(omegas):
        print(f"[w7] training omega {k+1}/{len(omegas)}: {omega}")
        try:
            obj = _train_one_omega(
                env_name=args.env,
                omega=omega,
                steps=args.steps_per_omega,
                obj_num=args.obj_num,
                seed=args.seed + k,
                device=args.device,
                ref_point=ref_point,
            )
            obj = np.asarray(obj, dtype=np.float64)
            if obj.shape != (args.obj_num,):
                raise RuntimeError(
                    f"evaluate() returned shape {obj.shape} but expected "
                    f"({args.obj_num},) -- env reward dim likely mismatched"
                )
            points.append(obj)
            print(f"[w7]   -> {obj}")
        except Exception as e:
            fallback_reason = (
                f"omega {k} ({omega.tolist()}): "
                f"{type(e).__name__}: {e}"
            )
            print(f"[w7][warn] training failed -- {fallback_reason}")
            traceback.print_exc()
            synthetic = True
            break

    if synthetic:
        print("[w7] using synthetic dev fallback (5x4 normal points)")
        pts = _synthetic_points(args.seed)
    else:
        pts = np.stack(points, axis=0)

    csv_path = out_dir / "points.csv"
    np.savetxt(csv_path, pts, delimiter=",", header=",".join(DEFAULT_DIM_LABELS))

    pf = pareto_front(pts)
    hv = compute_hv(pts, np.asarray(ref_point, dtype=np.float64))
    paths = plot_2d_projections(
        pts,
        out_dir,
        dim_labels=DEFAULT_DIM_LABELS,
        ref_point=np.asarray(ref_point, dtype=np.float64),
    )
    summary = pareto_summary_table(
        pts,
        np.asarray(ref_point, dtype=np.float64),
        dim_labels=DEFAULT_DIM_LABELS,
    )

    md_lines = ["# Week 7 Pareto sweep summary", ""]
    md_lines.append(f"- env: `{args.env}`")
    md_lines.append(f"- steps-per-omega: {args.steps_per_omega}")
    md_lines.append(f"- n_omegas: {args.n_omegas}")
    md_lines.append(f"- obj_num: {args.obj_num}")
    md_lines.append(f"- ref_point: {ref_point}")
    if synthetic:
        md_lines.append("")
        md_lines.append(
            f"**NOTE: synthetic dev points used.** "
            f"Reason: {fallback_reason}"
        )
    md_lines.append("")
    md_lines.append(summary)
    md_lines.append("")
    md_lines.append("## Artifacts")
    md_lines.append("")
    md_lines.append(f"- points.csv: `{csv_path.resolve()}`")
    for k, v in paths.items():
        md_lines.append(f"- {k}: `{v}`")
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n")

    result = {
        "env": args.env,
        "steps_per_omega": args.steps_per_omega,
        "obj_num": args.obj_num,
        "ref_point": ref_point,
        "omegas": omegas.tolist(),
        "points": pts.tolist(),
        "pareto_front": pf.tolist(),
        "hv": float(hv),
        "synthetic": synthetic,
        "fallback_reason": fallback_reason,
        "artifacts": paths,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))

    print(f"[w7] wrote summary.md, points.csv, result.json + plots in {out_dir}")
    print(f"[w7] HV={hv:.4f}, |Pareto|={len(pf)}, synthetic={synthetic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
