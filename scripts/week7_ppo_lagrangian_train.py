"""Week 7 deliverable: PPO-Lagrangian training driver.

Builds a ``LagrangianConfig`` + ``PPOLagrangianConfig`` from CLI args,
constructs a synthetic ``HardwareTelemetry`` callback, runs
``train_ppo_lagrangian`` end-to-end, and writes:

  * ``training_log.jsonl``    -- per-step dump (lambdas, violations, ...)
  * ``result.json``           -- final lambdas, override count, mean
                                 violation per constraint
  * ``summary.md``            -- human-readable summary
  * ``lambdas_convergence.png`` (best-effort, requires matplotlib)
  * ``violation_rate.png``    (best-effort, requires matplotlib)

Mac fallback: if the requested env name contains ``Bullet`` and pybullet
is unavailable, the script falls back to ``Pendulum-v1`` and prints a
warning. This keeps the smoke test runnable on dev machines without
pybullet installed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

# Resolve project root so this script runs both from the repo root and
# from inside the worktree.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tetrarl.morl.native.lagrangian import (  # noqa: E402
    LagrangianConfig,
    PPOLagrangianConfig,
    train_ppo_lagrangian,
)
from tetrarl.morl.native.override import (  # noqa: E402
    HardwareTelemetry,
    OverrideLayer,
    OverrideThresholds,
)


def _resolve_env(env_name: str) -> tuple[str, bool]:
    """Return (resolved_env_name, fell_back).

    If env_name mentions "Bullet" and pybullet is unavailable, we fall
    back to Pendulum-v1 so Mac smoke tests can still proceed.
    """
    if "Bullet" not in env_name:
        return env_name, False
    try:
        # PyBullet env registration; both forms are common in the wild.
        try:
            import pybullet_envs_gymnasium  # type: ignore  # noqa: F401
        except Exception:
            import pybullet_envs  # type: ignore  # noqa: F401
        # Confirm the env actually constructs.
        probe = gym.make(env_name)
        probe.close()
        return env_name, False
    except Exception as e:
        print(
            f"[warn] pybullet env '{env_name}' unavailable ({type(e).__name__}: {e}); "
            "falling back to 'Pendulum-v1' for this run."
        )
        return "Pendulum-v1", True


def _build_telemetry_fn(args: argparse.Namespace):
    """Synthetic telemetry callback for now.

    Real on-device telemetry comes in a later week; for the smoke test we
    just echo the latest step latency and report fixed budgets for energy
    + memory.
    """
    last_latency_ms = {"v": 0.0}

    def telemetry_fn(latency_ms: float = 0.0) -> HardwareTelemetry:
        if latency_ms > 0:
            last_latency_ms["v"] = float(latency_ms)
        return HardwareTelemetry(
            latency_ema_ms=last_latency_ms["v"],
            energy_remaining_j=10.0,
            memory_util=0.5,
        )

    return telemetry_fn


def _maybe_plot(out_dir: Path, history: list, ylabel: str, title: str, filename: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available ({e}); skipping {filename}")
        return
    if not history:
        return
    steps = [s for s, _ in history]
    arr = np.asarray([v for _, v in history], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"constraint_{i}" for i in range(arr.shape[1])]
    if arr.shape[1] >= 3:
        labels[:3] = ["lambda_T (latency)", "lambda_E (energy)", "lambda_M (memory)"]
        if "violation" in title.lower():
            labels[:3] = [
                "violation_rate_T",
                "violation_rate_E",
                "violation_rate_M",
            ]
    for i in range(arr.shape[1]):
        ax.plot(steps, arr[:, i], label=labels[i])
    ax.set_xlabel("env step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train PPO-Lagrangian on a continuous-control env."
    )
    parser.add_argument("--env", type=str, default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument(
        "--knob-mapper",
        type=str,
        choices=["n_steps", "n_epochs", "mini_batch_size"],
        default="n_steps",
    )
    parser.add_argument("--with-override", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latency-target-ms", type=float, default=30.0)
    parser.add_argument("--energy-target-j", type=float, default=5.0)
    parser.add_argument("--memory-target-util", type=float, default=0.85)
    parser.add_argument("--num-steps", type=int, default=256, help="rollout length")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.n_envs != 1:
        print(
            f"[warn] --n-envs={args.n_envs} requested but the loop is "
            "single-env (per Week 7 risk note). Recording the value only."
        )

    out_dir = Path(
        args.out_dir
        if args.out_dir is not None
        else f"runs/w7_ppo_lag_{int(time.time())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env_name, fell_back = _resolve_env(args.env)

    def env_fn() -> gym.Env:
        if "Bullet" in env_name:
            try:
                import pybullet_envs_gymnasium  # noqa: F401
            except Exception:
                import pybullet_envs  # noqa: F401
        return gym.make(env_name)

    lagr_cfg = LagrangianConfig(
        n_constraints=3,
        targets=[
            args.latency_target_ms,
            args.energy_target_j,
            args.memory_target_util,
        ],
    )
    ppo_cfg = PPOLagrangianConfig(
        seed=args.seed,
        num_steps=args.num_steps,
    )

    override: OverrideLayer | None = None
    if args.with_override:
        # Pick a fallback action that's safe for both discrete and
        # continuous spaces. Discrete: 0; continuous: midpoint vector.
        probe = env_fn()
        try:
            if isinstance(probe.action_space, gym.spaces.Box):
                fallback: Any = np.zeros(
                    probe.action_space.shape, dtype=np.float32
                )
            else:
                fallback = 0
        finally:
            probe.close()
        override = OverrideLayer(
            thresholds=OverrideThresholds(
                max_latency_ms=args.latency_target_ms,
                max_memory_util=args.memory_target_util,
            ),
            fallback_action=fallback,
        )

    log_path = out_dir / "training_log.jsonl"

    telemetry_fn = _build_telemetry_fn(args)

    print(f"[info] env={env_name} (fallback={fell_back}) total_steps={args.total_steps}")
    print(f"[info] out_dir={out_dir}")

    result = train_ppo_lagrangian(
        env_fn=env_fn,
        lagrangian_config=lagr_cfg,
        ppo_config=ppo_cfg,
        telemetry_fn=telemetry_fn,
        override=override,
        total_steps=args.total_steps,
        knob_mapper=args.knob_mapper,
        with_override=args.with_override,
        log_jsonl_path=str(log_path),
        verbose=not args.quiet,
    )

    final_lambdas = result["final_lambdas"]
    mean_violations = result["mean_violations"]

    summary = {
        "env": env_name,
        "fell_back": fell_back,
        "requested_env": args.env,
        "total_steps_requested": args.total_steps,
        "total_steps_completed": result["total_steps"],
        "knob_mapper": args.knob_mapper,
        "with_override": bool(args.with_override),
        "override_fire_count": int(result["override_fire_count"]),
        "final_lambdas": final_lambdas,
        "mean_violations": mean_violations,
        "targets": list(lagr_cfg.targets),
        "seed": args.seed,
    }
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2))

    summary_md = [
        "# PPO-Lagrangian run summary",
        "",
        f"- env: `{env_name}` (requested: `{args.env}`)",
        f"- fell_back_to_pendulum: `{fell_back}`",
        f"- total_steps: requested={args.total_steps}, completed={result['total_steps']}",
        f"- knob_mapper: `{args.knob_mapper}` (closed-loop coupling deferred)",
        f"- with_override: `{bool(args.with_override)}`, fires={int(result['override_fire_count'])}",
        f"- seed: {args.seed}",
        "",
        "## Final lambdas",
        "| constraint | target | final lambda | mean violation |",
        "|---|---|---|---|",
        f"| latency_ms | {lagr_cfg.targets[0]:.3f} | {final_lambdas[0]:.4f} | {mean_violations[0]:.4f} |",
        f"| energy_j   | {lagr_cfg.targets[1]:.3f} | {final_lambdas[1]:.4f} | {mean_violations[1]:.4f} |",
        f"| memory_util| {lagr_cfg.targets[2]:.3f} | {final_lambdas[2]:.4f} | {mean_violations[2]:.4f} |",
        "",
        "Per-step JSONL: `training_log.jsonl`. Plots: `lambdas_convergence.png`, `violation_rate.png` (if matplotlib available).",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_md))

    _maybe_plot(
        out_dir,
        result["lambdas_history"],
        ylabel="lambda",
        title="PPO-Lagrangian: lambdas convergence",
        filename="lambdas_convergence.png",
    )
    _maybe_plot(
        out_dir,
        result["violation_rate_history"],
        ylabel="violation rate (per rollout)",
        title="PPO-Lagrangian: per-constraint violation rate",
        filename="violation_rate.png",
    )

    print(
        f"[ok] wrote summary.md, result.json, training_log.jsonl in {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
