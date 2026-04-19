"""Week 10 fix: prove override actions cause measurable telemetry deltas.

The W10 PR #29 (HEAD 977468e) ran the full eval matrix on Orin AGX but
got an *identical* 47.1% violation rate for both Lagrangian-only and
Lagrangian+override variants. The agent flagged this as "stub telemetry
not reacting to override actions" — the per-step ``memory_util`` baked
into ``EvalRunner.run`` was a pure function of ``episode_step``
(``0.1 + 0.001 * episode_step``), so the override layer's action could
never move it.

These tests pin the *behavioral* requirement that the W10 fix introduces:

  * ``memory_util`` recorded in the per-step JSONL must be a function of
    the action that was actually executed (post-override). When the
    override layer fires and clamps the action to the safe fallback,
    the recorded memory pressure must be lower than the no-override
    baseline at the same step.
  * ``override_fire_count`` must be > 0 in a run where the synthetic
    pressure drives memory above the override threshold (otherwise the
    override layer has no observable effect anywhere).
  * The Lagrangian-violation script's per-run violation rate (over the
    same JSONL stream) must differ between override-on and override-off
    sweeps. Identical rates collapse the W10 acceptance criterion.

The single-env path (``EvalRunner.run``) and the vector path
(``EvalRunner._run_vec_env``) are both covered so the W10 violation
table can be built either way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tetrarl.eval.runner import EvalConfig, EvalRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def test_override_action_lowers_recorded_memory_util(tmp_path: Path) -> None:
    """Override on a max_action arbiter must reduce JSONL memory_util.

    With ``agent_type=max_action`` the arbiter always proposes the highest
    discrete action, which (by the W10 memory model) drives synthetic
    memory pressure above the override layer's ``max_memory_util=0.13``
    threshold. With override OFF (ablation=override_layer) the JSONL must
    record the unmodified high-action memory pressure; with override ON
    (ablation=none) the override clamps the action to the safe fallback
    and the recorded memory pressure must be strictly lower for a
    nontrivial fraction of steps.
    """
    out_off = tmp_path / "off"
    out_on = tmp_path / "on"

    cfg_off = EvalConfig(
        env_name="CartPole-v1",
        agent_type="max_action",
        ablation="override_layer",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=out_off,
    )
    cfg_on = EvalConfig(
        env_name="CartPole-v1",
        agent_type="max_action",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=out_on,
    )

    EvalRunner().run(cfg_off)
    EvalRunner().run(cfg_on)

    rec_off = _read_jsonl(next(out_off.glob("*.jsonl")))
    rec_on = _read_jsonl(next(out_on.glob("*.jsonl")))
    assert rec_off and rec_on

    mem_off = [float(r["memory_util"]) for r in rec_off]
    mem_on = [float(r["memory_util"]) for r in rec_on]
    n = min(len(mem_off), len(mem_on))
    assert n > 0

    deltas = [mem_off[i] - mem_on[i] for i in range(n)]
    assert max(deltas) > 0.0, (
        "override-on must record strictly lower memory_util on at least "
        f"one step; got per-step (off-on)={deltas}"
    )

    mean_off = sum(mem_off[:n]) / n
    mean_on = sum(mem_on[:n]) / n
    assert mean_on < mean_off, (
        f"mean memory_util with override ({mean_on:.4f}) should be lower "
        f"than without override ({mean_off:.4f})"
    )


def test_override_fire_count_nonzero_under_high_action_arbiter(
    tmp_path: Path,
) -> None:
    """A max_action arbiter must drive memory above the override threshold.

    ``override_fire_count`` is the canary the W10 violation table reads;
    if it is zero the override has demonstrably done nothing and the
    "Lagrangian + override" column degenerates into the Lagrangian-only
    column.
    """
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="max_action",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=tmp_path,
    )
    result = EvalRunner().run(cfg)
    assert result.override_fire_count > 0, (
        "override_fire_count must be > 0 in a run where the action-driven "
        "synthetic memory model crosses max_memory_util=0.13; got 0, which "
        "means the override layer has no observable effect."
    )


def test_override_changes_violation_rate_in_lagrangian_table(
    tmp_path: Path,
) -> None:
    """End-to-end: violation rate must differ between override-on/off runs.

    Mirrors the W10 Task 5 flow: run a small matrix with override on and
    off (same seed, same agent), compute the per-run violation rate using
    the same logic as ``scripts/week10_lagrangian_violation_table.py``
    (``compute_run_violation_rate``), and assert the two rates differ.
    The PR #29 baseline collapsed both to 0.471 because telemetry was
    action-independent.
    """
    from scripts.week10_lagrangian_violation_table import (
        compute_run_violation_rate,
    )

    out_off = tmp_path / "off"
    out_on = tmp_path / "on"

    common = dict(
        env_name="CartPole-v1",
        agent_type="max_action",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
    )
    EvalRunner().run(EvalConfig(
        ablation="override_layer", out_dir=out_off, **common,
    ))
    EvalRunner().run(EvalConfig(
        ablation="none", out_dir=out_on, **common,
    ))

    recs_off = _read_jsonl(next(out_off.glob("*.jsonl")))
    recs_on = _read_jsonl(next(out_on.glob("*.jsonl")))

    rate_off = compute_run_violation_rate(
        recs_off,
        latency_threshold_ms=100.0,
        memory_threshold=0.115,
        energy_budget_j=1e9,
    )
    rate_on = compute_run_violation_rate(
        recs_on,
        latency_threshold_ms=100.0,
        memory_threshold=0.115,
        energy_budget_j=1e9,
    )
    assert rate_on < rate_off, (
        f"violation rate with override ({rate_on:.3f}) should be strictly "
        f"lower than without override ({rate_off:.3f}); identical rates "
        "indicate telemetry is not reacting to override actions (the W10 "
        "PR #29 bug)."
    )


def test_override_action_lowers_memory_util_vector_path(tmp_path: Path) -> None:
    """Same invariant as the single-env case, but for the n_envs>1 path.

    ``EvalRunner._run_vec_env`` must use the action-aware memory model so
    the W10 violation table built on a vector-mode sweep also separates
    override-on from override-off.
    """
    out_off = tmp_path / "off"
    out_on = tmp_path / "on"

    common = dict(
        env_name="CartPole-v1",
        agent_type="max_action",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        n_envs=2,
    )
    EvalRunner().run(EvalConfig(
        ablation="override_layer", out_dir=out_off, **common,
    ))
    EvalRunner().run(EvalConfig(
        ablation="none", out_dir=out_on, **common,
    ))

    rec_off = _read_jsonl(next(out_off.glob("*.jsonl")))
    rec_on = _read_jsonl(next(out_on.glob("*.jsonl")))
    assert rec_off and rec_on

    mean_off = sum(float(r["memory_util"]) for r in rec_off) / len(rec_off)
    mean_on = sum(float(r["memory_util"]) for r in rec_on) / len(rec_on)
    assert mean_on < mean_off, (
        f"vector path: mean memory_util with override ({mean_on:.4f}) "
        f"should be lower than without ({mean_off:.4f})"
    )
