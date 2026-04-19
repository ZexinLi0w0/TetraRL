"""Tests for the DVFS-DRL-Multitask baseline (Week 9 Task C).

Implements Algorithm 3 from "DVFS-DRL-Multitask" (2024):
soft-deadline reward shaping
    r_shaped = r_base - lambda * max(0, latency - deadline)^2
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tetrarl.eval.runner import (
    EvalConfig,
    EvalRunner,
    _make_rl_arbiter,
    load_sweep_yaml,
)
from tetrarl.morl.baselines.dvfs_drl_multitask import (
    DVFSDRLMultitaskArbiter,
    soft_deadline_reward_shape,
)

# --- soft-deadline reward shaping --------------------------------------------


def test_soft_deadline_returns_base_when_under_deadline():
    assert (
        soft_deadline_reward_shape(
            r_base=1.0, latency_ms=10.0, deadline_ms=50.0, lambda_=1.0
        )
        == 1.0
    )


def test_soft_deadline_zero_excess_at_exact_deadline():
    assert (
        soft_deadline_reward_shape(
            r_base=2.0, latency_ms=50.0, deadline_ms=50.0, lambda_=1.0
        )
        == 2.0
    )


def test_soft_deadline_penalises_excess_quadratically():
    # 60 - 50 = 10 ms excess, lambda=1.0 -> penalty = 1 * 100 = 100
    val = soft_deadline_reward_shape(
        r_base=1.0, latency_ms=60.0, deadline_ms=50.0, lambda_=1.0
    )
    assert val == pytest.approx(1.0 - 100.0)


def test_soft_deadline_lambda_zero_returns_base_even_under_overrun():
    val = soft_deadline_reward_shape(
        r_base=1.0, latency_ms=200.0, deadline_ms=50.0, lambda_=0.0
    )
    assert val == 1.0


def test_soft_deadline_lambda_scales_penalty_linearly():
    full = soft_deadline_reward_shape(
        r_base=0.0, latency_ms=60.0, deadline_ms=50.0, lambda_=1.0
    )
    half = soft_deadline_reward_shape(
        r_base=0.0, latency_ms=60.0, deadline_ms=50.0, lambda_=0.5
    )
    assert half == pytest.approx(0.5 * full)


def test_soft_deadline_negative_lambda_rejected():
    with pytest.raises(ValueError):
        soft_deadline_reward_shape(
            r_base=0.0, latency_ms=10.0, deadline_ms=50.0, lambda_=-0.1
        )


def test_soft_deadline_negative_deadline_rejected():
    with pytest.raises(ValueError):
        soft_deadline_reward_shape(
            r_base=0.0, latency_ms=10.0, deadline_ms=-1.0, lambda_=1.0
        )


# --- arbiter ----------------------------------------------------------------


def test_arbiter_acts_within_action_space():
    arb = DVFSDRLMultitaskArbiter(n_actions=4, seed=0)
    omega = np.array([0.5, 0.5], dtype=np.float32)
    a = arb.act(state=None, omega=omega)
    assert isinstance(a, int)
    assert 0 <= a < 4


def test_arbiter_deterministic_for_same_state_omega_seed():
    omega = np.array([0.5, 0.5], dtype=np.float32)
    a1 = DVFSDRLMultitaskArbiter(n_actions=4, seed=7).act(None, omega)
    a2 = DVFSDRLMultitaskArbiter(n_actions=4, seed=7).act(None, omega)
    assert a1 == a2


def test_arbiter_omega_sensitivity_distinguishes_corners():
    """Two corner one-hot omegas must yield different action distributions."""
    arb_lat = DVFSDRLMultitaskArbiter(n_actions=4, seed=0)
    arb_thr = DVFSDRLMultitaskArbiter(n_actions=4, seed=0)
    actions_lat = [
        arb_lat.act(None, np.array([1.0, 0.0], dtype=np.float32))
        for _ in range(64)
    ]
    actions_thr = [
        arb_thr.act(None, np.array([0.0, 1.0], dtype=np.float32))
        for _ in range(64)
    ]
    assert actions_lat != actions_thr


def test_arbiter_default_deadline_and_lambda_positive():
    arb = DVFSDRLMultitaskArbiter(n_actions=2, seed=0)
    assert arb.deadline_ms > 0
    assert arb.lambda_ > 0


def test_arbiter_invalid_n_actions_raises():
    with pytest.raises(ValueError):
        DVFSDRLMultitaskArbiter(n_actions=0, seed=0)


# --- runner integration -----------------------------------------------------


def test_runner_recognises_dvfs_drl_multitask_agent_type():
    arb = _make_rl_arbiter(
        agent_type="dvfs_drl_multitask",
        ablation="none",
        n_actions=2,
        seed=0,
    )
    assert arb.__class__.__name__ == "DVFSDRLMultitaskArbiter"


def test_runner_runs_one_episode_with_dvfs_drl_multitask(tmp_path):
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="dvfs_drl_multitask",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=tmp_path,
    )
    result = EvalRunner().run(cfg)
    assert result.n_steps > 0
    assert result.n_episodes == 1


def test_runner_dvfs_drl_multitask_does_not_collide_with_rl_arbiter_ablation(tmp_path):
    """When ablation is rl_arbiter, the dvfs_drl_multitask agent type should
    still be replaced by _RandomArbiter (the ablation contract overrides
    the agent_type registry)."""
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="dvfs_drl_multitask",
        ablation="rl_arbiter",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=tmp_path,
    )
    framework = EvalRunner()._build_framework(cfg)
    assert framework.rl_arbiter.__class__.__name__ == "_RandomArbiter"


# --- YAML config ------------------------------------------------------------

_CFG_PATH = (
    Path(__file__).resolve().parent.parent
    / "tetrarl"
    / "eval"
    / "configs"
    / "dvfs_drl_multitask_nano.yaml"
)


def test_dvfs_drl_multitask_nano_yaml_exists_and_loads():
    assert _CFG_PATH.exists(), f"missing {_CFG_PATH}"
    cfgs = load_sweep_yaml(_CFG_PATH)
    assert len(cfgs) >= 1
    for c in cfgs:
        assert c.agent_type == "dvfs_drl_multitask"


def test_dvfs_drl_multitask_nano_yaml_targets_orin_nano_platform():
    cfgs = load_sweep_yaml(_CFG_PATH)
    assert any(c.platform == "orin_nano" for c in cfgs)
