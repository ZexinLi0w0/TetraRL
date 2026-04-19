"""Tests for PPO-Lagrangian: dual update, shaped reward, training loop."""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pytest

from tetrarl.morl.native.lagrangian import (
    LagrangianConfig,
    LagrangianDual,
    PPOLagrangianConfig,
    shaped_reward,
    train_ppo_lagrangian,
)
from tetrarl.morl.native.override import (
    HardwareTelemetry,
    OverrideLayer,
    OverrideThresholds,
)

# ---------------------------------------------------------------------------
# Dual update unit tests
# ---------------------------------------------------------------------------


def test_pi_update_converges_on_constant_violation():
    """Constant positive violation on constraint 0 -> lambda_T grows."""
    cfg = LagrangianConfig(
        n_constraints=3,
        targets=[0.0, 0.0, 0.0],
        lambda_lr_p=0.05,
        lambda_lr_i=0.01,
        lambda_max=100.0,
        integral_max=50.0,
    )
    dual = LagrangianDual(cfg)
    violation = np.array([0.5, 0.0, 0.0])
    prev_lambda_T = 0.0
    last_lambdas = None
    for _ in range(200):
        last_lambdas = dual.update(violation)
        # Monotonic in lambda_T (until clamp).
        assert last_lambdas[0] >= prev_lambda_T - 1e-12
        prev_lambda_T = last_lambdas[0]
    assert last_lambdas is not None
    assert last_lambdas[0] > 1.0
    assert last_lambdas[1] == 0.0
    assert last_lambdas[2] == 0.0


def test_pi_update_zero_when_no_violation():
    cfg = LagrangianConfig(
        n_constraints=3,
        targets=[0.0, 0.0, 0.0],
    )
    dual = LagrangianDual(cfg)
    zero = np.zeros(3)
    last_lambdas = None
    for _ in range(100):
        last_lambdas = dual.update(zero)
    assert last_lambdas is not None
    assert np.allclose(last_lambdas, 0.0)


def test_anti_windup_integral_clamp():
    """With integral_max=10, the integral accumulator must not explode.

    Per-step lambda growth analytical bound:
        delta = Kp * v + Ki * integral
              = 0.05 * 1000 + 0.01 * 10
              = 50 + 0.1 = 50.1
    so over the very first few steps lambda jumps ~50.1 each step until
    it slams into the lambda_max clamp; after the clamp engages further
    growth is zero. The point of THIS test is the integral piece — if
    the integral were unclamped it would reach 1000*100=100_000 and
    drive the rule to absurd numbers.
    """
    cfg = LagrangianConfig(
        n_constraints=3,
        targets=[0.0, 0.0, 0.0],
        lambda_lr_p=0.05,
        lambda_lr_i=0.01,
        lambda_max=1e9,  # disable lambda clamp so we isolate the integral
        integral_max=10.0,
    )
    dual = LagrangianDual(cfg)
    violation = np.array([1000.0, 0.0, 0.0])
    for _ in range(100):
        dual.update(violation)
    # Integral accumulator must stay clamped to +/- integral_max.
    assert abs(dual._integral_accum[0]) <= cfg.integral_max + 1e-9
    # Per-step delta is bounded by Kp*v + Ki*integral_max = 50 + 0.1 = 50.1
    # So 100 steps cannot exceed 100 * 50.1 = 5010.
    upper_bound = 100 * (cfg.lambda_lr_p * 1000.0 + cfg.lambda_lr_i * cfg.integral_max)
    assert dual.get_lambdas()[0] <= upper_bound + 1e-6


def test_anti_windup_lambda_clamp():
    cfg = LagrangianConfig(
        n_constraints=3,
        targets=[0.0, 0.0, 0.0],
        lambda_lr_p=10.0,
        lambda_lr_i=1.0,
        lambda_max=5.0,
        integral_max=50.0,
    )
    dual = LagrangianDual(cfg)
    huge = np.array([1000.0, 0.0, 0.0])
    for _ in range(20):
        dual.update(huge)
    assert dual.get_lambdas()[0] == pytest.approx(cfg.lambda_max)


def test_one_sided_lambda():
    """A negative violation shrinks lambda but never goes below lambda_min=0."""
    cfg = LagrangianConfig(
        n_constraints=3,
        targets=[0.0, 0.0, 0.0],
        lambda_lr_p=0.5,
        lambda_lr_i=0.1,
        lambda_min=0.0,
        lambda_max=100.0,
        integral_max=50.0,
        init_lambdas=[5.0, 0.0, 0.0],
    )
    dual = LagrangianDual(cfg)
    # We're way under target now -> negative violation.
    neg = np.array([-2.0, 0.0, 0.0])
    last_lambdas = None
    for _ in range(50):
        last_lambdas = dual.update(neg)
    assert last_lambdas is not None
    assert last_lambdas[0] >= cfg.lambda_min - 1e-12
    # Should have shrunk well below the initial value.
    assert last_lambdas[0] < 5.0


def test_shaped_reward_subtracts_penalty():
    # 10 - (0.1*1 + 0.2*2 + 0.3*3) = 10 - 1.4 = 8.6
    r = shaped_reward(10.0, np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]))
    assert r == pytest.approx(8.6)


def test_lagrangian_dual_reset():
    cfg = LagrangianConfig(
        n_constraints=3,
        targets=[0.0, 0.0, 0.0],
        init_lambdas=[1.0, 2.0, 3.0],
    )
    dual = LagrangianDual(cfg)
    dual.update(np.array([1.0, 1.0, 1.0]))
    dual.update(np.array([1.0, 1.0, 1.0]))
    pre_reset = dual.get_lambdas()
    pre_integral = dual._integral_accum.copy()
    assert not np.allclose(pre_reset, [1.0, 2.0, 3.0])
    assert not np.allclose(pre_integral, 0.0)
    dual.reset()
    assert np.allclose(dual.get_lambdas(), [1.0, 2.0, 3.0])
    assert np.allclose(dual._integral_accum, 0.0)


# ---------------------------------------------------------------------------
# Training-loop integration smoke tests
# ---------------------------------------------------------------------------


def _cartpole_fn():
    return gym.make("CartPole-v1")


def _benign_telemetry_fn(latency_ms: float = 0.0) -> HardwareTelemetry:
    """Always-clean telemetry: no constraint will ever be violated."""
    return HardwareTelemetry(
        latency_ema_ms=0.0,
        energy_remaining_j=0.0,
        memory_util=0.0,
    )


def _bad_telemetry_fn(latency_ms: float = 0.0) -> HardwareTelemetry:
    """Always-violating telemetry: latency is way over any sensible target."""
    return HardwareTelemetry(
        latency_ema_ms=1e6,
        energy_remaining_j=0.0,
        memory_util=0.0,
    )


def test_train_loop_runs_cartpole_short(tmp_path):
    cfg = LagrangianConfig(
        n_constraints=3,
        targets=[1e9, 1e9, 1e9],  # never violated -> isolates the loop test
    )
    ppo_cfg = PPOLagrangianConfig(
        num_steps=64,
        num_minibatches=2,
        update_epochs=2,
        seed=0,
    )
    log_path = tmp_path / "log.jsonl"
    result = train_ppo_lagrangian(
        env_fn=_cartpole_fn,
        lagrangian_config=cfg,
        ppo_config=ppo_cfg,
        telemetry_fn=_benign_telemetry_fn,
        override=None,
        total_steps=512,
        knob_mapper="n_steps",
        with_override=False,
        log_jsonl_path=str(log_path),
        verbose=False,
    )
    assert "network" in result
    assert "lambdas_history" in result
    assert "violation_rate_history" in result
    assert "override_fire_count" in result
    assert "total_steps" in result
    assert "knob_mapper" in result
    assert result["knob_mapper"] == "n_steps"
    assert result["total_steps"] == 512
    assert len(result["lambdas_history"]) > 1  # initial entry + per-iter entries
    # JSONL log must be non-empty and parseable.
    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 512
    rec = json.loads(lines[0])
    for key in (
        "step",
        "latency_ms",
        "energy_j",
        "memory_util",
        "reward_raw",
        "reward_shaped",
        "lambda_T",
        "lambda_E",
        "lambda_M",
        "override_fired",
        "violation_T",
        "violation_E",
        "violation_M",
    ):
        assert key in rec, f"missing key {key} in JSONL record"


def test_override_on_reduces_violation_rate_vs_off(tmp_path):
    """Wire-up test: when override is enabled and telemetry triggers it, the
    fire count is positive; when disabled, the fire count is zero."""
    # Override with a low max_latency_ms so the bad telemetry definitely fires.
    override = OverrideLayer(
        thresholds=OverrideThresholds(max_latency_ms=10.0),
        fallback_action=0,
    )
    cfg = LagrangianConfig(
        n_constraints=3,
        targets=[1e9, 1e9, 1e9],  # don't pollute with shaping; isolate override
    )
    ppo_cfg = PPOLagrangianConfig(
        num_steps=64,
        num_minibatches=2,
        update_epochs=1,
        seed=0,
    )

    on_log = tmp_path / "on.jsonl"
    on_result = train_ppo_lagrangian(
        env_fn=_cartpole_fn,
        lagrangian_config=cfg,
        ppo_config=ppo_cfg,
        telemetry_fn=_bad_telemetry_fn,
        override=override,
        total_steps=128,
        knob_mapper="n_steps",
        with_override=True,
        log_jsonl_path=str(on_log),
        verbose=False,
    )
    assert on_result["override_fire_count"] > 0

    # Re-create the override (it has internal counters) for the OFF run.
    override.reset()
    off_log = tmp_path / "off.jsonl"
    off_result = train_ppo_lagrangian(
        env_fn=_cartpole_fn,
        lagrangian_config=cfg,
        ppo_config=ppo_cfg,
        telemetry_fn=_bad_telemetry_fn,
        override=override,
        total_steps=128,
        knob_mapper="n_steps",
        with_override=False,  # OFF
        log_jsonl_path=str(off_log),
        verbose=False,
    )
    assert off_result["override_fire_count"] == 0
