"""Tests for Week 10 MORL baseline arbiters.

These are behavioural surrogates of published baselines for HV
comparison only — not full re-trainings. Each must:
- Be deterministic given (seed, omega).
- Honour the discrete arbiter interface: act(state, omega) -> int.
- Be visibly distinguishable from the other baselines in policy
  (so HV differs).
"""
from __future__ import annotations

import numpy as np
import pytest

from tetrarl.eval.runner import _make_rl_arbiter
from tetrarl.morl.baselines.duojoule import DuoJouleArbiter
from tetrarl.morl.baselines.envelope_morl import EnvelopeMORLArbiter
from tetrarl.morl.baselines.focops import FOCOPSArbiter
from tetrarl.morl.baselines.max_action import MaxActionArbiter
from tetrarl.morl.baselines.max_performance import MaxPerformanceArbiter
from tetrarl.morl.baselines.pcn import PCNArbiter
from tetrarl.morl.baselines.ppo_lagrangian_arbiter import PPOLagrangianArbiter


@pytest.mark.parametrize(
    "cls",
    [
        EnvelopeMORLArbiter,
        PPOLagrangianArbiter,
        FOCOPSArbiter,
        DuoJouleArbiter,
        MaxActionArbiter,
        MaxPerformanceArbiter,
        PCNArbiter,
    ],
)
def test_arbiter_returns_action_in_range(cls):
    arb = cls(n_actions=4, seed=0)
    omega = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    a = arb.act(state=None, omega=omega)
    assert isinstance(a, int)
    assert 0 <= a < 4


@pytest.mark.parametrize(
    "cls",
    [
        EnvelopeMORLArbiter,
        PPOLagrangianArbiter,
        FOCOPSArbiter,
        DuoJouleArbiter,
        PCNArbiter,
    ],
)
def test_arbiter_deterministic_given_seed_and_omega(cls):
    """Two arbiters with the same seed + omega must produce identical
    action sequences (for stochastic baselines too — they all RNG-seed)."""
    omega = np.array([0.5, 0.0, 0.5, 0.0], dtype=np.float32)
    a = cls(n_actions=3, seed=42)
    b = cls(n_actions=3, seed=42)
    seq_a = [a.act(state=None, omega=omega) for _ in range(20)]
    seq_b = [b.act(state=None, omega=omega) for _ in range(20)]
    assert seq_a == seq_b


def test_max_action_always_emits_top_index():
    arb = MaxActionArbiter(n_actions=5, seed=0)
    omega = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    for _ in range(10):
        assert arb.act(state=None, omega=omega) == 4


def test_max_performance_always_emits_zero():
    arb = MaxPerformanceArbiter(n_actions=5, seed=0)
    omega = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    for _ in range(10):
        assert arb.act(state=None, omega=omega) == 0


def test_duojoule_responds_to_energy_weight():
    """DuoJoule should pick a different action when the energy weight
    (omega[3]) is high vs low, demonstrating energy-awareness."""
    arb = DuoJouleArbiter(n_actions=4, seed=0)
    omega_perf = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    omega_energy = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    a_perf = arb.act(state=None, omega=omega_perf)
    a_energy = arb.act(state=None, omega=omega_energy)
    assert a_perf != a_energy


def test_envelope_morl_responds_to_omega_corner():
    """Envelope MORL with reward-only omega should differ from
    energy-only omega (its envelope op consumes omega)."""
    arb1 = EnvelopeMORLArbiter(n_actions=4, seed=0)
    arb2 = EnvelopeMORLArbiter(n_actions=4, seed=0)
    omega_r = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    omega_e = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    seq_r = [arb1.act(state=None, omega=omega_r) for _ in range(50)]
    seq_e = [arb2.act(state=None, omega=omega_e) for _ in range(50)]
    assert seq_r != seq_e


def test_pcn_raises_on_continuous_action_dim():
    """PCN paper is discrete-action only; arbiter must validate."""
    with pytest.raises(ValueError):
        PCNArbiter(n_actions=0, seed=0)


@pytest.mark.parametrize(
    "agent_type",
    [
        "envelope_morl",
        "ppo_lagrangian",
        "focops",
        "duojoule",
        "max_a",
        "max_p",
        "pcn",
    ],
)
def test_make_rl_arbiter_resolves_new_baselines(agent_type):
    arb = _make_rl_arbiter(agent_type=agent_type, ablation="none",
                           n_actions=3, seed=0)
    omega = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    a = arb.act(state=None, omega=omega)
    assert isinstance(a, int)
    assert 0 <= a < 3


def test_baselines_produce_distinguishable_policies_under_uniform_omega():
    """Under uniform omega the seven baselines must not all collapse to
    the same fixed action. We require at least 4 distinct first-action
    outputs across the 7 surrogates so HV downstream is non-degenerate."""
    omega = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    actions = []
    for name in ["envelope_morl", "ppo_lagrangian", "focops", "duojoule",
                 "max_a", "max_p", "pcn"]:
        arb = _make_rl_arbiter(agent_type=name, ablation="none",
                               n_actions=4, seed=0)
        # Take 10 actions and use the most-frequent one as a proxy for
        # the policy's mode under uniform omega.
        seq = [arb.act(state=None, omega=omega) for _ in range(10)]
        from collections import Counter
        mode = Counter(seq).most_common(1)[0][0]
        actions.append(mode)
    assert len(set(actions)) >= 4
