"""Tests for the DuoJoule re-implementation (Yan et al. RTSS'24).

Covers the full ``DuoJouleAgent`` wrapper (act / store / update / greedy
controller) and the back-compat ``DuoJouleArbiter`` shim that the eval
runner registry imports. CPU-only; must pass on macOS.
"""

from __future__ import annotations

import numpy as np

from tetrarl.morl.agents.pd_morl import Transition
from tetrarl.morl.baselines.duojoule import DuoJouleAgent, DuoJouleArbiter


def _make_agent(seed: int = 0, **overrides) -> DuoJouleAgent:
    kw = dict(
        state_dim=4,
        action_dim=3,
        n_objectives=2,
        hidden_dim=32,
        lr=1e-3,
        device="cpu",
        seed=seed,
        epsilon_decay=100,
        buffer_capacity=2_000,
        n_relabel=1,
    )
    kw.update(overrides)
    return DuoJouleAgent(**kw)


def test_duojoule_agent_imports_on_mac():
    agent = _make_agent()
    assert agent.current_batch_size in agent.batch_sizes
    assert agent.current_replay_ratio in agent.replay_ratios
    assert agent.step_count == 0


def test_duojoule_agent_act_returns_valid_action():
    agent = _make_agent()
    state = np.zeros(4, dtype=np.float32)
    omega = np.array([0.5, 0.5], dtype=np.float32)
    for _ in range(10):
        a = agent.act(state, omega, explore=True)
        assert isinstance(a, int)
        assert 0 <= a < 3


def test_duojoule_agent_store_update_smoke():
    agent = _make_agent(initial_batch_idx=0)  # batch=32 -> warm fast
    rng = np.random.default_rng(0)
    omega = np.array([0.5, 0.5], dtype=np.float32)
    # Warm enough transitions to fill ``batch_size`` and trigger a real update.
    for _ in range(64):
        s = rng.standard_normal(4).astype(np.float32)
        ns = rng.standard_normal(4).astype(np.float32)
        a = agent.act(s, omega, explore=True)
        agent.store(
            Transition(
                state=s,
                action=a,
                reward_vec=np.array([1.0, -0.1], dtype=np.float32),
                next_state=ns,
                done=False,
                omega=omega,
            )
        )
    metrics = agent.update()
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    # end_episode must not raise on a freshly-warm agent either.
    agent.end_episode()


def test_duojoule_controller_converges_on_synthetic_2arm_bandit(monkeypatch):
    """Feed the controller a synthetic 2-arm signal and assert it settles
    on the lower-score arm within 30 episodes.

    We use a 1x2 search space (1 batch size, 2 replay ratios) so the only
    knob is R_idx and ``_random_neighbour`` deterministically alternates
    between the two arms. The synthetic score returns 0.0 for R_idx==1 and
    1.0 for R_idx==0, so the controller must converge on R_idx==1.
    """
    agent = DuoJouleAgent(
        state_dim=2,
        action_dim=2,
        n_objectives=2,
        hidden_dim=16,
        device="cpu",
        seed=0,
        batch_sizes=(64,),
        replay_ratios=(1, 2),
        initial_batch_idx=0,
        initial_replay_idx=0,
        buffer_capacity=128,
        n_relabel=0,
    )

    def fake_score(B_idx: int, R_idx: int) -> float:
        return 0.0 if R_idx == 1 else 1.0

    monkeypatch.setattr(agent, "_compute_efficiency_score", fake_score)

    for _ in range(30):
        agent.end_episode()

    assert agent.current_replay_ratio == 2, (
        f"controller did not converge on lower-score arm; "
        f"final R_idx -> ratio={agent.current_replay_ratio}"
    )


def test_duojoule_arbiter_back_compat():
    arb = DuoJouleArbiter(n_actions=4, seed=0)
    omega_perf = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    omega_energy = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    a_perf = arb.act(state=None, omega=omega_perf)
    a_energy = arb.act(state=None, omega=omega_energy)

    assert isinstance(a_perf, int) and 0 <= a_perf < 4
    assert isinstance(a_energy, int) and 0 <= a_energy < 4
    # Energy-dominant must pick low-energy action 0; perf-dominant the top.
    assert a_energy == 0
    assert a_perf == 3


def test_duojoule_agent_save_load_roundtrip(tmp_path):
    agent = _make_agent(seed=1, initial_batch_idx=2, initial_replay_idx=1)
    ckpt = tmp_path / "duojoule.pt"
    agent.save(str(ckpt))

    fresh = _make_agent(seed=99, initial_batch_idx=0, initial_replay_idx=0)
    fresh.load(str(ckpt))
    assert fresh.current_batch_size == agent.current_batch_size
    assert fresh.current_replay_ratio == agent.current_replay_ratio
