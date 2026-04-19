"""Week 4 integration tests for the TetraRL native agent.

Verifies that the optional masking, hardware override, and GNN extractor
pathways wired into TetraRLNativeAgent / preference-conditioned PPO behave
correctly while keeping the default-OFF baseline backward compatible.
"""

from __future__ import annotations

import numpy as np
import pytest

from tetrarl.morl.native.agent import TetraRLNativeAgent
from tetrarl.morl.native.gnn_extractor import GCNFeatureExtractor
from tetrarl.morl.native.override import (
    HardwareTelemetry,
    OverrideThresholds,
)


_COMMON_KW = dict(
    env_name="dst",
    obj_num=2,
    ref_point=[0.0, -25.0],
    total_timesteps=2_000,
    num_steps=128,
    hidden_dim=16,
    eval_interval=10,
    eval_episodes=1,
    n_eval_interior=2,
    seed=0,
    device="cpu",
)


def test_default_off_matches_baseline():
    """Defaults must still train and never fire the (absent) override."""
    agent = TetraRLNativeAgent(**_COMMON_KW)
    results = agent.train(verbose=False)

    assert results["override_fire_count"] == 0
    front = agent.get_pareto_front()
    assert front["hv"] > 0.0


def test_masking_with_noop_does_not_break():
    """use_masking=True with NoOpMask must complete training cleanly."""
    agent = TetraRLNativeAgent(use_masking=True, **_COMMON_KW)
    results = agent.train(verbose=False)

    assert results is not None
    front = agent.get_pareto_front()
    assert front["hv"] >= 0.0


def test_override_fires_when_telemetry_bad():
    """A telemetry source that always violates must trip the override."""
    bad_tele = lambda: HardwareTelemetry(latency_ema_ms=999.0)

    kw = dict(_COMMON_KW)
    kw["total_timesteps"] = 1_000

    agent = TetraRLNativeAgent(
        use_override=True,
        override_thresholds=OverrideThresholds(max_latency_ms=10.0),
        override_fallback=0,
        telemetry_fn=bad_tele,
        **kw,
    )
    results = agent.train(verbose=False)
    assert results["override_fire_count"] > 0


def test_gnn_extractor_constructible_inside_agent():
    """Construction-only check that the GNN injection point is wired."""
    extractor = GCNFeatureExtractor(in_dim=2, hidden_dim=8, out_dim=8)
    agent = TetraRLNativeAgent(
        use_gnn=True,
        gnn_extractor=extractor,
        **_COMMON_KW,
    )

    # Network is built lazily inside train(); should be None up front.
    assert agent._network is None
    assert agent._gnn_extractor is extractor
    assert agent._gnn_extractor.out_dim == 8


def test_gnn_train_raises_on_flat_env_consistency_check():
    """train(use_gnn=True) on a flat-obs env must raise ValueError.

    The Stage 2 rollout supports graph envs end-to-end, so the old
    NotImplementedError guard is gone. In its place, train_preference_ppo
    now enforces a strict consistency check: a Dict observation space must
    be paired with a gnn_extractor and vice versa. DST is a flat env, so
    pairing it with a GNN extractor must fail fast.
    """
    extractor = GCNFeatureExtractor(in_dim=2, hidden_dim=8, out_dim=8)
    kw = dict(_COMMON_KW)
    kw["total_timesteps"] = 256
    kw["num_steps"] = 128

    agent = TetraRLNativeAgent(
        use_gnn=True,
        gnn_extractor=extractor,
        **kw,
    )

    with pytest.raises(ValueError):
        agent.train(verbose=False)
