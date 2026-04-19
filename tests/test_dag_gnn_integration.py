"""Integration: TetraRLNativeAgent + GCN extractor on DAG scheduling env."""
from __future__ import annotations

import pytest
import torch

from tetrarl.envs.dag_scheduler import DAGReadyMask, DAGSchedulerEnv
from tetrarl.morl.native.agent import TetraRLNativeAgent
from tetrarl.morl.native.gnn_extractor import GCNFeatureExtractor
from tetrarl.morl.native.preference_ppo import (
    PreferenceNetwork,
    PreferencePPOConfig,
    train_preference_ppo,
)


def test_preference_network_accepts_graph_obs():
    torch.manual_seed(0)
    gnn = GCNFeatureExtractor(in_dim=4, hidden_dim=8, out_dim=8)
    net = PreferenceNetwork(
        obs_dim=4, act_dim=6, pref_dim=3, hidden_dim=16,
        continuous=False, gnn_extractor=gnn,
    )
    nf = torch.randn(6, 4)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    omega = torch.tensor([[0.4, 0.3, 0.3]], dtype=torch.float32)
    action, logp, ent, val = net.get_action_and_value(
        graph_obs={"node_features": nf, "edge_index": ei, "batch": None},
        omega=omega,
    )
    assert action.shape == (1,)
    assert val.shape == (1, 1)


def test_preference_network_batched_graph_obs():
    torch.manual_seed(0)
    gnn = GCNFeatureExtractor(in_dim=4, hidden_dim=8, out_dim=8)
    net = PreferenceNetwork(
        obs_dim=4, act_dim=6, pref_dim=3, hidden_dim=16,
        continuous=False, gnn_extractor=gnn,
    )
    # 2 graphs with 3 nodes each
    nf = torch.randn(6, 4)
    ei = torch.tensor([[0, 3], [1, 4]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    omega = torch.tensor(
        [[0.4, 0.3, 0.3], [0.1, 0.5, 0.4]], dtype=torch.float32
    )
    action, logp, ent, val = net.get_action_and_value(
        graph_obs={"node_features": nf, "edge_index": ei, "batch": batch},
        omega=omega,
    )
    assert action.shape == (2,)
    assert val.shape == (2, 1)


def test_train_preference_ppo_runs_on_dag_with_gnn():
    """Tiny end-to-end: 512 steps, n=4 tasks, must complete without error and HV >= 0."""
    config = PreferencePPOConfig(
        n_objectives=3,
        total_timesteps=512,
        num_steps=128,
        hidden_dim=16,
        seed=0,
        eval_interval=2,
        eval_episodes=1,
        n_eval_interior=2,
        ref_point=[0.0, -50.0, -50.0],
    )
    def env_fn():
        return DAGSchedulerEnv(n_tasks=4, density=0.3, seed=0, reward_dim=3)
    gnn = GCNFeatureExtractor(in_dim=4, hidden_dim=16, out_dim=16)
    results = train_preference_ppo(
        config, env_fn, device="cpu", verbose=False,
        gnn_extractor=gnn, mask=DAGReadyMask(),
    )
    assert results["global_step"] == 512
    assert results["best_hv"] >= 0.0


def test_native_agent_dag_with_gnn_smoke():
    agent = TetraRLNativeAgent(
        env_name="dag",
        obj_num=4,
        ref_point=[0.0, -50.0, -50.0, -1.0],
        total_timesteps=512,
        num_steps=128,
        hidden_dim=16,
        eval_interval=2,
        eval_episodes=1,
        n_eval_interior=2,
        seed=0,
        use_gnn=True,
        use_masking=True,
        action_mask=DAGReadyMask(),
        n_tasks=4,
        density=0.3,
    )
    results = agent.train(verbose=False)
    front = agent.get_pareto_front()
    assert results["global_step"] == 512
    assert front["hv"] >= 0.0


def test_inconsistent_gnn_flat_env_raises():
    """Passing a gnn_extractor to a flat-obs env should raise ValueError."""
    from tetrarl.envs.dst import DeepSeaTreasure
    config = PreferencePPOConfig(
        n_objectives=2, total_timesteps=128, num_steps=64,
        hidden_dim=8, seed=0, ref_point=[0.0, -25.0],
    )
    gnn = GCNFeatureExtractor(in_dim=4, hidden_dim=8, out_dim=8)
    with pytest.raises(ValueError):
        train_preference_ppo(
            config, lambda: DeepSeaTreasure(), gnn_extractor=gnn, verbose=False
        )
