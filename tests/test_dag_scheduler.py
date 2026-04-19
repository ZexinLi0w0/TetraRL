"""Tests for the synthetic DAG scheduling MO environment.

This env is built specifically to exercise the GCN feature extractor in
`tetrarl/morl/native/gnn_extractor.py`: it produces graph-structured
observations (node_features, edge_index) and a 3-objective reward vector
[throughput, -energy, -peak_memory].
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from tetrarl.envs.dag_scheduler import (
    DAGReadyMask,
    DAGSchedulerEnv,
    generate_random_dag,
)


def test_generate_random_dag_topology_valid():
    rng = np.random.default_rng(0)
    edge_index, node_costs = generate_random_dag(
        n_tasks=8, density=0.4, rng=rng
    )
    # Every edge (u, v) must satisfy u < v (topological by node index).
    assert edge_index.shape[0] == 2
    if edge_index.shape[1] > 0:
        assert (edge_index[0] < edge_index[1]).all()
    # node_costs has rows for each task with (compute, memory, deadline).
    assert node_costs.shape == (8, 3)
    assert (node_costs > 0).all()


def test_generate_random_dag_seeded_determinism():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    e1, c1 = generate_random_dag(8, 0.5, rng_a)
    e2, c2 = generate_random_dag(8, 0.5, rng_b)
    assert np.array_equal(e1, e2)
    assert np.allclose(c1, c2)


def test_env_obs_space_is_dict_with_graph_keys():
    env = DAGSchedulerEnv(n_tasks=6, density=0.3, seed=0)
    assert isinstance(env.observation_space, gym.spaces.Dict)
    keys = set(env.observation_space.spaces.keys())
    assert {"node_features", "edge_index", "num_edges", "valid_mask"} <= keys


def test_env_action_space_is_discrete_n():
    env = DAGSchedulerEnv(n_tasks=8, density=0.3, seed=0)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 8


def test_env_reward_vector_is_3d():
    env = DAGSchedulerEnv(n_tasks=6, density=0.3, seed=0)
    assert env.reward_dim == 3
    obs, _ = env.reset(seed=0)
    mask = obs["valid_mask"].astype(bool)
    valid_actions = np.where(mask)[0]
    assert len(valid_actions) >= 1
    _, r_vec, _, _, _ = env.step(int(valid_actions[0]))
    assert r_vec.shape == (3,)


def test_env_reset_returns_dict_obs():
    env = DAGSchedulerEnv(n_tasks=5, density=0.3, seed=0)
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert obs["node_features"].shape[0] == 5
    assert obs["edge_index"].shape[0] == 2
    assert obs["valid_mask"].shape == (5,)
    assert isinstance(info, dict)


def test_initial_valid_mask_marks_only_root_tasks():
    # density=0 means no edges so all tasks are roots.
    env = DAGSchedulerEnv(n_tasks=5, density=0.0, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs["valid_mask"].astype(bool).all()


def test_invalid_action_does_not_complete_task():
    """Picking a task whose deps are not done should not advance state."""
    env = DAGSchedulerEnv(n_tasks=5, density=1.0, seed=0)
    obs, _ = env.reset(seed=0)
    mask = obs["valid_mask"].astype(bool)
    invalid_actions = np.where(~mask)[0]
    if len(invalid_actions) == 0:
        pytest.skip("no invalid actions in this seed/density")
    invalid = int(invalid_actions[0])
    next_obs, r_vec, term, trunc, _ = env.step(invalid)
    # Throughput component should be 0 on an invalid step.
    assert r_vec[0] == 0.0
    assert next_obs["valid_mask"].shape == (5,)


def test_episode_terminates_when_all_tasks_complete():
    env = DAGSchedulerEnv(n_tasks=4, density=0.0, seed=0)
    env.reset(seed=0)
    terminated = False
    n_steps = 0
    while not terminated and n_steps < 50:
        obs = env._get_obs()
        valid = np.where(obs["valid_mask"].astype(bool))[0]
        if len(valid) == 0:
            break
        _, _, terminated, _, _ = env.step(int(valid[0]))
        n_steps += 1
    assert terminated, "env did not terminate after scheduling all tasks"


def test_throughput_sums_to_n_at_completion():
    env = DAGSchedulerEnv(n_tasks=4, density=0.0, seed=0)
    env.reset(seed=0)
    total = np.zeros(3)
    terminated = False
    while not terminated:
        obs = env._get_obs()
        valid = np.where(obs["valid_mask"].astype(bool))[0]
        _, r_vec, terminated, trunc, _ = env.step(int(valid[0]))
        total += r_vec
        if trunc:
            break
    assert total[0] == 4.0


def test_negative_energy_and_memory_components():
    env = DAGSchedulerEnv(n_tasks=4, density=0.0, seed=0)
    env.reset(seed=0)
    obs = env._get_obs()
    valid = np.where(obs["valid_mask"].astype(bool))[0]
    _, r_vec, _, _, _ = env.step(int(valid[0]))
    assert r_vec[1] <= 0.0
    assert r_vec[2] <= 0.0


def test_env_seed_determinism():
    env_a = DAGSchedulerEnv(n_tasks=6, density=0.3, seed=42)
    env_b = DAGSchedulerEnv(n_tasks=6, density=0.3, seed=42)
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    assert np.allclose(obs_a["node_features"], obs_b["node_features"])
    assert np.array_equal(obs_a["edge_index"], obs_b["edge_index"])


def test_dag_ready_mask_returns_valid_mask_from_obs():
    env = DAGSchedulerEnv(n_tasks=5, density=0.3, seed=0)
    obs, _ = env.reset(seed=0)
    m = DAGReadyMask()
    out = m.compute(obs, act_dim=5)
    assert out.dtype == bool
    assert out.shape == (5,)
    assert np.array_equal(out, obs["valid_mask"].astype(bool))
