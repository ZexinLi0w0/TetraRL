"""Tests that the DAG-scheduler-MO env is registered with Gymnasium so the
unified eval runner can resolve ``env_name=dag_scheduler_mo-v0`` via
``gym.make`` without per-script env construction.

Also tests the ``MOAggregateWrapper`` that scalarises the env's 4-D reward
vector via the current omega so the existing scalar-reward eval loop can
consume it unmodified.
"""
from __future__ import annotations

import numpy as np


def test_dag_scheduler_mo_env_registered_under_gymnasium():
    """``gymnasium.make("dag_scheduler_mo-v0")`` must succeed and return a
    Gymnasium env with a Discrete action space."""
    import gymnasium as gym

    import tetrarl.envs  # noqa: F401  side-effect: register the env

    env = gym.make("dag_scheduler_mo-v0")
    try:
        assert hasattr(env.action_space, "n")
        assert int(env.action_space.n) > 0
    finally:
        env.close()


def test_mo_aggregate_wrapper_scalarises_vector_reward():
    """The wrapper turns a 4-vector reward into ``omega @ r_vec`` so the
    runner's ``float(reward)`` cast still works."""
    from tetrarl.envs.dag_scheduler import DAGSchedulerEnv
    from tetrarl.envs.wrappers import MOAggregateWrapper

    base = DAGSchedulerEnv(n_tasks=4, density=0.0, seed=0, reward_dim=4)
    omega = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    wrapped = MOAggregateWrapper(base, omega=omega)
    obs, _info = wrapped.reset(seed=0)
    obs, reward, term, trunc, info = wrapped.step(0)
    # omega = reward-only -> scalar reward is the throughput component.
    assert isinstance(reward, float)
    assert "reward_vec" in info
    assert len(info["reward_vec"]) == 4
    np.testing.assert_allclose(reward, float(info["reward_vec"][0]))


def test_mo_aggregate_wrapper_omega_zero_corner_zeros_reward():
    """When omega is zero on the throughput dim, the wrapper returns the
    inner-product on the *other* dims only."""
    from tetrarl.envs.dag_scheduler import DAGSchedulerEnv
    from tetrarl.envs.wrappers import MOAggregateWrapper

    base = DAGSchedulerEnv(n_tasks=4, density=0.0, seed=0, reward_dim=4)
    omega = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # energy-only
    wrapped = MOAggregateWrapper(base, omega=omega)
    obs, _info = wrapped.reset(seed=0)
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_allclose(reward, float(info["reward_vec"][1]))
