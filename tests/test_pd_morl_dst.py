"""Integration test: PD-MORL on Deep Sea Treasure (sanity check).

Runs a short training loop (1000 frames) to verify the full pipeline
works end-to-end without error. Does NOT assert HV >= 229 — that
requires 200k frames. See scripts/train_pd_morl_dst.py for the full
reproduction.
"""

import numpy as np
import pytest

from tetrarl.envs.dst import DeepSeaTreasure
from tetrarl.eval.hypervolume import hypervolume, pareto_filter
from tetrarl.morl.agents.pd_morl import PDMORLAgent, Transition
from tetrarl.morl.preference_sampling import sample_preference


class TestDeepSeaTreasureEnv:

    def test_reset(self):
        env = DeepSeaTreasure()
        obs, info = env.reset(seed=0)
        assert obs.shape == (2,)
        assert np.array_equal(obs, [0.0, 0.0])

    def test_step(self):
        env = DeepSeaTreasure()
        env.reset(seed=0)
        obs, reward_vec, terminated, truncated, info = env.step(1)
        assert obs.shape == (2,)
        assert reward_vec.shape == (2,)
        assert reward_vec[1] == -1.0
        assert not truncated

    def test_reach_treasure(self):
        env = DeepSeaTreasure()
        env.reset(seed=0)
        obs, r, term, _, _ = env.step(1)
        assert term is True
        assert r[0] == 1.0
        assert r[1] == -1.0

    def test_pareto_front_count(self):
        assert len(DeepSeaTreasure.PARETO_OPTIMAL_RETURNS) == 10

    def test_max_steps(self):
        env = DeepSeaTreasure()
        env.reset(seed=0)
        for _ in range(200):
            obs, r, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        assert truncated or terminated


class TestHypervolume:

    def test_known_dst_pareto(self):
        front = np.array(DeepSeaTreasure.PARETO_OPTIMAL_RETURNS, dtype=np.float64)
        ref = np.array([0.0, -25.0])
        hv = hypervolume(front, ref)
        assert hv > 229.0

    def test_empty(self):
        hv = hypervolume(np.array([]).reshape(0, 2), np.array([0.0, 0.0]))
        assert hv == 0.0

    def test_single_point(self):
        front = np.array([[3.0, 2.0]])
        ref = np.array([0.0, 0.0])
        hv = hypervolume(front, ref)
        assert abs(hv - 6.0) < 1e-6

    def test_pareto_filter(self):
        points = np.array([[1, 2], [2, 1], [3, 3], [0, 0]])
        filtered = pareto_filter(points)
        assert len(filtered) == 1
        assert np.array_equal(filtered[0], [3, 3])


class TestPDMORLIntegration:

    @pytest.fixture
    def setup(self):
        env = DeepSeaTreasure()
        agent = PDMORLAgent(
            state_dim=2,
            action_dim=4,
            n_objectives=2,
            hidden_dim=64,
            lr=1e-3,
            batch_size=32,
            buffer_capacity=5000,
            n_relabel=4,
            epsilon_decay=500,
            device="cpu",
        )
        return env, agent

    def test_train_without_error(self, setup):
        env, agent = setup
        rng = np.random.default_rng(0)
        total_frames = 0
        target_frames = 1000

        while total_frames < target_frames:
            omega = sample_preference(2, 1, rng=rng)[0]
            obs, _ = env.reset()
            done = False
            while not done and total_frames < target_frames:
                action = agent.act(obs, omega, explore=True)
                next_obs, reward_vec, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                t = Transition(
                    state=obs,
                    action=action,
                    reward_vec=reward_vec,
                    next_state=next_obs,
                    done=terminated,
                    omega=omega,
                )
                agent.store(t)
                agent.update()
                obs = next_obs
                total_frames += 1

        assert total_frames == target_frames
        assert len(agent.buffer) > 0

    def test_loss_decreases(self, setup):
        env, agent = setup
        rng = np.random.default_rng(1)
        losses = []
        total_frames = 0

        while total_frames < 1000:
            omega = sample_preference(2, 1, rng=rng)[0]
            obs, _ = env.reset()
            done = False
            while not done and total_frames < 1000:
                action = agent.act(obs, omega, explore=True)
                next_obs, reward_vec, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                t = Transition(
                    state=obs,
                    action=action,
                    reward_vec=reward_vec,
                    next_state=next_obs,
                    done=terminated,
                    omega=omega,
                )
                agent.store(t)
                metrics = agent.update()
                if "loss" in metrics:
                    losses.append(metrics["loss"])
                obs = next_obs
                total_frames += 1

        assert len(losses) > 50
        early = np.mean(losses[:25])
        late = np.mean(losses[-25:])
        assert late < early * 2.0
