"""Tests for eval_pd_morl_dst.py — evaluator + HV from saved model."""

import json
import os
import tempfile

import numpy as np
import torch

from tetrarl.eval.hypervolume import hypervolume, pareto_filter
from tetrarl.morl.agents.pd_morl import PDMORLAgent


class TestMakeAnchors:
    """Test anchor generation on 2-D simplex."""

    def test_anchors_sum_to_one(self) -> None:
        from scripts.eval_pd_morl_dst import make_anchors_2d

        anchors = make_anchors_2d(11)
        np.testing.assert_allclose(anchors.sum(axis=1), 1.0, atol=1e-6)

    def test_anchors_count(self) -> None:
        from scripts.eval_pd_morl_dst import make_anchors_2d

        for n in [1, 5, 11, 21]:
            assert len(make_anchors_2d(n)) == n

    def test_anchors_endpoints(self) -> None:
        from scripts.eval_pd_morl_dst import make_anchors_2d

        anchors = make_anchors_2d(11)
        np.testing.assert_allclose(anchors[0], [0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(anchors[-1], [1.0, 0.0], atol=1e-6)


class TestEvalPipeline:
    """Test the evaluation pipeline end-to-end with a fresh (untrained) agent."""

    def test_eval_produces_valid_hv(self) -> None:
        from scripts.eval_pd_morl_dst import evaluate_policy, make_anchors_2d

        agent = PDMORLAgent(
            state_dim=2, action_dim=4, n_objectives=2,
            hidden_dim=32, device="cpu",
        )
        anchors = make_anchors_2d(3)
        returns = evaluate_policy(agent, anchors, episodes_per_anchor=1, seed=0)

        assert returns.shape == (3, 2)
        assert np.all(returns[:, 1] <= 0)

        front = pareto_filter(returns)
        ref = np.array([0.0, -25.0])
        hv = hypervolume(front, ref)
        assert hv >= 0.0

    def test_save_load_roundtrip(self) -> None:
        agent = PDMORLAgent(
            state_dim=2, action_dim=4, n_objectives=2,
            hidden_dim=32, device="cpu",
        )
        agent.step_count = 12345

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save(path)
            agent2 = PDMORLAgent(
                state_dim=2, action_dim=4, n_objectives=2,
                hidden_dim=32, device="cpu",
            )
            agent2.load(path)
            assert agent2.step_count == 12345

            state = np.array([0.0, 0.0], dtype=np.float32)
            omega = np.array([0.5, 0.5], dtype=np.float32)
            with torch.no_grad():
                s = torch.tensor(state).unsqueeze(0)
                w = torch.tensor(omega).unsqueeze(0)
                q1 = agent.q_net(s, w)
                q2 = agent2.q_net(s, w)
            torch.testing.assert_close(q1, q2)
        finally:
            os.unlink(path)

    def test_eval_json_output(self) -> None:
        from scripts.eval_pd_morl_dst import evaluate_policy, make_anchors_2d

        agent = PDMORLAgent(
            state_dim=2, action_dim=4, n_objectives=2,
            hidden_dim=32, device="cpu",
        )
        anchors = make_anchors_2d(3)
        returns = evaluate_policy(agent, anchors, episodes_per_anchor=1, seed=0)
        front = pareto_filter(returns)
        ref = np.array([0.0, -25.0])
        hv = hypervolume(front, ref)

        results = {
            "achieved_hv": round(float(hv), 4),
            "reference_hv": 229.0,
            "gap_pct": round((1.0 - hv / 229.0) * 100.0, 2),
            "pareto_front": front.tolist(),
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(results, f)
            path = f.name

        try:
            with open(path) as f:
                loaded = json.load(f)
            assert "achieved_hv" in loaded
            assert "reference_hv" in loaded
            assert "gap_pct" in loaded
            assert isinstance(loaded["pareto_front"], list)
        finally:
            os.unlink(path)
