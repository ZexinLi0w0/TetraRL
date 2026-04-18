"""Smoke test: TetraRL native preference-conditioned PPO on DST.

Verifies end-to-end training, Pareto front discovery, and evaluation
on the Deep Sea Treasure benchmark. Uses short training (10k steps)
for fast CI execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from tetrarl.morl.native.agent import TetraRLNativeAgent


@pytest.fixture(scope="module")
def trained_agent():
    agent = TetraRLNativeAgent(
        env_name="dst",
        obj_num=2,
        ref_point=[0.0, -25.0],
        total_timesteps=10_000,
        num_steps=128,
        hidden_dim=32,
        eval_interval=5,
        eval_episodes=2,
        n_eval_interior=5,
        seed=42,
        device="cpu",
    )
    agent.train(verbose=True)
    return agent


class TestNativePPOTraining:

    def test_train_completes(self, trained_agent):
        results = trained_agent._results
        assert results is not None
        assert results["global_step"] == (10_000 // 128) * 128

    def test_hv_positive(self, trained_agent):
        front = trained_agent.get_pareto_front()
        assert front["hv"] > 0.0, (
            f"Expected HV > 0, got {front['hv']}"
        )

    def test_pareto_nonempty(self, trained_agent):
        front = trained_agent.get_pareto_front()
        assert len(front["objectives"]) >= 1

    def test_hv_history_recorded(self, trained_agent):
        front = trained_agent.get_pareto_front()
        assert len(front["hv_history"]) >= 1

    def test_evaluate_returns_correct_shape(self, trained_agent):
        omega = np.array([0.5, 0.5], dtype=np.float32)
        result = trained_agent.evaluate(omega, n_episodes=2)
        assert result.shape == (2,)

    def test_evaluate_at_corners(self, trained_agent):
        for omega in [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]:
            result = trained_agent.evaluate(omega, n_episodes=2)
            assert result.shape == (2,)


class TestNativePPOSaveLoad:

    def test_save_load_roundtrip(self, trained_agent, tmp_path):
        trained_agent.save(tmp_path / "test_model")
        new_agent = TetraRLNativeAgent(
            env_name="dst",
            obj_num=2,
            ref_point=[0.0, -25.0],
            hidden_dim=32,
            device="cpu",
        )
        new_agent.load(tmp_path / "test_model")

        omega = np.array([0.5, 0.5], dtype=np.float32)
        orig = trained_agent.evaluate(omega, n_episodes=1)
        loaded = new_agent.evaluate(omega, n_episodes=1)
        assert orig.shape == loaded.shape


def run_full_smoke_test():
    """Standalone 100k-step smoke test with detailed output."""
    print("=== TetraRL Native PPO — DST Smoke Test (100k steps) ===\n")

    agent = TetraRLNativeAgent(
        env_name="dst",
        obj_num=2,
        ref_point=[0.0, -25.0],
        total_timesteps=100_000,
        num_steps=256,
        hidden_dim=64,
        eval_interval=10,
        eval_episodes=3,
        n_eval_interior=10,
        seed=42,
        device="cpu",
    )
    results = agent.train(verbose=True)
    front = agent.get_pareto_front()

    print(f"\n{'='*50}")
    print(f"Final HV: {front['hv']:.2f}")
    print(f"|Pareto front|: {len(front['objectives'])}")
    print(f"Pareto front objectives:")
    for i, obj in enumerate(front["objectives"]):
        print(f"  [{i}] treasure={obj[0]:.1f}, time={obj[1]:.1f}")

    print(f"\nHV history:")
    for step, hv, n_pf in front["hv_history"]:
        print(f"  step={step:>7d}  HV={hv:>8.2f}  |PF|={n_pf}")

    omega_test = np.array([0.8, 0.2], dtype=np.float32)
    eval_result = agent.evaluate(omega_test, n_episodes=5)
    print(f"\nEval at omega=[0.8, 0.2]: {eval_result}")

    print(f"\n{'='*50}")
    ok = front["hv"] > 0 and len(front["objectives"]) >= 1
    print(f"SMOKE TEST {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    success = run_full_smoke_test()
    sys.exit(0 if success else 1)
