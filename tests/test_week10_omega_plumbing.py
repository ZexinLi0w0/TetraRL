"""Tests for Week 10 omega plumbing through EvalConfig.extra.

The full eval matrix needs each (agent, env, seed) run to use a different
omega vector. Since the existing runner hardcoded ``DEFAULT_OMEGA`` (2-D),
we plumb the per-run omega through ``EvalConfig.extra["omega"]`` so the
matrix YAML can drive 4-D Pareto-front sweeps without code changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tetrarl.eval.runner import EvalConfig, EvalRunner, _make_preference_plane


def test_make_preference_plane_uses_extra_omega_when_provided():
    """When the EvalConfig carries extra['omega'], the preference plane
    must emit *that* vector (not the runner's hardcoded default)."""
    omega_4d = [0.0, 0.0, 1.0, 0.0]  # memory-only corner, 4-D
    plane = _make_preference_plane(ablation="none", omega=omega_4d)
    out = plane.get()
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.asarray(omega_4d, dtype=np.float32))


def test_make_preference_plane_default_when_no_omega():
    """No omega in config -> fallback to existing 2-D DEFAULT_OMEGA path."""
    plane = _make_preference_plane(ablation="none", omega=None)
    out = plane.get()
    assert out.shape == (2,)


def test_eval_config_extra_round_trips_omega(tmp_path: Path):
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=tmp_path / "runs",
        extra={"omega": [0.25, 0.25, 0.25, 0.25]},
    )
    yaml_path = tmp_path / "c.yaml"
    cfg.to_yaml(yaml_path)
    cfg2 = EvalConfig.from_yaml(yaml_path)
    assert cfg2.extra["omega"] == [0.25, 0.25, 0.25, 0.25]


def test_runner_records_per_run_omega_in_jsonl(tmp_path: Path):
    """Smoke: a CartPole run with 4-D omega in extra writes records whose
    'omega' field has length 4."""
    omega_4d = [1.0, 0.0, 0.0, 0.0]
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=tmp_path,
        extra={"omega": omega_4d},
    )
    runner = EvalRunner()
    runner.run(cfg)

    jsonl = tmp_path / "none__random__seed0.jsonl"
    assert jsonl.exists()
    import json
    first = json.loads(jsonl.read_text().splitlines()[0])
    assert len(first["omega"]) == 4
    np.testing.assert_allclose(first["omega"], omega_4d)
