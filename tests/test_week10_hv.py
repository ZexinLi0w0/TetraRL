"""Tests for tetrarl.eval.hv — per-run HV from JSONL + Welch t-test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tetrarl.eval.hv import (
    HVRecord,
    aggregate_hv_table,
    compute_run_hv,
    welch_pvalue,
)


def _write_run(path: Path, n_episodes: int = 3, n_steps: int = 5,
               reward: float = 1.0, latency: float = 5.0,
               memory: float = 0.2, energy: float = 0.001,
               omega=(0.25, 0.25, 0.25, 0.25)) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ep in range(n_episodes):
            for st in range(n_steps):
                rec = {
                    "episode": ep,
                    "step": st,
                    "action": 0,
                    "reward": float(reward),
                    "latency_ms": float(latency),
                    "energy_j": float(energy),
                    "memory_util": float(memory),
                    "omega": list(map(float, omega)),
                }
                f.write(json.dumps(rec) + "\n")


def test_compute_run_hv_returns_finite_positive_for_dominating_run(tmp_path: Path):
    """A run whose per-episode point dominates the reference point in all
    dims must yield strictly positive, finite HV."""
    p = tmp_path / "none__random__seed0.jsonl"
    _write_run(p, reward=2.0, latency=1.0, memory=0.1, energy=1e-4)
    ref = np.array([0.0, -10.0, -1.0, -1.0])  # rew, -lat, -mem, -energy
    hv = compute_run_hv(p, ref_point=ref)
    assert np.isfinite(hv)
    assert hv > 0.0


def test_compute_run_hv_zero_when_dominated_by_ref(tmp_path: Path):
    """When all episode points are dominated by the reference, HV is 0."""
    p = tmp_path / "none__random__seed0.jsonl"
    # reward 0.1 below ref's reward of 1.0 -> dominated.
    _write_run(p, reward=0.1, latency=100.0, memory=0.9, energy=10.0)
    ref = np.array([1.0, -1.0, -0.5, -1e-3])
    hv = compute_run_hv(p, ref_point=ref)
    assert hv == 0.0


def test_aggregate_hv_table_groups_by_agent_env_seed(tmp_path: Path):
    """Given a directory of JSONLs and a manifest mapping filename ->
    (agent, env, seed), aggregate_hv_table returns a list of HVRecord
    sorted by (agent, env, seed)."""
    f1 = tmp_path / "none__preference_ppo__seed0.jsonl"
    f2 = tmp_path / "none__preference_ppo__seed1.jsonl"
    f3 = tmp_path / "none__random__seed0.jsonl"
    _write_run(f1, reward=2.0, latency=1.0, memory=0.1, energy=1e-4)
    _write_run(f2, reward=2.5, latency=1.0, memory=0.1, energy=1e-4)
    _write_run(f3, reward=1.5, latency=1.0, memory=0.1, energy=1e-4)
    manifest = {
        f1.name: ("preference_ppo", "CartPole-v1", 0),
        f2.name: ("preference_ppo", "CartPole-v1", 1),
        f3.name: ("random", "CartPole-v1", 0),
    }
    ref = np.array([0.0, -10.0, -1.0, -1.0])
    table = aggregate_hv_table(tmp_path, manifest, ref_point=ref)
    assert len(table) == 3
    assert all(isinstance(r, HVRecord) for r in table)
    agents = sorted({r.agent for r in table})
    assert agents == ["preference_ppo", "random"]
    seeds_for_pref = sorted([r.seed for r in table if r.agent == "preference_ppo"])
    assert seeds_for_pref == [0, 1]


def test_welch_pvalue_lower_when_means_clearly_separated():
    """Two clearly-separated samples should produce a small p-value."""
    a = np.array([10.0, 11.0, 9.5, 10.5, 10.2])
    b = np.array([1.0, 1.5, 0.8, 1.2, 1.0])
    p = welch_pvalue(a, b)
    assert 0.0 <= p <= 1.0
    assert p < 0.01


def test_welch_pvalue_high_when_samples_identical():
    a = np.array([5.0, 5.0, 5.0, 5.0])
    b = np.array([5.0, 5.0, 5.0, 5.0])
    p = welch_pvalue(a, b)
    # scipy returns NaN when both samples have zero variance; the helper
    # should normalise to 1.0 in that case (no separability).
    assert p == 1.0 or np.isclose(p, 1.0, atol=1e-6)


def test_compute_run_hv_handles_empty_jsonl(tmp_path: Path):
    p = tmp_path / "none__random__seed0.jsonl"
    p.write_text("")
    ref = np.array([0.0, -10.0, -1.0, -1.0])
    hv = compute_run_hv(p, ref_point=ref)
    assert hv == 0.0


def test_aggregate_hv_table_skips_files_not_in_manifest(tmp_path: Path):
    f1 = tmp_path / "none__preference_ppo__seed0.jsonl"
    f2 = tmp_path / "extraneous.jsonl"
    _write_run(f1)
    _write_run(f2)
    manifest = {f1.name: ("preference_ppo", "CartPole-v1", 0)}
    ref = np.array([0.0, -10.0, -1.0, -1.0])
    table = aggregate_hv_table(tmp_path, manifest, ref_point=ref)
    assert len(table) == 1
    assert table[0].agent == "preference_ppo"
