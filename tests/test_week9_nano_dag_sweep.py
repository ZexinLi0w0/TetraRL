"""Tests for scripts/week9_nano_dag_sweep.py (Week 9 Task A).

Sweeps the TetraRL stack across 3 preference vectors omega on the
4-D DAG scheduling env, validating per-omega artefacts and the
aggregate summary CSV.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.week9_nano_dag_sweep import (
    OMEGAS_4D,
    parse_omegas,
    run_sweep,
)


def test_omegas_4d_has_three_required_keys():
    assert set(OMEGAS_4D) == {"energy_corner", "memory_corner", "center"}


def test_omegas_are_4d_and_sum_to_one():
    for name, omega in OMEGAS_4D.items():
        assert omega.shape == (4,), f"{name} not 4-D"
        assert np.isclose(omega.sum(), 1.0, atol=1e-6), f"{name} does not sum to 1"


def test_omegas_corners_select_correct_reward_dimension():
    # DAG reward layout: [throughput, -energy_step, -peak_memory_delta, -energy_norm_step]
    assert int(np.argmax(OMEGAS_4D["energy_corner"])) == 1
    assert int(np.argmax(OMEGAS_4D["memory_corner"])) == 2


def test_omegas_center_is_uniform():
    assert np.allclose(OMEGAS_4D["center"], np.full(4, 0.25, dtype=np.float32))


def test_parse_omegas_recognises_three_keywords():
    parsed = parse_omegas("energy_corner,memory_corner,center")
    names = [n for n, _ in parsed]
    assert names == ["energy_corner", "memory_corner", "center"]
    for _, o in parsed:
        assert o.shape == (4,)


def test_parse_omegas_rejects_unknown_keyword():
    with pytest.raises(ValueError):
        parse_omegas("nonsense_omega")


def test_parse_omegas_strips_whitespace():
    parsed = parse_omegas(" energy_corner , center ")
    names = [n for n, _ in parsed]
    assert names == ["energy_corner", "center"]


def test_parse_omegas_empty_string_rejected():
    with pytest.raises(ValueError):
        parse_omegas("")


def test_run_sweep_creates_per_omega_jsonl(tmp_path):
    omegas = parse_omegas("energy_corner,center")
    run_sweep(
        n_episodes=2,
        omegas=omegas,
        out_dir=tmp_path,
        seed=0,
        n_tasks=4,
        density=0.3,
        platform="mac_stub",
    )
    assert (tmp_path / "energy_corner" / "trace.jsonl").exists()
    assert (tmp_path / "center" / "trace.jsonl").exists()


def test_run_sweep_writes_summary_csv(tmp_path):
    omegas = parse_omegas("energy_corner,memory_corner,center")
    run_sweep(
        n_episodes=1,
        omegas=omegas,
        out_dir=tmp_path,
        seed=0,
        n_tasks=4,
        density=0.3,
        platform="mac_stub",
    )
    csv_path = tmp_path / "summary.csv"
    assert csv_path.exists()
    rows = csv_path.read_text().strip().splitlines()
    assert len(rows) == 4  # header + 3 data rows
    assert "omega_name" in rows[0]


def test_run_sweep_returns_summary_block(tmp_path):
    omegas = parse_omegas("center")
    summary = run_sweep(
        n_episodes=1,
        omegas=omegas,
        out_dir=tmp_path,
        seed=0,
        n_tasks=4,
        density=0.3,
        platform="mac_stub",
    )
    assert "per_omega" in summary
    assert "summary_csv" in summary
    assert len(summary["per_omega"]) == 1
    row = summary["per_omega"][0]
    assert row["omega_name"] == "center"
    assert row["n_episodes"] == 1
    assert row["n_steps"] >= 1
    assert "mean_scalarised_reward" in row


def test_run_sweep_jsonl_records_scalarised_and_vector_reward(tmp_path):
    omegas = parse_omegas("center")
    run_sweep(
        n_episodes=1,
        omegas=omegas,
        out_dir=tmp_path,
        seed=0,
        n_tasks=4,
        density=0.3,
        platform="mac_stub",
    )
    lines = (tmp_path / "center" / "trace.jsonl").read_text().strip().splitlines()
    assert len(lines) > 0
    rec = json.loads(lines[0])
    assert "scalarised_reward" in rec
    assert "reward_vec" in rec
    assert len(rec["reward_vec"]) == 4
    assert "omega" in rec
    assert len(rec["omega"]) == 4


def test_run_sweep_distinct_omegas_yield_distinct_mean_scalarised_reward(tmp_path):
    """Energy-corner and memory-corner should weight different reward dims,
    so even with identical action streams the scalarised mean must differ."""
    omegas = parse_omegas("energy_corner,memory_corner")
    summary = run_sweep(
        n_episodes=3,
        omegas=omegas,
        out_dir=tmp_path,
        seed=0,
        n_tasks=6,
        density=0.4,
        platform="mac_stub",
    )
    means = {r["omega_name"]: r["mean_scalarised_reward"] for r in summary["per_omega"]}
    assert means["energy_corner"] != means["memory_corner"]


def test_run_sweep_seed_reproducibility(tmp_path):
    """Seed must reproduce action/reward streams; latency_ms is wall-clock so excluded."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    omegas = parse_omegas("center")
    run_sweep(
        n_episodes=2, omegas=omegas, out_dir=out_a,
        seed=42, n_tasks=4, density=0.3, platform="mac_stub",
    )
    run_sweep(
        n_episodes=2, omegas=omegas, out_dir=out_b,
        seed=42, n_tasks=4, density=0.3, platform="mac_stub",
    )
    a_lines = (out_a / "center" / "trace.jsonl").read_text().splitlines()
    b_lines = (out_b / "center" / "trace.jsonl").read_text().splitlines()
    assert len(a_lines) == len(b_lines) > 0
    deterministic_keys = ("episode", "step", "action", "reward_vec", "scalarised_reward", "omega")
    for la, lb in zip(a_lines, b_lines):
        ra = json.loads(la)
        rb = json.loads(lb)
        for k in deterministic_keys:
            assert ra[k] == rb[k], f"field {k!r} differs: {ra[k]} != {rb[k]}"


def test_cli_help_subprocess(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "week9_nano_dag_sweep.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--n-episodes" in result.stdout
    assert "--omegas" in result.stdout
    assert "--out-dir" in result.stdout


def test_cli_runs_sweep_subprocess(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "week9_nano_dag_sweep.py"),
            "--n-episodes", "1",
            "--omegas", "center",
            "--out-dir", str(tmp_path),
            "--platform", "mac_stub",
            "--n-tasks", "4",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "center" / "trace.jsonl").exists()
