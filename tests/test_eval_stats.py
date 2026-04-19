"""Tests for tetrarl.eval.stats — Welch's t-test + ablation aggregation."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tetrarl.eval.stats import (
    aggregate_ablation,
    format_paper_table,
    welch_t_test,
)

_CSV_COLUMNS = [
    "env_name",
    "agent_type",
    "ablation",
    "platform",
    "seed",
    "n_episodes",
    "n_steps",
    "mean_reward",
    "override_fire_count",
    "tail_p99_ms",
    "mean_energy_j",
    "mean_memory_util",
    "wall_time_s",
]


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    """Write a runner-shaped summary.csv at ``path`` from a list of dicts."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            full = {col: r.get(col, "") for col in _CSV_COLUMNS}
            writer.writerow(full)


def _row(
    ablation: str,
    seed: int,
    mean_reward: float,
    tail_p99_ms: float = 1.0,
    n_steps: int = 10,
    override_fire_count: int = 0,
) -> dict:
    return {
        "env_name": "CartPole-v1",
        "agent_type": "random",
        "ablation": ablation,
        "platform": "mac_stub",
        "seed": seed,
        "n_episodes": 1,
        "n_steps": n_steps,
        "mean_reward": mean_reward,
        "override_fire_count": override_fire_count,
        "tail_p99_ms": tail_p99_ms,
        "mean_energy_j": 0.001,
        "mean_memory_util": 0.1,
        "wall_time_s": 0.5,
    }


def test_welch_t_test_returns_tuple_of_floats():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.5, 2.5, 3.5, 4.5]
    t, p = welch_t_test(a, b)
    assert isinstance(t, float)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_welch_t_test_zero_p_when_means_far_apart():
    a = [100.0, 101.0, 99.0, 100.5, 100.2, 99.8]
    b = [0.0, 0.5, -0.5, 0.2, -0.2, 0.1]
    _, p = welch_t_test(a, b)
    assert p < 0.001


def test_welch_t_test_high_p_when_samples_identical():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    _, p = welch_t_test(a, b)
    # Identical samples: t ≈ 0 → p ≈ 1.0
    assert p > 0.5


def test_welch_t_test_raises_on_singleton_sample():
    with pytest.raises(ValueError):
        welch_t_test([1.0], [2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        welch_t_test([1.0, 2.0, 3.0], [4.0])
    with pytest.raises(ValueError):
        welch_t_test([], [1.0, 2.0])


def test_aggregate_ablation_groups_by_ablation(tmp_path):
    rows = [
        _row("none", 0, 100.0),
        _row("none", 1, 102.0),
        _row("preference_plane", 0, 80.0),
        _row("preference_plane", 1, 82.0),
        _row("resource_manager", 0, 90.0),
        _row("resource_manager", 1, 91.0),
    ]
    csv_path = tmp_path / "summary.csv"
    _write_summary_csv(csv_path, rows)

    result = aggregate_ablation(csv_path)
    assert len(result) == 3
    arms = [r["ablation"] for r in result]
    assert arms == ["none", "preference_plane", "resource_manager"]

    baseline = result[0]
    assert baseline["n_seeds"] == 2
    for key in (
        "t_stat_n_steps_vs_baseline",
        "p_n_steps_vs_baseline",
        "t_stat_override_fire_count_vs_baseline",
        "p_override_fire_count_vs_baseline",
        "t_stat_mean_reward_vs_baseline",
        "p_mean_reward_vs_baseline",
        "t_stat_tail_p99_vs_baseline",
        "p_tail_p99_vs_baseline",
    ):
        assert baseline[key] is None

    for r in result[1:]:
        assert r["t_stat_n_steps_vs_baseline"] is not None
        assert r["p_n_steps_vs_baseline"] is not None
        assert r["t_stat_override_fire_count_vs_baseline"] is not None
        assert r["p_override_fire_count_vs_baseline"] is not None
        assert r["t_stat_mean_reward_vs_baseline"] is not None
        assert r["p_mean_reward_vs_baseline"] is not None
        assert r["t_stat_tail_p99_vs_baseline"] is not None
        assert r["p_tail_p99_vs_baseline"] is not None


def test_aggregate_ablation_computes_p_value_correctly(tmp_path):
    rows = [
        _row("none", 0, 100.0),
        _row("none", 1, 101.0),
        _row("none", 2, 99.5),
        _row("none", 3, 100.5),
        _row("preference_plane", 0, 10.0),
        _row("preference_plane", 1, 11.0),
        _row("preference_plane", 2, 9.5),
        _row("preference_plane", 3, 10.5),
    ]
    csv_path = tmp_path / "summary.csv"
    _write_summary_csv(csv_path, rows)

    result = aggregate_ablation(csv_path)
    pp = next(r for r in result if r["ablation"] == "preference_plane")
    assert pp["p_mean_reward_vs_baseline"] < 0.05
    assert pp["mean_reward_mean"] < 50.0  # sanity: arm should be the low-reward one


def test_aggregate_ablation_ignores_unknown_arm(tmp_path):
    rows = [
        _row("none", 0, 100.0),
        _row("none", 1, 101.0),
        _row("not_a_real_arm", 0, 50.0),
        _row("not_a_real_arm", 1, 51.0),
    ]
    csv_path = tmp_path / "summary.csv"
    _write_summary_csv(csv_path, rows)

    result = aggregate_ablation(csv_path)
    arms = [r["ablation"] for r in result]
    assert arms == ["none"]


def test_format_paper_table_returns_markdown():
    rows = [
        {
            "ablation": "none",
            "n_seeds": 3,
            "n_steps_mean": 3000.0,
            "n_steps_std": 30.0,
            "mean_reward_mean": 100.0,
            "mean_reward_std": 1.0,
            "tail_p99_ms_mean": 1.5,
            "tail_p99_ms_std": 0.1,
            "override_fire_count_mean": 0.0,
            "override_fire_count_std": 0.0,
            "mean_energy_j_mean": 0.002,
            "mean_energy_j_std": 0.0001,
            "t_stat_n_steps_vs_baseline": None,
            "p_n_steps_vs_baseline": None,
            "t_stat_override_fire_count_vs_baseline": None,
            "p_override_fire_count_vs_baseline": None,
            "t_stat_mean_reward_vs_baseline": None,
            "p_mean_reward_vs_baseline": None,
            "t_stat_tail_p99_vs_baseline": None,
            "p_tail_p99_vs_baseline": None,
        },
        {
            "ablation": "preference_plane",
            "n_seeds": 3,
            "n_steps_mean": 4300.0,
            "n_steps_std": 100.0,
            "mean_reward_mean": 50.0,
            "mean_reward_std": 2.0,
            "tail_p99_ms_mean": 1.6,
            "tail_p99_ms_std": 0.1,
            "override_fire_count_mean": 200.0,
            "override_fire_count_std": 10.0,
            "mean_energy_j_mean": 0.002,
            "mean_energy_j_std": 0.0001,
            "t_stat_n_steps_vs_baseline": 25.0,
            "p_n_steps_vs_baseline": 0.0001,
            "t_stat_override_fire_count_vs_baseline": 30.0,
            "p_override_fire_count_vs_baseline": 0.0002,
            "t_stat_mean_reward_vs_baseline": -20.0,
            "p_mean_reward_vs_baseline": 0.0001,
            "t_stat_tail_p99_vs_baseline": 1.0,
            "p_tail_p99_vs_baseline": 0.4,
        },
        {
            "ablation": "resource_manager",
            "n_seeds": 3,
            "n_steps_mean": 3050.0,
            "n_steps_std": 30.0,
            "mean_reward_mean": 95.0,
            "mean_reward_std": 2.0,
            "tail_p99_ms_mean": 1.55,
            "tail_p99_ms_std": 0.1,
            "override_fire_count_mean": 50.0,
            "override_fire_count_std": 5.0,
            "mean_energy_j_mean": 0.002,
            "mean_energy_j_std": 0.0001,
            "t_stat_n_steps_vs_baseline": 0.6,
            "p_n_steps_vs_baseline": 0.5,
            "t_stat_override_fire_count_vs_baseline": 0.4,
            "p_override_fire_count_vs_baseline": 0.7,
            "t_stat_mean_reward_vs_baseline": -2.0,
            "p_mean_reward_vs_baseline": 0.04,
            "t_stat_tail_p99_vs_baseline": 0.5,
            "p_tail_p99_vs_baseline": 0.6,
        },
        {
            "ablation": "rl_arbiter",
            "n_seeds": 3,
            "n_steps_mean": 3010.0,
            "n_steps_std": 25.0,
            "mean_reward_mean": 99.0,
            "mean_reward_std": 1.0,
            "tail_p99_ms_mean": 1.5,
            "tail_p99_ms_std": 0.1,
            "override_fire_count_mean": 45.0,
            "override_fire_count_std": 4.0,
            "mean_energy_j_mean": 0.002,
            "mean_energy_j_std": 0.0001,
            "t_stat_n_steps_vs_baseline": 0.2,
            "p_n_steps_vs_baseline": 0.8,
            "t_stat_override_fire_count_vs_baseline": 0.1,
            "p_override_fire_count_vs_baseline": 0.9,
            "t_stat_mean_reward_vs_baseline": -0.5,
            "p_mean_reward_vs_baseline": 0.6,
            "t_stat_tail_p99_vs_baseline": 0.0,
            "p_tail_p99_vs_baseline": 0.99,
        },
    ]

    table = format_paper_table(rows)
    assert isinstance(table, str)
    lines = table.splitlines()
    assert lines[0].startswith("### Table 4")
    # blank line, then header, then alignment row, then 4 body rows
    assert lines[1] == ""
    assert lines[2].startswith("| Ablation")
    # New header columns are present in canonical order.
    assert "n_steps (μ ± σ)" in lines[2]
    assert "Override fires (μ ± σ)" in lines[2]
    assert "Tail p99 ms (μ ± σ)" in lines[2]
    assert "Mean energy J (μ ± σ)" in lines[2]
    assert "Mean Reward (μ ± σ)" in lines[2]
    assert "p (n_steps)" in lines[2]
    assert "p (override)" in lines[2]
    assert "p (tail p99)" in lines[2]
    assert "p (reward)" in lines[2]
    assert lines[3].startswith("|---")
    body = lines[4:]
    assert len(body) == 4

    none_line = body[0]
    pp_line = body[1]
    rm_line = body[2]
    rl_line = body[3]

    assert "none" in none_line
    # Baseline must show em-dash in the p-value cells and an empty sig.
    assert "—" in none_line

    assert "***" in pp_line  # p<0.001 → ***
    assert "*" in rm_line and "**" not in rm_line and "***" not in rm_line  # 0.01<=p<0.05 → *
    assert "ns" in rl_line  # all p>=0.05 → ns


def test_aggregate_ablation_includes_n_steps_p_value(tmp_path):
    # Baseline n_steps clustered near 3000, arm n_steps clustered near 4300:
    # large separation should yield p < 0.05 for the arm.
    rows = [
        _row("none", 0, 1.0, n_steps=3000),
        _row("none", 1, 1.0, n_steps=3050),
        _row("none", 2, 1.0, n_steps=2980),
        _row("preference_plane", 0, 1.0, n_steps=4300),
        _row("preference_plane", 1, 1.0, n_steps=4350),
        _row("preference_plane", 2, 1.0, n_steps=4280),
    ]
    csv_path = tmp_path / "summary.csv"
    _write_summary_csv(csv_path, rows)

    result = aggregate_ablation(csv_path)
    pp = next(r for r in result if r["ablation"] == "preference_plane")
    assert "p_n_steps_vs_baseline" in pp
    assert pp["p_n_steps_vs_baseline"] is not None
    assert pp["p_n_steps_vs_baseline"] < 0.05
    assert pp["n_steps_mean"] > 4000.0


def test_aggregate_ablation_includes_override_fire_count_p_value(tmp_path):
    # Baseline override fires near 50, arm override fires near 300:
    # clear separation should yield p < 0.05 for the arm.
    rows = [
        _row("none", 0, 1.0, override_fire_count=50),
        _row("none", 1, 1.0, override_fire_count=55),
        _row("none", 2, 1.0, override_fire_count=45),
        _row("preference_plane", 0, 1.0, override_fire_count=300),
        _row("preference_plane", 1, 1.0, override_fire_count=320),
        _row("preference_plane", 2, 1.0, override_fire_count=295),
    ]
    csv_path = tmp_path / "summary.csv"
    _write_summary_csv(csv_path, rows)

    result = aggregate_ablation(csv_path)
    pp = next(r for r in result if r["ablation"] == "preference_plane")
    assert "p_override_fire_count_vs_baseline" in pp
    assert pp["p_override_fire_count_vs_baseline"] is not None
    assert pp["p_override_fire_count_vs_baseline"] < 0.05
    assert pp["override_fire_count_mean"] > 200.0


def test_aggregate_ablation_ordering_is_canonical(tmp_path):
    # Insert in reversed order; aggregate should still return canonical order.
    rows = []
    for arm in ["override_layer", "rl_arbiter", "resource_manager", "preference_plane", "none"]:
        rows.append(_row(arm, 0, 50.0))
        rows.append(_row(arm, 1, 51.0))
    csv_path = tmp_path / "summary.csv"
    _write_summary_csv(csv_path, rows)

    result = aggregate_ablation(csv_path)
    arms = [r["ablation"] for r in result]
    assert arms == [
        "none",
        "preference_plane",
        "resource_manager",
        "rl_arbiter",
        "override_layer",
    ]
    # Rounded to keep sigma==0 from biting NaN — std should be a finite float.
    for r in result:
        assert not math.isnan(r["mean_reward_std"])
