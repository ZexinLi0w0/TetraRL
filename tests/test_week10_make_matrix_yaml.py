"""Tests for scripts/week10_make_matrix_yaml.py — Week 10 sweep generator."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tetrarl.eval.runner import load_sweep_yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "week10_make_matrix_yaml.py"


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_help_lists_required_flags():
    res = _run(["--help"])
    assert res.returncode == 0
    for flag in ("--out", "--agents", "--envs", "--seeds", "--n-episodes",
                 "--platform", "--out-dir"):
        assert flag in res.stdout


def test_default_matrix_has_90_entries(tmp_path: Path):
    """Default agents = 3 (preference_ppo + 2 baselines per spec line
    \"{SAC, PPO, preference_ppo}\"), default envs = 2, ω = 5, seeds = 3
    => 3 * 2 * 5 * 3 = 90."""
    out_yaml = tmp_path / "w10.yaml"
    res = _run(["--out", str(out_yaml)])
    assert res.returncode == 0, res.stderr
    cfgs = load_sweep_yaml(out_yaml)
    assert len(cfgs) == 90


def test_dag_only_fallback_yields_45_entries(tmp_path: Path):
    """When --envs lists only the DAG env (PyBullet missing) we expect
    3 * 1 * 5 * 3 = 45."""
    out_yaml = tmp_path / "w10.yaml"
    res = _run(["--out", str(out_yaml), "--envs", "dag_scheduler_mo-v0"])
    assert res.returncode == 0, res.stderr
    cfgs = load_sweep_yaml(out_yaml)
    assert len(cfgs) == 45


def test_full_baseline_matrix_yields_240_entries(tmp_path: Path):
    """All 8 agents (TetraRL + 7 baselines) * 2 envs * 5 omegas * 3 seeds = 240."""
    agents = ",".join([
        "preference_ppo", "envelope_morl", "ppo_lagrangian", "focops",
        "duojoule", "max_a", "max_p", "pcn",
    ])
    out_yaml = tmp_path / "w10.yaml"
    res = _run(["--out", str(out_yaml), "--agents", agents])
    assert res.returncode == 0, res.stderr
    cfgs = load_sweep_yaml(out_yaml)
    assert len(cfgs) == 240


def test_each_entry_carries_omega_in_extra(tmp_path: Path):
    out_yaml = tmp_path / "w10.yaml"
    res = _run(["--out", str(out_yaml), "--envs", "dag_scheduler_mo-v0"])
    assert res.returncode == 0
    cfgs = load_sweep_yaml(out_yaml)
    # Five distinct omegas should be present.
    omegas = {tuple(c.extra.get("omega", [])) for c in cfgs}
    assert len(omegas) == 5
    # Each omega is length-4.
    for o in omegas:
        assert len(o) == 4


def test_omega_corners_are_one_hot_plus_uniform(tmp_path: Path):
    out_yaml = tmp_path / "w10.yaml"
    res = _run(["--out", str(out_yaml), "--envs", "dag_scheduler_mo-v0"])
    assert res.returncode == 0
    cfgs = load_sweep_yaml(out_yaml)
    omegas = {tuple(c.extra.get("omega", [])) for c in cfgs}
    expected = {
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.25, 0.25, 0.25, 0.25),
    }
    assert omegas == expected


def test_out_dir_propagates_to_each_entry(tmp_path: Path):
    out_yaml = tmp_path / "w10.yaml"
    res = _run([
        "--out", str(out_yaml),
        "--envs", "dag_scheduler_mo-v0",
        "--out-dir", "runs/w10_orin_full",
    ])
    assert res.returncode == 0
    cfgs = load_sweep_yaml(out_yaml)
    assert all(str(c.out_dir).endswith("runs/w10_orin_full") for c in cfgs)


def test_platform_and_n_episodes_flags_propagate(tmp_path: Path):
    out_yaml = tmp_path / "w10.yaml"
    res = _run([
        "--out", str(out_yaml),
        "--envs", "dag_scheduler_mo-v0",
        "--platform", "orin_agx",
        "--n-episodes", "200",
    ])
    assert res.returncode == 0
    cfgs = load_sweep_yaml(out_yaml)
    assert all(c.platform == "orin_agx" for c in cfgs)
    assert all(c.n_episodes == 200 for c in cfgs)


def test_seeds_flag_overrides_default(tmp_path: Path):
    out_yaml = tmp_path / "w10.yaml"
    res = _run([
        "--out", str(out_yaml),
        "--envs", "dag_scheduler_mo-v0",
        "--seeds", "0,1,2,3,4",
    ])
    assert res.returncode == 0
    cfgs = load_sweep_yaml(out_yaml)
    # 3 default agents * 1 env * 5 omegas * 5 seeds = 75
    assert len(cfgs) == 75


def test_emitted_yaml_is_consumable_by_runner(tmp_path: Path):
    """The generated YAML round-trips through load_sweep_yaml without
    raising and each EvalConfig has the required fields populated."""
    out_yaml = tmp_path / "w10.yaml"
    res = _run(["--out", str(out_yaml), "--envs", "dag_scheduler_mo-v0"])
    assert res.returncode == 0
    cfgs = load_sweep_yaml(out_yaml)
    for c in cfgs:
        assert c.env_name
        assert c.agent_type
        assert c.platform
        assert c.n_episodes > 0
        assert isinstance(c.seed, int)
        assert "omega" in c.extra


def test_matrix_entries_carry_jsonl_name_with_omega_idx_suffix(tmp_path: Path):
    """Each entry's extra['jsonl_name'] must include __o<idx> so the
    runner writes 5 distinct files per (agent, seed) instead of
    overwriting on the canonical name."""
    out_yaml = tmp_path / "w10.yaml"
    res = _run([
        "--out", str(out_yaml),
        "--envs", "dag_scheduler_mo-v0",
        "--agents", "preference_ppo",
        "--seeds", "0",
    ])
    assert res.returncode == 0, res.stderr
    cfgs = load_sweep_yaml(out_yaml)
    names = {c.extra.get("jsonl_name") for c in cfgs}
    assert len(names) == 5
    for n in names:
        assert n is not None
        assert "__o" in n
        assert n.endswith(".jsonl")
