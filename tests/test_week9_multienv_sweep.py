"""Tests for scripts/week9_multienv_sweep.py — W9 multi-env scaling matrix."""
from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/ importable as a top-level package for the test session.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import week9_multienv_sweep  # noqa: E402


def test_build_sweep_configs_yields_six_configs(tmp_path):
    """W9: matrix has 3 n_envs × 2 dvfs_modes = 6 configs."""
    cfgs = week9_multienv_sweep.build_sweep_configs(
        out_dir=tmp_path,
        n_episodes=1,
        seed=0,
        platform="mac_stub",
        env_name="CartPole-v1",
    )
    assert len(cfgs) == 6


def test_build_sweep_configs_matrix_is_n_envs_x_dvfs_mode(tmp_path):
    """W9: every (n_envs, dvfs_mode) combination appears exactly once."""
    cfgs = week9_multienv_sweep.build_sweep_configs(
        out_dir=tmp_path,
        n_episodes=1,
        seed=0,
        platform="mac_stub",
        env_name="CartPole-v1",
    )
    seen = set()
    for c in cfgs:
        key = (c.n_envs, c.extra.get("dvfs_mode"))
        assert key not in seen, f"duplicate config: {key}"
        seen.add(key)
    expected = {
        (n, m)
        for n in (1, 2, 4)
        for m in ("fixed_max", "userspace_with_arbiter")
    }
    assert seen == expected


def test_build_sweep_configs_extra_carries_dvfs_mode(tmp_path):
    """W9: dvfs_mode lives in cfg.extra (not a first-class EvalConfig field)."""
    cfgs = week9_multienv_sweep.build_sweep_configs(
        out_dir=tmp_path,
        n_episodes=1,
        seed=0,
        platform="mac_stub",
        env_name="CartPole-v1",
    )
    for c in cfgs:
        assert "dvfs_mode" in c.extra
        assert c.extra["dvfs_mode"] in ("fixed_max", "userspace_with_arbiter")


def test_dry_run_prints_six_configs_without_executing(tmp_path, capsys):
    """W9: --dry-run lists the 6 configs and writes no JSONLs."""
    rc = week9_multienv_sweep.main(
        [
            "--dry-run",
            "--out-dir",
            str(tmp_path),
            "--n-episodes",
            "1",
            "--platform",
            "mac_stub",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    # Six configs should be listed in the dry-run output.
    listed = sum(1 for line in captured.out.splitlines() if "n_envs=" in line and "dvfs_mode=" in line)
    assert listed == 6, f"expected 6 listed configs, got {listed}\n{captured.out}"
    jsonl_files = list(tmp_path.rglob("*.jsonl"))
    assert jsonl_files == [], f"--dry-run wrote JSONLs: {jsonl_files}"


def test_main_smoke_run_writes_summary_md(tmp_path):
    """W9: end-to-end smoke writes multienv_summary.md with the expected header."""
    rc = week9_multienv_sweep.main(
        [
            "--n-episodes",
            "1",
            "--platform",
            "mac_stub",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    summary = tmp_path / "multienv_summary.md"
    assert summary.exists()
    text = summary.read_text()
    # Header columns spec:
    for col in (
        "n_envs",
        "dvfs_mode",
        "total_episodes",
        "tail_p99_ms",
        "wall_time_s",
        "tail_p99_ratio_vs_nenvs1_same_dvfs",
    ):
        assert col in text, f"summary.md missing column header {col!r}\n---\n{text}"
