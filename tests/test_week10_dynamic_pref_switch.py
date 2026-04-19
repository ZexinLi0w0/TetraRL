"""Tests for scripts/week10_dynamic_pref_switch.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "week10_dynamic_pref_switch.py"
)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True,
    )


def test_help_lists_required_flags():
    res = _run(["--help"])
    assert res.returncode == 0
    for flag in ("--out-dir", "--n-episodes", "--switch-episode",
                 "--seed", "--agent"):
        assert flag in res.stdout


def test_emits_png_svg_md(tmp_path: Path):
    res = _run([
        "--out-dir", str(tmp_path),
        "--n-episodes", "30",
        "--switch-episode", "15",
        "--seed", "0",
    ])
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "dynamic_pref_switch.png").exists()
    assert (tmp_path / "dynamic_pref_switch.svg").exists()
    assert (tmp_path / "dynamic_pref_switch_summary.md").exists()


def test_summary_md_reports_pre_and_post_means(tmp_path: Path):
    _run([
        "--out-dir", str(tmp_path),
        "--n-episodes", "30",
        "--switch-episode", "15",
        "--seed", "0",
    ])
    md = (tmp_path / "dynamic_pref_switch_summary.md").read_text()
    assert "pre-switch" in md.lower()
    assert "post-switch" in md.lower()
    assert "reward" in md.lower()
    assert "collapse" in md.lower()


def test_writes_per_episode_csv(tmp_path: Path):
    _run([
        "--out-dir", str(tmp_path),
        "--n-episodes", "20",
        "--switch-episode", "10",
        "--seed", "0",
    ])
    csv_path = tmp_path / "dynamic_pref_switch.csv"
    assert csv_path.exists()
    rows = csv_path.read_text().strip().splitlines()
    # 1 header + 20 episode rows
    assert len(rows) == 21
    header = rows[0].split(",")
    for col in ("episode", "reward", "omega_0", "omega_1",
                "omega_2", "omega_3"):
        assert col in header


def test_switch_episode_actually_changes_omega(tmp_path: Path):
    """The omega in the per-episode CSV must differ between an episode
    before the switch and one after."""
    _run([
        "--out-dir", str(tmp_path),
        "--n-episodes", "20",
        "--switch-episode", "10",
        "--seed", "0",
    ])
    csv_path = tmp_path / "dynamic_pref_switch.csv"
    rows = csv_path.read_text().strip().splitlines()[1:]
    parsed = [r.split(",") for r in rows]
    pre = parsed[5]   # episode 5
    post = parsed[15]  # episode 15 (after switch=10)
    pre_omega = tuple(round(float(x), 6) for x in pre[2:6])
    post_omega = tuple(round(float(x), 6) for x in post[2:6])
    assert pre_omega != post_omega
