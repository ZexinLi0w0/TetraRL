"""Mac-side smoke test for scripts/week7_nano_cartpole.py.

Mirrors tests/test_week6_e2e.py: imports the driver's public function,
runs a short loop with stub telemetry (no real tegrastats / no real
DVFS sysfs writes), and asserts the framework wiring is intact and
the override layer's fire_count is exposed in the summary.

These tests must run on Mac in CI without root or any Jetson hardware.
"""
from __future__ import annotations

import json

from scripts.week7_nano_cartpole import run_nano_cartpole


def test_smoke_runs_and_returns_summary():
    summary = run_nano_cartpole(
        n_episodes=10,
        memory_pressure_mb=10,
        with_override=True,
        use_real_dvfs=False,
        use_real_tegrastats=False,
        out_dir=None,
        seed=0,
        max_memory_util=0.99,
        platform="nano",
    )
    assert "override_fire_count" in summary
    assert "oom_events" in summary
    assert summary["total_steps"] > 0
    assert summary["n_episodes"] == 10
    assert summary["platform"] == "nano"


def test_override_fires_when_threshold_crossed(tmp_path):
    """With max_memory_util=0.0, every step has memory_util > 0 → override fires."""
    summary = run_nano_cartpole(
        n_episodes=5,
        memory_pressure_mb=0,
        with_override=True,
        use_real_dvfs=False,
        use_real_tegrastats=False,
        out_dir=str(tmp_path),
        seed=0,
        max_memory_util=0.0,
        platform="nano",
    )
    assert summary["override_fire_count"] >= 1, (
        f"expected override to fire under memory pressure, "
        f"got fire_count={summary['override_fire_count']}"
    )
    assert summary["oom_events"] == 0


def test_summary_json_and_trace_written(tmp_path):
    summary = run_nano_cartpole(
        n_episodes=2,
        memory_pressure_mb=0,
        with_override=True,
        use_real_dvfs=False,
        use_real_tegrastats=False,
        out_dir=str(tmp_path),
        seed=0,
        max_memory_util=0.99,
        platform="nano",
    )
    summary_path = tmp_path / "summary.json"
    trace_path = tmp_path / "trace.jsonl"
    assert summary_path.exists(), "summary.json was not written"
    assert trace_path.exists(), "trace.jsonl was not written"

    on_disk = json.loads(summary_path.read_text())
    assert on_disk["override_fire_count"] == summary["override_fire_count"]
    assert on_disk["oom_events"] == 0


def test_oom_event_recorded_when_pressure_too_large():
    """Pressure absurdly large -> MemoryError caught -> oom_events incremented."""
    summary = run_nano_cartpole(
        n_episodes=1,
        memory_pressure_mb=10 ** 12,  # 1 EB; guaranteed MemoryError
        with_override=True,
        use_real_dvfs=False,
        use_real_tegrastats=False,
        out_dir=None,
        seed=0,
        max_memory_util=0.99,
        platform="nano",
    )
    assert summary["oom_events"] >= 1


def test_dvfs_uses_stub_when_no_real_dvfs(tmp_path):
    """`use_real_dvfs=False` must not require sysfs (Mac-safe)."""
    summary = run_nano_cartpole(
        n_episodes=1,
        memory_pressure_mb=0,
        with_override=True,
        use_real_dvfs=False,
        use_real_tegrastats=False,
        out_dir=str(tmp_path),
        seed=0,
        max_memory_util=0.99,
        platform="nano",
    )
    assert summary["total_steps"] > 0
