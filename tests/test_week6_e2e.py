"""Week 6 end-to-end framework smoke: pytest unit tests.

Exercises ``scripts/week6_e2e_smoke.py`` for a small (10-episode) run on
CartPole-v1 and asserts that the TetraRLFramework wiring produces the
expected per-step records (4-D telemetry, no NaNs, override fire_count
recorded, framework overhead bounded).

These are the lightweight CI-friendly companion to the full 100-episode
smoke script. We import the script's public symbols (factored out for
this purpose) rather than shelling out, so test failures point at the
exact line in the smoke script.
"""
from __future__ import annotations

import math

from scripts.week6_e2e_smoke import (
    RandomArbiter,
    StubTelemetrySource,
    make_framework,
    run_smoke,
)


def _run_short():
    """Run the same plumbing as the smoke script for 10 episodes."""
    return run_smoke(episodes=10, out_path=None, seed=0)


def test_e2e_history_populated_after_run():
    result = _run_short()
    assert len(result["history"]) == result["total_steps"] > 0


def test_e2e_all_4_dims_present_each_step():
    result = _run_short()
    required = {"reward", "latency_ms", "energy_j", "memory_util"}
    assert all(required.issubset(rec.keys()) for rec in result["history"])


def test_e2e_no_nan_in_telemetry():
    result = _run_short()
    fields = ("reward", "latency_ms", "energy_j", "memory_util")
    for rec in result["history"]:
        for f in fields:
            v = rec[f]
            assert v is not None and not math.isnan(float(v)), (
                f"NaN/None in field {f}: record={rec}"
            )


def test_e2e_override_fire_count_recorded():
    result = _run_short()
    assert int(result["override_fire_count"]) >= 0


def test_e2e_framework_overhead_under_50ms():
    result = _run_short()
    assert float(result["mean_framework_step_ms"]) < 50.0
