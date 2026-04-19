"""Tests for the per-component OverheadProfiler (Week 8 Task 1).

These tests pin down the public surface needed for Table 5 in the paper:
per-component (i)-(vi) wall-clock distribution + python memory delta,
exposed both as raw samples (for CSV) and as summary statistics.

The profiler is intentionally a black box wrt timing source; we use a
``_record_sample`` hook to inject deterministic distributions where
percentile math has to be exact (otherwise sleep-based timing would be
flaky in CI). For the sleep-based smoke test we use generous bounds.
"""
from __future__ import annotations

import csv
import time

import numpy as np
import pytest

from tetrarl.core.framework import (
    ResourceManager,
    StaticPreferencePlane,
    TetraRLFramework,
)
from tetrarl.eval.overhead import ComponentTimer, OverheadProfiler
from tetrarl.morl.native.override import (
    HardwareTelemetry,
    OverrideLayer,
    OverrideThresholds,
)

# ---------------------------------------------------------------------------
# Stubs (mirror tests/test_framework.py style)
# ---------------------------------------------------------------------------


class _StubArbiter:
    def __init__(self, action=1):
        self.action = action
        self.calls = []

    def act(self, state, omega):
        self.calls.append((np.asarray(state).copy(), np.asarray(omega).copy()))
        return self.action


class _StubTelemetry:
    def __init__(self, readings):
        self._readings = list(readings)
        self._i = 0

    def latest(self):
        if self._i >= len(self._readings):
            return self._readings[-1] if self._readings else None
        r = self._readings[self._i]
        self._i += 1
        return r


class _StubDvfsController:
    """Minimal DVFS controller exposing the surface framework.step() touches."""

    def __init__(self, n_levels: int = 4):
        self._n = int(n_levels)
        self.set_calls: list[int] = []

    def available_frequencies(self) -> dict:
        return {"gpu": list(range(self._n)), "cpu": list(range(self._n))}

    def set_freq(self, gpu_idx: int) -> None:
        self.set_calls.append(int(gpu_idx))


def _telemetry_to_hw(reading):
    if reading is None:
        return HardwareTelemetry()
    return HardwareTelemetry(
        latency_ema_ms=getattr(reading, "latency_ema_ms", None),
        energy_remaining_j=getattr(reading, "energy_remaining_j", None),
        memory_util=getattr(reading, "memory_util", None),
    )


def _make_framework(arbiter, telemetry, override=None, omega=None,
                    dvfs_controller=None, profiler=None):
    pref = StaticPreferencePlane(omega if omega is not None else np.array([0.5, 0.5]))
    rm = ResourceManager()
    if override is None:
        override = OverrideLayer(OverrideThresholds(), fallback_action=0)
    return TetraRLFramework(
        preference_plane=pref,
        rl_arbiter=arbiter,
        resource_manager=rm,
        override_layer=override,
        telemetry_source=telemetry,
        telemetry_adapter=_telemetry_to_hw,
        dvfs_controller=dvfs_controller,
        profiler=profiler,
    )


# ---------------------------------------------------------------------------
# 1. ComponentTimer wall-clock smoke test
# ---------------------------------------------------------------------------


def test_component_timer_records_elapsed_for_10ms_sleep():
    prof = OverheadProfiler(track_memory=False)
    with prof.time("sleep10"):
        time.sleep(0.010)
    samples = prof.samples_ns("sleep10")
    assert len(samples) == 1
    elapsed_ns = samples[0]
    # Lower bound 9 ms (allow tiny scheduler jitter); upper bound 200 ms (sanity).
    assert elapsed_ns >= 9_000_000, f"too short: {elapsed_ns} ns"
    assert elapsed_ns <= 200_000_000, f"absurdly long: {elapsed_ns} ns"


# ---------------------------------------------------------------------------
# 2. Multiple `with prof.time(...)` calls accumulate per-name
# ---------------------------------------------------------------------------


def test_multiple_with_blocks_accumulate_per_name():
    prof = OverheadProfiler(track_memory=False)
    for _ in range(5):
        with prof.time("comp_a"):
            pass
    for _ in range(3):
        with prof.time("comp_b"):
            pass
    assert len(prof.samples_ns("comp_a")) == 5
    assert len(prof.samples_ns("comp_b")) == 3


# ---------------------------------------------------------------------------
# 3. summarize() schema includes all required keys
# ---------------------------------------------------------------------------


def test_summarize_schema_keys_present():
    prof = OverheadProfiler(track_memory=True)
    with prof.time("foo"):
        pass
    s = prof.summarize()
    assert "foo" in s
    entry = s["foo"]
    for key in ("mean_ms", "p50_ms", "p99_ms", "mem_mb", "n_samples"):
        assert key in entry, f"summarize missing key: {key}"


# ---------------------------------------------------------------------------
# 4. summarize() returns {} when no samples
# ---------------------------------------------------------------------------


def test_summarize_empty_when_no_samples():
    prof = OverheadProfiler(track_memory=False)
    assert prof.summarize() == {}


# ---------------------------------------------------------------------------
# 5. p50/p99 match np.percentile on injected distribution
# ---------------------------------------------------------------------------


def test_percentiles_match_numpy_on_injected_distribution():
    prof = OverheadProfiler(track_memory=False)
    rng = np.random.default_rng(42)
    raw_us = rng.integers(low=100, high=10_000, size=500)  # 0.1ms..10ms
    for v in raw_us:
        prof._record_sample("rl_arbiter_act", elapsed_ns=int(v) * 1_000,
                            mem_delta_bytes=0)
    s = prof.summarize()["rl_arbiter_act"]
    expected_p50_ms = float(np.percentile(raw_us, 50)) / 1_000.0
    expected_p99_ms = float(np.percentile(raw_us, 99)) / 1_000.0
    assert s["p50_ms"] == pytest.approx(expected_p50_ms, rel=1e-6, abs=1e-6)
    assert s["p99_ms"] == pytest.approx(expected_p99_ms, rel=1e-6, abs=1e-6)
    assert s["n_samples"] == 500


# ---------------------------------------------------------------------------
# 6. track_memory=True picks up a ~1MB allocation
# ---------------------------------------------------------------------------


def test_track_memory_true_detects_one_mb_alloc():
    prof = OverheadProfiler(track_memory=True)
    with prof.time("alloc"):
        # Force ~1 MB of *python-tracked* allocation. A list of ints is the
        # cleanest cross-platform option (each int in a list is ~28 B + ptr).
        big = [0] * (250_000)
        # Touch it so optimizer can't elide.
        big[-1] = 1
    s = prof.summarize()["alloc"]
    assert s["mem_mb"] > 0.5, f"expected >0.5 MB, got {s['mem_mb']}"


# ---------------------------------------------------------------------------
# 7. track_memory=False ⇒ mem_mb == 0.0
# ---------------------------------------------------------------------------


def test_track_memory_false_yields_zero_mem():
    prof = OverheadProfiler(track_memory=False)
    with prof.time("alloc"):
        _big = [0] * 100_000  # noqa: F841
    s = prof.summarize()["alloc"]
    assert s["mem_mb"] == 0.0


# ---------------------------------------------------------------------------
# 8. reset() clears samples
# ---------------------------------------------------------------------------


def test_reset_clears_samples():
    prof = OverheadProfiler(track_memory=False)
    for _ in range(4):
        with prof.time("x"):
            pass
    assert len(prof.samples_ns("x")) == 4
    prof.reset()
    assert prof.samples_ns("x") == []
    assert prof.summarize() == {}
    assert prof.rows() == []


# ---------------------------------------------------------------------------
# 9. to_markdown() includes a header and each component name
# ---------------------------------------------------------------------------


def test_to_markdown_includes_header_and_names():
    prof = OverheadProfiler(track_memory=False)
    for name in ("preference_plane_get", "rl_arbiter_act"):
        prof._record_sample(name, elapsed_ns=1_000_000, mem_delta_bytes=0)
    md = prof.to_markdown()
    assert isinstance(md, str)
    # Standard markdown table header marker.
    assert "|" in md
    assert "---" in md
    # Both component names appear in the table body.
    assert "preference_plane_get" in md
    assert "rl_arbiter_act" in md


# ---------------------------------------------------------------------------
# 10. to_csv() writes per-sample rows with the required schema
# ---------------------------------------------------------------------------


def test_to_csv_writes_per_sample_rows(tmp_path):
    prof = OverheadProfiler(track_memory=False)
    prof._record_sample("a", elapsed_ns=1_000_000, mem_delta_bytes=0)
    prof.step_marker()
    prof._record_sample("a", elapsed_ns=2_000_000, mem_delta_bytes=0)
    prof._record_sample("b", elapsed_ns=3_000_000, mem_delta_bytes=0)
    out = tmp_path / "overhead.csv"
    prof.to_csv(out)
    assert out.exists()
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    required = {"component", "step", "elapsed_ns", "mem_delta_bytes"}
    assert required.issubset(rows[0].keys()), f"got cols {rows[0].keys()}"


# ---------------------------------------------------------------------------
# 11. step_marker() advances the step index
# ---------------------------------------------------------------------------


def test_step_marker_separates_step_indices():
    prof = OverheadProfiler(track_memory=False)
    prof._record_sample("rl", elapsed_ns=1_000, mem_delta_bytes=0)
    prof.step_marker()
    prof._record_sample("rl", elapsed_ns=2_000, mem_delta_bytes=0)
    rows = prof.rows()
    steps = sorted({r["step"] for r in rows})
    assert steps == [0, 1]


# ---------------------------------------------------------------------------
# 12. Framework backward compat: profiler kwarg is optional
# ---------------------------------------------------------------------------


def test_framework_backward_compat_no_profiler():
    arbiter = _StubArbiter(action=2)
    telemetry = _StubTelemetry([HardwareTelemetry(latency_ema_ms=1.0,
                                                   energy_remaining_j=100.0,
                                                   memory_util=0.1)])
    fw = _make_framework(arbiter, telemetry)  # no profiler
    # No crash on attribute access; default must be None.
    assert getattr(fw, "profiler", "MISSING") is None
    rec = fw.step(np.zeros(4, dtype=np.float32))
    # Existing record schema preserved (sample of pre-W8 keys).
    for key in ("action", "proposed_action", "omega", "override_fired",
                "reward", "latency_ms", "energy_j", "memory_util",
                "dvfs_idx", "concurrent_dvfs_used"):
        assert key in rec, f"profiler=None broke pre-existing record key {key}"


# ---------------------------------------------------------------------------
# 13. Framework with profiler records all 6 in-step components
# ---------------------------------------------------------------------------


def test_framework_with_profiler_records_six_components():
    arbiter = _StubArbiter(action=1)
    telemetry = _StubTelemetry([HardwareTelemetry(latency_ema_ms=1.0,
                                                   energy_remaining_j=100.0,
                                                   memory_util=0.1)])
    dvfs = _StubDvfsController(n_levels=4)
    prof = OverheadProfiler(track_memory=False)
    fw = _make_framework(arbiter, telemetry, dvfs_controller=dvfs, profiler=prof)
    fw.step(np.zeros(4, dtype=np.float32))
    s = prof.summarize()
    expected = {
        "preference_plane_get",
        "tegra_daemon_sample",
        "rl_arbiter_act",
        "override_layer_step",
        "resource_manager_decide",
        "dvfs_controller_set",
    }
    assert expected.issubset(set(s.keys())), \
        f"missing components in profile: {expected - set(s.keys())}"


# ---------------------------------------------------------------------------
# 14. Framework with profiler but no dvfs_controller skips DVFS components
# ---------------------------------------------------------------------------


def test_framework_with_profiler_no_dvfs_skips_dvfs_components():
    arbiter = _StubArbiter(action=1)
    telemetry = _StubTelemetry([HardwareTelemetry(latency_ema_ms=1.0,
                                                   energy_remaining_j=100.0,
                                                   memory_util=0.1)])
    prof = OverheadProfiler(track_memory=False)
    fw = _make_framework(arbiter, telemetry, dvfs_controller=None, profiler=prof)
    fw.step(np.zeros(4, dtype=np.float32))
    s = prof.summarize()
    assert "dvfs_controller_set" not in s
    assert "resource_manager_decide" not in s
    # But the always-on components remain.
    for name in ("preference_plane_get", "tegra_daemon_sample",
                 "rl_arbiter_act", "override_layer_step"):
        assert name in s


# ---------------------------------------------------------------------------
# Bonus: ComponentTimer is the type returned by prof.time(...)
# ---------------------------------------------------------------------------


def test_time_returns_component_timer_instance():
    prof = OverheadProfiler(track_memory=False)
    cm = prof.time("x")
    assert isinstance(cm, ComponentTimer)
