"""Tests for the TetraRLFramework 4-component orchestrator.

The framework wires (i) Preference Plane, (ii) Resource Manager,
(iii) RL Arbiter, (iv) Hardware Override Layer into a single step()
pipeline that returns the action that should actually be executed,
plus a per-step telemetry record with all 4 reward dimensions.

We use mocks for the heavy components so this test stays fast and
deterministic; an end-to-end smoke test on CartPole lives separately.
"""
from __future__ import annotations

import numpy as np
import pytest

from tetrarl.core.framework import (
    TetraRLFramework,
    StaticPreferencePlane,
    ResourceManager,
)
from tetrarl.morl.native.override import (
    HardwareTelemetry,
    OverrideLayer,
    OverrideThresholds,
)


class _StubArbiter:
    """Minimal RL Arbiter: returns a fixed action, records calls."""

    def __init__(self, action=1):
        self.action = action
        self.calls = []

    def act(self, state, omega):
        self.calls.append((np.asarray(state).copy(), np.asarray(omega).copy()))
        return self.action


class _StubTelemetry:
    """Returns a programmable telemetry sequence."""

    def __init__(self, readings):
        self._readings = list(readings)
        self._i = 0

    def latest(self):
        if self._i >= len(self._readings):
            return self._readings[-1] if self._readings else None
        r = self._readings[self._i]
        self._i += 1
        return r


def _telemetry_to_hw(reading):
    """Adapter: TegrastatsReading-like -> HardwareTelemetry."""
    if reading is None:
        return HardwareTelemetry()
    return HardwareTelemetry(
        latency_ema_ms=getattr(reading, "latency_ema_ms", None),
        energy_remaining_j=getattr(reading, "energy_remaining_j", None),
        memory_util=getattr(reading, "memory_util", None),
    )


def _make_framework(arbiter, telemetry, override=None, omega=None):
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
    )


def test_step_returns_arbiter_action_when_override_quiet():
    arbiter = _StubArbiter(action=2)
    telemetry = _StubTelemetry([HardwareTelemetry(latency_ema_ms=1.0,
                                                   energy_remaining_j=100.0,
                                                   memory_util=0.1)])
    fw = _make_framework(arbiter, telemetry)
    state = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    record = fw.step(state)
    assert record["action"] == 2
    assert record["override_fired"] is False


def test_step_overrides_when_threshold_breached():
    arbiter = _StubArbiter(action=2)
    telemetry = _StubTelemetry([HardwareTelemetry(latency_ema_ms=999.0)])
    override = OverrideLayer(OverrideThresholds(max_latency_ms=10.0),
                             fallback_action=0)
    fw = _make_framework(arbiter, telemetry, override=override)
    state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    record = fw.step(state)
    assert record["action"] == 0
    assert record["override_fired"] is True


def test_step_passes_omega_to_arbiter():
    omega = np.array([0.7, 0.3], dtype=np.float32)
    arbiter = _StubArbiter(action=0)
    telemetry = _StubTelemetry([HardwareTelemetry()])
    fw = _make_framework(arbiter, telemetry, omega=omega)
    fw.step(np.zeros(4, dtype=np.float32))
    assert len(arbiter.calls) == 1
    _, recorded_omega = arbiter.calls[0]
    np.testing.assert_allclose(recorded_omega, omega)


def test_record_contains_4d_reward_fields():
    """Per Week 6 spec: framework must expose all 4 dimensions per step."""
    arbiter = _StubArbiter(action=0)
    telemetry = _StubTelemetry([HardwareTelemetry(latency_ema_ms=1.5,
                                                   energy_remaining_j=80.0,
                                                   memory_util=0.4)])
    fw = _make_framework(arbiter, telemetry)
    record = fw.step(np.zeros(4, dtype=np.float32))
    # The record must report each of the 4 dimensions at every step.
    for key in ("reward", "latency_ms", "energy_j", "memory_util"):
        assert key in record, f"missing {key} in framework step record"


def test_observe_reward_attaches_to_last_record():
    """The env's reward arrives AFTER step(); framework provides observe()."""
    arbiter = _StubArbiter(action=0)
    telemetry = _StubTelemetry([HardwareTelemetry()])
    fw = _make_framework(arbiter, telemetry)
    rec = fw.step(np.zeros(4, dtype=np.float32))
    fw.observe_reward(1.25)
    assert rec["reward"] == pytest.approx(1.25)


def test_resource_manager_decides_dvfs_target_index():
    """ResourceManager.decide_dvfs returns an int index into the freq table."""
    rm = ResourceManager()
    # Healthy telemetry -> top-of-range (max performance).
    idx_hi = rm.decide_dvfs(HardwareTelemetry(latency_ema_ms=1.0,
                                               energy_remaining_j=100.0,
                                               memory_util=0.1),
                              n_levels=10)
    # Stressed telemetry -> bottom of range (save energy/avoid OOM).
    idx_lo = rm.decide_dvfs(HardwareTelemetry(latency_ema_ms=200.0,
                                               energy_remaining_j=1.0,
                                               memory_util=0.95),
                              n_levels=10)
    assert 0 <= idx_lo <= idx_hi < 10
    assert idx_lo < idx_hi


def test_static_preference_plane_returns_same_omega():
    omega = np.array([0.6, 0.4], dtype=np.float32)
    plane = StaticPreferencePlane(omega)
    np.testing.assert_allclose(plane.get(), omega)
    np.testing.assert_allclose(plane.get(), omega)  # stable across calls


def test_history_accumulates_records():
    arbiter = _StubArbiter(action=0)
    telemetry = _StubTelemetry([
        HardwareTelemetry(latency_ema_ms=1.0),
        HardwareTelemetry(latency_ema_ms=2.0),
        HardwareTelemetry(latency_ema_ms=3.0),
    ])
    fw = _make_framework(arbiter, telemetry)
    for _ in range(3):
        fw.step(np.zeros(4, dtype=np.float32))
    assert len(fw.history) == 3
