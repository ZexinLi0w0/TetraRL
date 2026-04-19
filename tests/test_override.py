"""Tests for the hardware-emergency override layer."""
from __future__ import annotations

import numpy as np

from tetrarl.morl.native.override import (
    HardwareTelemetry,
    OverrideLayer,
    OverrideThresholds,
)


def _make(thresholds: OverrideThresholds, fallback=0, cooldown_steps: int = 0):
    return OverrideLayer(
        thresholds=thresholds,
        fallback_action=fallback,
        cooldown_steps=cooldown_steps,
    )


def test_no_thresholds_never_fires():
    layer = _make(OverrideThresholds())
    tel = HardwareTelemetry(
        latency_ema_ms=1e6,
        energy_remaining_j=-1e6,
        memory_util=1.0,
    )
    fired, fb = layer.step(tel)
    assert fired is False
    assert fb is None
    assert layer.fire_count == 0
    assert layer.last_reasons == []


def test_no_telemetry_never_fires():
    layer = _make(OverrideThresholds(
        max_latency_ms=100.0,
        min_energy_j=10.0,
        max_memory_util=0.9,
    ))
    fired, fb = layer.step(HardwareTelemetry())
    assert fired is False
    assert fb is None
    assert layer.fire_count == 0


def test_latency_threshold_fires():
    layer = _make(OverrideThresholds(max_latency_ms=100.0), fallback=0)
    fired, fb = layer.step(HardwareTelemetry(latency_ema_ms=120.0))
    assert fired is True
    assert fb == 0
    assert layer.fire_count == 1
    assert any("latency" in r for r in layer.last_reasons)


def test_latency_under_threshold_does_not_fire():
    layer = _make(OverrideThresholds(max_latency_ms=100.0))
    fired, fb = layer.step(HardwareTelemetry(latency_ema_ms=50.0))
    assert fired is False
    assert fb is None
    assert layer.fire_count == 0


def test_energy_threshold_fires():
    layer = _make(OverrideThresholds(min_energy_j=10.0), fallback=0)
    fired, fb = layer.step(HardwareTelemetry(energy_remaining_j=5.0))
    assert fired is True
    assert fb == 0
    assert any("energy" in r for r in layer.last_reasons)


def test_memory_threshold_fires():
    layer = _make(OverrideThresholds(max_memory_util=0.9), fallback=0)
    fired, fb = layer.step(HardwareTelemetry(memory_util=0.95))
    assert fired is True
    assert fb == 0
    assert any("memory" in r for r in layer.last_reasons)


def test_multiple_violations_all_reported():
    layer = _make(OverrideThresholds(
        max_latency_ms=100.0,
        min_energy_j=10.0,
    ))
    tel = HardwareTelemetry(latency_ema_ms=200.0, energy_remaining_j=1.0)
    fired, _ = layer.step(tel)
    assert fired is True
    assert len(layer.last_reasons) >= 2


def test_cooldown_keeps_firing():
    layer = _make(OverrideThresholds(max_latency_ms=100.0),
                  fallback=0, cooldown_steps=2)
    bad = HardwareTelemetry(latency_ema_ms=200.0)
    healthy = HardwareTelemetry(latency_ema_ms=10.0)

    # First call: bad telemetry fires, arms cooldown=2
    fired, _ = layer.step(bad)
    assert fired is True
    # Next 2 calls: healthy telemetry but cooldown keeps firing
    fired, _ = layer.step(healthy)
    assert fired is True
    fired, _ = layer.step(healthy)
    assert fired is True
    # 4th call: cooldown exhausted, healthy -> no fire
    fired, fb = layer.step(healthy)
    assert fired is False
    assert fb is None
    assert layer.fire_count == 3


def test_cooldown_zero_no_hysteresis():
    layer = _make(OverrideThresholds(max_latency_ms=100.0),
                  fallback=0, cooldown_steps=0)
    fired, _ = layer.step(HardwareTelemetry(latency_ema_ms=200.0))
    assert fired is True
    fired, fb = layer.step(HardwareTelemetry(latency_ema_ms=10.0))
    assert fired is False
    assert fb is None


def test_reset_clears_state():
    layer = _make(OverrideThresholds(max_latency_ms=100.0),
                  fallback=0, cooldown_steps=5)
    layer.step(HardwareTelemetry(latency_ema_ms=200.0))
    layer.step(HardwareTelemetry(latency_ema_ms=200.0))
    assert layer.fire_count > 0
    assert layer.last_reasons != []

    layer.reset()
    assert layer.fire_count == 0
    assert layer.last_reasons == []
    # Cooldown also cleared: a healthy tick should NOT fire
    fired, fb = layer.step(HardwareTelemetry(latency_ema_ms=10.0))
    assert fired is False
    assert fb is None


def test_fallback_action_passed_through():
    # int fallback
    layer_int = _make(OverrideThresholds(max_latency_ms=100.0), fallback=0)
    _, fb_int = layer_int.step(HardwareTelemetry(latency_ema_ms=200.0))
    assert fb_int == 0
    assert isinstance(fb_int, int)

    # numpy array fallback
    arr = np.array([0.0, 0.0])
    layer_arr = _make(OverrideThresholds(max_latency_ms=100.0), fallback=arr)
    _, fb_arr = layer_arr.step(HardwareTelemetry(latency_ema_ms=200.0))
    assert isinstance(fb_arr, np.ndarray)
    assert np.array_equal(fb_arr, arr)
    # Same object passed through unchanged
    assert fb_arr is arr

    # arbitrary object fallback
    sentinel = {"action": "safe", "value": object()}
    layer_obj = _make(OverrideThresholds(max_latency_ms=100.0), fallback=sentinel)
    _, fb_obj = layer_obj.step(HardwareTelemetry(latency_ema_ms=200.0))
    assert fb_obj is sentinel
