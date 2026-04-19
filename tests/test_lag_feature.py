"""Tests for the NeuOS LAG (Latency Above Ground-truth) feature extractor.

LAG (Bateni & Liu, RTAS 2020) is the per-DNN ratio of the current
inference latency to its target latency: ``lag = latency / target``.
NeuOS uses this as an additional state feature for multi-DNN co-running
schedulers; when ``lag > 1`` the corunner is missing its deadline and
the policy should learn to de-prioritize new arrivals.

For TetraRL the single-task case computes
``telemetry.latency_ema_ms / soft_latency_ms`` directly from
:class:`HardwareTelemetry`; the multi-corunner case accepts an explicit
list of corunner latencies and returns one ratio per corunner.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from tetrarl.morl.native.lag_feature import LAGFeatureExtractor
from tetrarl.morl.native.override import HardwareTelemetry

# ---------------------------------------------------------------------------
# 1. Single-task LAG: latency / soft target
# ---------------------------------------------------------------------------


def test_single_task_lag_ratio_matches_latency_over_soft():
    extractor = LAGFeatureExtractor(soft_latency_ms=50.0, n_corunners=1)
    tel = HardwareTelemetry(latency_ema_ms=25.0)
    out = extractor.extract(tel)
    assert out.shape == (1,)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.array([0.5], dtype=np.float32))


# ---------------------------------------------------------------------------
# 2. Single-task with no telemetry latency -> zero LAG
# ---------------------------------------------------------------------------


def test_single_task_telemetry_none_returns_zero():
    extractor = LAGFeatureExtractor(soft_latency_ms=50.0, n_corunners=1)
    tel = HardwareTelemetry(latency_ema_ms=None)
    out = extractor.extract(tel)
    assert out.shape == (1,)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.zeros(1, dtype=np.float32))


# ---------------------------------------------------------------------------
# 3. Multi-corunner: per-corunner ratios
# ---------------------------------------------------------------------------


def test_multi_corunner_per_corunner_ratios():
    extractor = LAGFeatureExtractor(soft_latency_ms=20.0, n_corunners=3)
    tel = HardwareTelemetry(latency_ema_ms=None)
    out = extractor.extract(tel, corunner_latencies_ms=[10.0, 20.0, 40.0])
    assert out.shape == (3,)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.array([0.5, 1.0, 2.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# 4. Multi-corunner: length mismatch -> ValueError
# ---------------------------------------------------------------------------


def test_multi_corunner_length_mismatch_raises():
    extractor = LAGFeatureExtractor(soft_latency_ms=20.0, n_corunners=3)
    tel = HardwareTelemetry(latency_ema_ms=None)
    with pytest.raises(ValueError):
        extractor.extract(tel, corunner_latencies_ms=[10.0, 20.0])


# ---------------------------------------------------------------------------
# 5. Clip cap at clip_max for numerical stability
# ---------------------------------------------------------------------------


def test_clip_max_caps_extreme_lag():
    extractor = LAGFeatureExtractor(soft_latency_ms=10.0, n_corunners=1, clip_max=10.0)
    tel = HardwareTelemetry(latency_ema_ms=10_000.0)
    out = extractor.extract(tel)
    np.testing.assert_allclose(out, np.array([10.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# 6. clip_max=None disables the cap
# ---------------------------------------------------------------------------


def test_clip_max_none_disables_capping():
    extractor = LAGFeatureExtractor(soft_latency_ms=10.0, n_corunners=1, clip_max=None)
    tel = HardwareTelemetry(latency_ema_ms=10_000.0)
    out = extractor.extract(tel)
    np.testing.assert_allclose(out, np.array([1000.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# 7. Zero soft target rejected at construction
# ---------------------------------------------------------------------------


def test_zero_soft_latency_raises_at_construction():
    with pytest.raises(ValueError):
        LAGFeatureExtractor(soft_latency_ms=0.0, n_corunners=1)


# ---------------------------------------------------------------------------
# 8. Negative soft target rejected at construction
# ---------------------------------------------------------------------------


def test_negative_soft_latency_raises_at_construction():
    with pytest.raises(ValueError):
        LAGFeatureExtractor(soft_latency_ms=-1.0, n_corunners=1)


# ---------------------------------------------------------------------------
# 9. append_to_state appends LAG to existing state
# ---------------------------------------------------------------------------


def test_append_to_state_concatenates_lag_to_existing_state():
    extractor = LAGFeatureExtractor(soft_latency_ms=50.0, n_corunners=1)
    state = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    tel = HardwareTelemetry(latency_ema_ms=25.0)
    out = extractor.append_to_state(state, tel)
    assert out.shape == (5,)
    assert out.dtype == np.float32
    # First 4 elements unchanged, last element is the LAG ratio.
    np.testing.assert_allclose(out[:4], state)
    assert out[-1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 10. feature_dim equals n_corunners
# ---------------------------------------------------------------------------


def test_feature_dim_matches_n_corunners():
    extractor = LAGFeatureExtractor(soft_latency_ms=20.0, n_corunners=3)
    assert extractor.feature_dim == 3


# ---------------------------------------------------------------------------
# 11. Per-call overhead: must stay under 500 microseconds (W8 budget)
# ---------------------------------------------------------------------------


def test_extract_overhead_under_500us_per_call():
    extractor = LAGFeatureExtractor(soft_latency_ms=50.0, n_corunners=1)
    tel = HardwareTelemetry(latency_ema_ms=25.0)
    n = 1000
    t0 = time.perf_counter_ns()
    for _ in range(n):
        extractor.extract(tel)
    elapsed_ns = time.perf_counter_ns() - t0
    mean_us = (elapsed_ns / n) / 1_000.0
    assert mean_us < 500.0, f"mean per-call overhead {mean_us:.2f} us exceeds 500 us"
