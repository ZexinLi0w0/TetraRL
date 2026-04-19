"""Tests for the DVFS controller (stub mode for Mac dev)."""
from __future__ import annotations

import pytest

from tetrarl.sys.dvfs import (
    DVFSConfig,
    DVFSController,
    STUB_CPU_FREQS_KHZ,
    STUB_GPU_FREQS_HZ,
    TransitionLatency,
)


def test_auto_detects_stub_mode_on_mac():
    ctrl = DVFSController()
    assert ctrl.stub is True


def test_available_frequencies_in_stub_mode():
    ctrl = DVFSController(stub=True)
    avail = ctrl.available_frequencies()
    assert "cpu" in avail and "gpu" in avail
    assert avail["cpu"] == list(STUB_CPU_FREQS_KHZ)
    assert avail["gpu"] == list(STUB_GPU_FREQS_HZ)
    assert avail["cpu"] == sorted(avail["cpu"])
    assert avail["gpu"] == sorted(avail["gpu"])


def test_set_cpu_freq_updates_state():
    ctrl = DVFSController(stub=True)
    new_state = ctrl.set_freq(cpu_idx=0)
    assert new_state.cpu_freq_khz == STUB_CPU_FREQS_KHZ[0]
    assert ctrl.current_state().cpu_freq_khz == STUB_CPU_FREQS_KHZ[0]


def test_set_gpu_freq_updates_state():
    ctrl = DVFSController(stub=True)
    new_state = ctrl.set_freq(gpu_idx=2)
    assert new_state.gpu_freq_hz == STUB_GPU_FREQS_HZ[2]


def test_set_freq_both_axes():
    ctrl = DVFSController(stub=True)
    state = ctrl.set_freq(cpu_idx=3, gpu_idx=5)
    assert state.cpu_freq_khz == STUB_CPU_FREQS_KHZ[3]
    assert state.gpu_freq_hz == STUB_GPU_FREQS_HZ[5]


def test_set_freq_no_args_returns_current():
    ctrl = DVFSController(stub=True)
    before = ctrl.current_state()
    after = ctrl.set_freq()
    assert before == after


def test_invalid_cpu_idx_raises():
    ctrl = DVFSController(stub=True)
    with pytest.raises(IndexError):
        ctrl.set_freq(cpu_idx=999)
    with pytest.raises(IndexError):
        ctrl.set_freq(cpu_idx=-1)


def test_invalid_gpu_idx_raises():
    ctrl = DVFSController(stub=True)
    with pytest.raises(IndexError):
        ctrl.set_freq(gpu_idx=999)


def test_profile_transition_latency_returns_full_pair_table():
    ctrl = DVFSController(stub=True)
    results = ctrl.profile_transition_latency(domain="cpu", n_iters=1)
    n = len(STUB_CPU_FREQS_KHZ)
    assert len(results) == n * (n - 1)
    assert all(isinstance(r, TransitionLatency) for r in results)
    assert all(r.from_freq != r.to_freq for r in results)
    assert all(r.latency_ms >= 0.0 for r in results)


def test_profile_transition_latency_invalid_domain():
    ctrl = DVFSController(stub=True)
    with pytest.raises(ValueError):
        ctrl.profile_transition_latency(domain="nope")


def test_profile_transition_latency_zero_iters_raises():
    ctrl = DVFSController(stub=True)
    with pytest.raises(ValueError):
        ctrl.profile_transition_latency(n_iters=0)


def test_dvfs_config_dataclass_equality():
    a = DVFSConfig(cpu_freq_khz=1000, gpu_freq_hz=500_000_000)
    b = DVFSConfig(cpu_freq_khz=1000, gpu_freq_hz=500_000_000)
    assert a == b


def test_real_mode_uses_overridden_paths(tmp_path):
    cpu_avail = tmp_path / "cpu_avail"
    cpu_set = tmp_path / "cpu_set"
    cpu_cur = tmp_path / "cpu_cur"
    gpu_avail = tmp_path / "gpu_avail"
    gpu_min = tmp_path / "gpu_min"
    gpu_max = tmp_path / "gpu_max"
    gpu_cur = tmp_path / "gpu_cur"

    cpu_avail.write_text("100000 200000 300000\n")
    cpu_cur.write_text("200000\n")
    gpu_avail.write_text("500000000 800000000\n")
    gpu_cur.write_text("500000000\n")

    ctrl = DVFSController(
        stub=False,
        cpu_paths={
            "available": str(cpu_avail),
            "setspeed": str(cpu_set),
            "cur": str(cpu_cur),
            "governor": str(tmp_path / "gov"),
        },
        gpu_paths={
            "available": str(gpu_avail),
            "min": str(gpu_min),
            "max": str(gpu_max),
            "cur": str(gpu_cur),
        },
    )
    avail = ctrl.available_frequencies()
    assert avail["cpu"] == [100000, 200000, 300000]
    assert avail["gpu"] == [500000000, 800000000]

    ctrl.set_freq(cpu_idx=2, gpu_idx=1)
    assert cpu_set.read_text() == "300000"
    assert gpu_min.read_text() == "800000000"
    assert gpu_max.read_text() == "800000000"
