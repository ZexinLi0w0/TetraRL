"""Tests for the DVFS controller (stub mode for Mac dev)."""
from __future__ import annotations

import pytest

from tetrarl.sys.dvfs import (
    STUB_CPU_FREQS_KHZ,
    STUB_GPU_FREQS_HZ,
    DVFSConfig,
    DVFSController,
    TransitionLatency,
)
from tetrarl.sys.platforms import Platform


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


# --- Nano / multi-platform tests --------------------------------------------


def test_nano_profile_selects_nano_freq_tables():
    ctrl = DVFSController(platform=Platform.NANO, stub=True)
    avail = ctrl.available_frequencies()
    # Nano endpoints from L4T 32.7: bottom CPU = 102 MHz, top GPU = 921.6 MHz.
    assert avail["cpu"][0] == 102_000
    assert avail["gpu"][-1] == 921_600_000


def test_nano_set_freq_in_stub_mode_writes_no_file():
    # We cannot positively assert "no file written" without monkeypatching the
    # filesystem; the spec instead asks us to confirm set_freq on a Nano stub
    # controller returns a DVFSConfig populated with Nano-table frequencies
    # (which implicitly proves we never went through the real-mode write path,
    # since those sysfs nodes do not exist on this Mac dev box).
    ctrl = DVFSController(platform=Platform.NANO, stub=True)
    state = ctrl.set_freq(cpu_idx=5, gpu_idx=3)
    nano_cpu = ctrl.profile.cpu_freqs_hz
    nano_gpu = ctrl.profile.gpu_freqs_hz
    assert state.cpu_freq_khz == nano_cpu[5]
    assert state.gpu_freq_hz == nano_gpu[3]


def test_nano_and_orin_have_distinct_freq_tables():
    nano = DVFSController(platform=Platform.NANO, stub=True)
    orin = DVFSController(platform=Platform.ORIN_AGX, stub=True)
    assert (
        nano.available_frequencies()["cpu"]
        != orin.available_frequencies()["cpu"]
    )
    assert (
        nano.available_frequencies()["gpu"]
        != orin.available_frequencies()["gpu"]
    )


def test_backward_compat_default_is_orin_agx():
    ctrl = DVFSController(stub=True)
    avail = ctrl.available_frequencies()
    # Top of the existing Orin AGX stub table.
    assert avail["cpu"][-1] == 2_188_800
    assert avail["cpu"] == list(STUB_CPU_FREQS_KHZ)
    assert avail["gpu"] == list(STUB_GPU_FREQS_HZ)


def test_string_platform_arg_works():
    by_str = DVFSController(platform="nano", stub=True)
    by_enum = DVFSController(platform=Platform.NANO, stub=True)
    assert by_str.available_frequencies() == by_enum.available_frequencies()


def test_nano_real_mode_paths_use_57000000_gpu_node():
    # Nano sysfs writes only succeed on a physical Linux Nano (the spec
    # explicitly defers physical-Nano sysfs write tests). Here we just
    # inspect the derived gpu_paths to confirm we point at the right
    # devfreq node (57000000.gpu) rather than the Orin one (17000000.gpu).
    ctrl = DVFSController(platform=Platform.NANO, stub=True)
    gpu_paths = ctrl.gpu_paths
    assert "57000000.gpu" in gpu_paths["min"]
    assert "57000000.gpu" in gpu_paths["max"]
    assert "57000000.gpu" in gpu_paths["cur"]
    assert "17000000.gpu" not in gpu_paths["min"]


def test_nano_top_freqs_below_orin():
    nano = DVFSController(platform=Platform.NANO, stub=True)
    orin = DVFSController(platform=Platform.ORIN_AGX, stub=True)
    n_avail = nano.available_frequencies()
    o_avail = orin.available_frequencies()
    assert n_avail["cpu"][-1] < o_avail["cpu"][-1]
    assert n_avail["gpu"][-1] < o_avail["gpu"][-1]


# --- Orin Nano (8 GB) tests --------------------------------------------------


def test_orin_nano_profile_selects_orin_nano_freq_tables():
    ctrl = DVFSController(platform=Platform.ORIN_NANO, stub=True)
    avail = ctrl.available_frequencies()
    assert avail["cpu"][0] == 115_200
    assert avail["cpu"][-1] == 1_510_400
    assert avail["gpu"][0] == 306_000_000
    assert avail["gpu"][-1] == 624_750_000


def test_orin_nano_set_freq_uses_orin_nano_table():
    ctrl = DVFSController(platform=Platform.ORIN_NANO, stub=True)
    state = ctrl.set_freq(cpu_idx=10, gpu_idx=2)
    orin_nano_cpu = ctrl.profile.cpu_freqs_hz
    orin_nano_gpu = ctrl.profile.gpu_freqs_hz
    assert state.cpu_freq_khz == orin_nano_cpu[10]
    assert state.gpu_freq_hz == orin_nano_gpu[2]


def test_orin_nano_string_arg_works():
    by_str = DVFSController(platform="orin_nano", stub=True)
    by_enum = DVFSController(platform=Platform.ORIN_NANO, stub=True)
    assert by_str.available_frequencies() == by_enum.available_frequencies()


def test_orin_nano_gpu_paths_use_ga10b_address():
    ctrl = DVFSController(platform=Platform.ORIN_NANO, stub=True)
    gpu_paths = ctrl.gpu_paths
    assert "17000000.ga10b" in gpu_paths["min"]
    assert "17000000.ga10b" in gpu_paths["max"]
    assert "17000000.ga10b" in gpu_paths["cur"]
