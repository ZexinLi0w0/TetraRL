"""Tests for the tegrastats async sensor daemon."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from tetrarl.sys.platforms import Platform, get_profile
from tetrarl.sys.tegra_daemon import (
    TegrastatsDaemon,
    TegrastatsReading,
    parse_tegrastats_line,
)


FIXTURE = Path(__file__).parent / "fixtures" / "tegra_sample.txt"
NANO_FIXTURE = Path(__file__).parent / "fixtures" / "tegrastats_nano_sample.txt"


def test_parse_basic_fields():
    line = (
        "04-18-2026 09:08:28 RAM 4093/30596MB (lfb 7x4MB) SWAP 0/15298MB "
        "(cached 0MB) CPU [9%@2188,4%@2188] EMC_FREQ 2%@2133 "
        "GR3D_FREQ 25%@[1377] CPU@49C GPU@48.281C "
        "VDD_GPU_SOC 6429mW/6429mW VDD_CPU_CV 1235mW/1235mW"
    )
    r = parse_tegrastats_line(line)
    assert r is not None
    assert r.ram_used_mb == 4093
    assert r.ram_total_mb == 30596
    assert r.gr3d_freq_pct == pytest.approx(25.0)
    assert r.gpu_freq_mhz == 1377
    assert r.gpu_temp_c == pytest.approx(48.281)
    assert r.vdd_gpu_soc_mw == 6429
    assert r.vdd_cpu_cv_mw == 1235
    assert r.cpu_util_per_core == [9.0, 4.0]
    assert r.cpu_freq_mhz == 2188


def test_parse_handles_missing_optional_fields():
    line = "RAM 100/200MB CPU [50%@1000]"
    r = parse_tegrastats_line(line)
    assert r is not None
    assert r.ram_used_mb == 100
    assert r.ram_total_mb == 200
    assert r.cpu_util_per_core == [50.0]
    assert r.gr3d_freq_pct == 0.0
    assert r.vdd_gpu_soc_mw == 0


def test_parse_returns_none_for_garbage():
    assert parse_tegrastats_line("not-a-tegrastats-line") is None
    assert parse_tegrastats_line("") is None


def test_fixture_file_parses_all_lines():
    lines = FIXTURE.read_text().strip().splitlines()
    parsed = [parse_tegrastats_line(line) for line in lines]
    assert all(p is not None for p in parsed)
    assert parsed[-1].gr3d_freq_pct == pytest.approx(75.0)
    assert parsed[-1].vdd_gpu_soc_mw == 7400


def test_daemon_file_source_dispatches_readings():
    received: list[TegrastatsReading] = []

    daemon = TegrastatsDaemon(
        source=f"file:{FIXTURE}",
        sample_hz=200.0,
        dispatch_hz=200.0,
        ema_alpha=1.0,
        on_dispatch=received.append,
    )
    daemon.start()
    time.sleep(0.3)
    daemon.stop()

    assert len(received) >= 1
    assert received[-1].ram_total_mb == 30596


def test_daemon_ema_filter_smooths_values():
    daemon = TegrastatsDaemon(
        source=f"file:{FIXTURE}",
        sample_hz=200.0,
        dispatch_hz=50.0,
        ema_alpha=0.1,
    )
    daemon.start()
    time.sleep(0.2)
    daemon.stop()
    latest = daemon.latest()
    assert latest is not None
    assert 0.0 <= latest.gr3d_freq_pct < 75.0


def test_daemon_stop_is_idempotent():
    daemon = TegrastatsDaemon(
        source=f"file:{FIXTURE}",
        sample_hz=100.0,
        dispatch_hz=10.0,
    )
    daemon.start()
    daemon.stop()
    daemon.stop()


def test_daemon_latest_before_start_returns_none():
    daemon = TegrastatsDaemon(source=f"file:{FIXTURE}")
    assert daemon.latest() is None


def test_daemon_dispatch_rate_below_sample_rate():
    received: list[TegrastatsReading] = []
    daemon = TegrastatsDaemon(
        source=f"file:{FIXTURE}",
        sample_hz=1000.0,
        dispatch_hz=100.0,
        on_dispatch=received.append,
    )
    daemon.start()
    time.sleep(0.3)
    daemon.stop()
    assert 5 <= len(received) <= 200


def test_ema_alpha_one_is_passthrough():
    daemon = TegrastatsDaemon(
        source=f"file:{FIXTURE}",
        sample_hz=200.0,
        dispatch_hz=200.0,
        ema_alpha=1.0,
    )
    daemon.start()
    time.sleep(0.2)
    daemon.stop()
    latest = daemon.latest()
    assert latest is not None
    assert latest.gr3d_freq_pct in {0.0, 15.0, 50.0, 75.0}


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        TegrastatsDaemon(ema_alpha=0.0)
    with pytest.raises(ValueError):
        TegrastatsDaemon(ema_alpha=1.5)


def test_invalid_dispatch_rate_raises():
    with pytest.raises(ValueError):
        TegrastatsDaemon(sample_hz=10.0, dispatch_hz=100.0)


def test_nano_fixture_parses_with_nano_layout():
    lines = NANO_FIXTURE.read_text().strip().splitlines()
    assert len(lines) == 30
    parsed = [parse_tegrastats_line(line, layout="nano") for line in lines]
    successes = [p for p in parsed if p is not None]
    assert len(successes) >= 25
    assert any(p.vdd_gpu_soc_mw > 0 for p in successes)


def test_nano_fixture_orin_layout_misses_power():
    lines = NANO_FIXTURE.read_text().strip().splitlines()
    parsed = [parse_tegrastats_line(line) for line in lines]  # default "orin"
    successes = [p for p in parsed if p is not None]
    assert len(successes) >= 25
    # POM_5V_* won't match VDD_GPU_SOC / VDD_CPU_CV regexes -> all zero.
    assert all(p.vdd_gpu_soc_mw == 0 for p in successes)
    assert all(p.vdd_cpu_cv_mw == 0 for p in successes)


def test_daemon_nano_platform_uses_nano_alpha_default():
    nano_profile = get_profile(Platform.NANO)
    daemon = TegrastatsDaemon(platform=Platform.NANO)
    assert daemon.ema_alpha == nano_profile.default_ema_alpha


def test_daemon_orin_platform_default_alpha_unchanged():
    assert TegrastatsDaemon().ema_alpha == 0.1
    assert TegrastatsDaemon(platform=Platform.ORIN_AGX).ema_alpha == 0.1


def test_daemon_nano_fixture_end_to_end_dispatches_with_power():
    received: list[TegrastatsReading] = []
    daemon = TegrastatsDaemon(
        platform=Platform.NANO,
        source=f"file:{NANO_FIXTURE}",
        sample_hz=200.0,
        dispatch_hz=200.0,
        ema_alpha=1.0,
        on_dispatch=received.append,
    )
    daemon.start()
    time.sleep(0.3)
    daemon.stop()

    assert len(received) >= 1
    assert any(r.vdd_gpu_soc_mw > 0 or r.vdd_cpu_cv_mw > 0 for r in received)


def test_explicit_ema_alpha_overrides_profile_default():
    daemon = TegrastatsDaemon(platform=Platform.NANO, ema_alpha=0.05)
    assert daemon.ema_alpha == 0.05


def test_daemon_platform_name_attribute():
    nano_profile = get_profile(Platform.NANO)
    daemon = TegrastatsDaemon(platform=Platform.NANO)
    assert daemon.platform_name == nano_profile.name
    assert daemon.platform_name == "Jetson Nano (4 GB)"
