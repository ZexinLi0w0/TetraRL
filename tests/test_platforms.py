"""RED tests for tetrarl.sys.platforms (Week 7 Task 6).

These tests intentionally precede the module's existence: importing
``tetrarl.sys.platforms`` MUST raise ``ImportError`` until the
implementation phase lands. They pin the Platform enum, the
PlatformProfile dataclass schema, and the per-platform frequency tables
for Orin AGX and Jetson Nano.

Unit conventions
----------------
The dataclass field is named ``cpu_freqs_hz`` to match the spec, but the
values are stored as **kHz integers** (e.g. 102_000 means 102 MHz / 102000
kHz). This matches the existing ``STUB_CPU_FREQS_KHZ`` table in
``tetrarl/sys/dvfs.py`` and the cpufreq sysfs node ``scaling_setspeed``,
which expects kHz. ``gpu_freqs_hz`` values are true Hz, matching the
devfreq ``min_freq``/``max_freq`` nodes.
"""
from __future__ import annotations

import pytest

from tetrarl.sys.platforms import (
    PLATFORM_PROFILES,
    Platform,
    PlatformProfile,
    get_profile,
)


def test_platform_enum_string_values():
    assert Platform.ORIN_AGX.value == "orin_agx"
    assert Platform.NANO.value == "nano"


def test_platform_profiles_registry_has_both_platforms():
    assert Platform.ORIN_AGX in PLATFORM_PROFILES
    assert Platform.NANO in PLATFORM_PROFILES
    assert isinstance(PLATFORM_PROFILES[Platform.ORIN_AGX], PlatformProfile)
    assert isinstance(PLATFORM_PROFILES[Platform.NANO], PlatformProfile)


def test_get_profile_accepts_enum_and_string():
    via_enum = get_profile(Platform.NANO)
    via_str = get_profile("nano")
    assert via_enum is via_str or via_enum == via_str

    orin_via_enum = get_profile(Platform.ORIN_AGX)
    orin_via_str = get_profile("orin_agx")
    assert orin_via_enum is orin_via_str or orin_via_enum == orin_via_str


def test_cpu_freq_tables_monotonic_and_nonempty():
    for plat in (Platform.ORIN_AGX, Platform.NANO):
        prof = PLATFORM_PROFILES[plat]
        assert len(prof.cpu_freqs_hz) > 0
        assert prof.cpu_freqs_hz == sorted(prof.cpu_freqs_hz)


def test_gpu_freq_tables_monotonic_and_nonempty():
    for plat in (Platform.ORIN_AGX, Platform.NANO):
        prof = PLATFORM_PROFILES[plat]
        assert len(prof.gpu_freqs_hz) > 0
        assert prof.gpu_freqs_hz == sorted(prof.gpu_freqs_hz)


def test_nano_cpu_table_has_15_entries():
    nano = PLATFORM_PROFILES[Platform.NANO]
    assert len(nano.cpu_freqs_hz) == 15


def test_nano_gpu_table_has_12_entries():
    nano = PLATFORM_PROFILES[Platform.NANO]
    assert len(nano.gpu_freqs_hz) == 12


def test_nano_cpu_range_endpoints():
    """Nano CPU range per NVIDIA L4T 32.7 (values stored as kHz integers)."""
    nano = PLATFORM_PROFILES[Platform.NANO]
    assert nano.cpu_freqs_hz[0] == 102_000
    assert nano.cpu_freqs_hz[-1] == 1_479_000


def test_nano_gpu_range_endpoints():
    nano = PLATFORM_PROFILES[Platform.NANO]
    assert nano.gpu_freqs_hz[0] == 76_800_000
    assert nano.gpu_freqs_hz[-1] == 921_600_000


def test_orin_top_freq_exceeds_nano_top_freq():
    orin = PLATFORM_PROFILES[Platform.ORIN_AGX]
    nano = PLATFORM_PROFILES[Platform.NANO]
    assert orin.cpu_freqs_hz[-1] > nano.cpu_freqs_hz[-1]
    assert orin.gpu_freqs_hz[-1] > nano.gpu_freqs_hz[-1]


def test_cpu_setspeed_path_template_format():
    for plat in (Platform.ORIN_AGX, Platform.NANO):
        prof = PLATFORM_PROFILES[plat]
        assert prof.cpu_setspeed_path_template.startswith("/sys/")
        assert "{cpu}" in prof.cpu_setspeed_path_template


def test_gpu_setspeed_path_starts_with_sys():
    for plat in (Platform.ORIN_AGX, Platform.NANO):
        prof = PLATFORM_PROFILES[plat]
        assert prof.gpu_setspeed_path.startswith("/sys/")


def test_nano_gpu_path_contains_nano_devfreq_address():
    nano = PLATFORM_PROFILES[Platform.NANO]
    assert "57000000.gpu" in nano.gpu_setspeed_path


def test_orin_gpu_path_contains_orin_devfreq_address():
    orin = PLATFORM_PROFILES[Platform.ORIN_AGX]
    assert "17000000.gpu" in orin.gpu_setspeed_path


def test_tegrastats_field_layout_strings():
    assert PLATFORM_PROFILES[Platform.ORIN_AGX].tegrastats_field_layout == "orin"
    assert PLATFORM_PROFILES[Platform.NANO].tegrastats_field_layout == "nano"


def test_default_ema_alpha_in_range_and_nano_higher():
    orin = PLATFORM_PROFILES[Platform.ORIN_AGX]
    nano = PLATFORM_PROFILES[Platform.NANO]
    for alpha in (orin.default_ema_alpha, nano.default_ema_alpha):
        assert 0.0 < alpha <= 1.0
    # Nano samples are noisier per spec, so EMA weights the new sample more.
    assert nano.default_ema_alpha > orin.default_ema_alpha


def test_mem_total_mb_values():
    nano = PLATFORM_PROFILES[Platform.NANO]
    orin = PLATFORM_PROFILES[Platform.ORIN_AGX]
    assert nano.mem_total_mb == 4096
    assert orin.mem_total_mb >= 32 * 1024


def test_get_profile_unknown_platform_raises():
    with pytest.raises((KeyError, ValueError)):
        get_profile("xavier_nx")
