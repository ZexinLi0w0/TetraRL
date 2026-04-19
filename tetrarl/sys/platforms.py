"""Platform registry for Jetson DVFS / tegrastats.

Each PlatformProfile bundles the per-platform constants needed by
dvfs.py (frequency tables, sysfs node templates) and tegra_daemon.py
(tegrastats field layout, EMA defaults). Adding a new platform means
adding one entry to PLATFORM_PROFILES — callers stay unchanged.

Frequency table sources:
- Orin AGX: NVIDIA Jetson Linux R36.x cpufreq + devfreq sysfs (12 CPU
  Cortex-A78AE points, 14 Ampere GPU points; representative subset
  of the 29/11 actually exposed on a measured unit).
- Jetson Nano (4 GB): NVIDIA L4T 32.7 docs — 15 CPU points
  (102000..1479000 kHz) + 12 GPU points (76.8..921.6 MHz).

Unit convention: cpu_freqs_hz stores cpufreq's kHz integers (matches
the value written to scaling_setspeed); gpu_freqs_hz stores raw Hz.
The legacy `_hz` suffix on the CPU field is preserved for spec-API
compatibility — do not rename without updating callers in dvfs.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Union


class Platform(str, Enum):
    ORIN_AGX = "orin_agx"
    NANO = "nano"


@dataclass(frozen=True)
class PlatformProfile:
    name: str
    cpu_freqs_hz: list[int]  # actually kHz integers; see module docstring
    gpu_freqs_hz: list[int]
    cpu_setspeed_path_template: str  # must contain "{cpu}" for the cpu index
    gpu_setspeed_path: str
    tegrastats_field_layout: Literal["orin", "nano"]
    default_ema_alpha: float
    mem_total_mb: int


# Orin AGX values: align with the existing STUB_CPU_FREQS_KHZ / STUB_GPU_FREQS_HZ
# in dvfs.py. Read those constants to copy values exactly.
_ORIN_AGX_PROFILE = PlatformProfile(
    name="Orin AGX",
    cpu_freqs_hz=[
        115200, 192000, 384000, 576000, 729600, 921600,
        1113600, 1305600, 1497600, 1689600, 1881600, 2073600, 2188800,
    ],
    gpu_freqs_hz=[
        114750000, 216750000, 318750000, 420750000, 510000000,
        624750000, 726750000, 828750000, 930750000, 1032750000,
        1122000000, 1224750000, 1300500000, 1377000000,
    ],
    cpu_setspeed_path_template="/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_setspeed",
    gpu_setspeed_path="/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/userspace/set_freq",
    tegrastats_field_layout="orin",
    default_ema_alpha=0.1,
    mem_total_mb=32 * 1024,
)


# Nano (4 GB) — NVIDIA L4T 32.7 docs.
_NANO_PROFILE = PlatformProfile(
    name="Jetson Nano (4 GB)",
    cpu_freqs_hz=[
        102000, 204000, 307200, 403200, 518400, 614400, 710400, 825600,
        921600, 1036800, 1132800, 1224000, 1326000, 1428000, 1479000,
    ],
    gpu_freqs_hz=[
        76800000, 153600000, 230400000, 307200000, 384000000, 460800000,
        537600000, 614400000, 691200000, 768000000, 844800000, 921600000,
    ],
    cpu_setspeed_path_template="/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_setspeed",
    gpu_setspeed_path="/sys/devices/57000000.gpu/devfreq/57000000.gpu/userspace/set_freq",
    tegrastats_field_layout="nano",
    default_ema_alpha=0.2,  # Nano samples noisier (older sensor cadence) -> larger alpha
    mem_total_mb=4096,
)


PLATFORM_PROFILES: dict[Platform, PlatformProfile] = {
    Platform.ORIN_AGX: _ORIN_AGX_PROFILE,
    Platform.NANO: _NANO_PROFILE,
}


def get_profile(platform: Union[Platform, str]) -> PlatformProfile:
    """Return the PlatformProfile for an enum member or its string value."""
    if isinstance(platform, Platform):
        return PLATFORM_PROFILES[platform]
    try:
        key = Platform(platform)
    except ValueError as e:
        raise KeyError(f"unknown platform: {platform!r}") from e
    return PLATFORM_PROFILES[key]
