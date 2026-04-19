"""DVFS frequency controller for NVIDIA Jetson Orin AGX (and similar).

Reads cpufreq + devfreq sysfs nodes, exposes set_freq(cpu_idx, gpu_idx),
profiles transition latency for paper figures. Mac dev: pass `stub=True`
(or rely on auto-detect when sysfs nodes are absent) to avoid touching
the filesystem.

Per-platform constants (frequency tables, sysfs node paths) live in
``tetrarl.sys.platforms``. Pass ``platform=Platform.NANO`` (or the string
``"nano"``) to drive a Jetson Nano; the default remains Orin AGX.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from tetrarl.sys.platforms import (
    PLATFORM_PROFILES,
    Platform,
    PlatformProfile,
    get_profile,
)


# Backward-compat module-level aliases. These mirror the Orin AGX profile and
# are kept because pre-refactor tests (and external callers) import them.
STUB_CPU_FREQS_KHZ: list[int] = list(PLATFORM_PROFILES[Platform.ORIN_AGX].cpu_freqs_hz)
STUB_GPU_FREQS_HZ: list[int] = list(PLATFORM_PROFILES[Platform.ORIN_AGX].gpu_freqs_hz)


def _default_cpu_paths(profile: PlatformProfile) -> dict[str, str]:
    """Build the default cpufreq sysfs paths for a platform profile.

    Historical behavior targets cpu0 for setspeed (and cur/governor/available
    siblings under cpu0); Orin AGX and Nano both expose the standard cpufreq
    layout, so the available/cur/governor siblings are not per-platform.
    """
    return {
        "available": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
        "setspeed": profile.cpu_setspeed_path_template.format(cpu=0),
        "cur": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
        "governor": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
    }


def _default_gpu_paths(profile: PlatformProfile) -> dict[str, str]:
    """Derive devfreq sysfs siblings from the profile's set_freq path.

    The profile stores the userspace governor write target (``.../userspace/
    set_freq``); strip that suffix to find the devfreq directory and build
    available/min/max/cur as siblings.
    """
    set_freq = profile.gpu_setspeed_path
    suffix = "/userspace/set_freq"
    if not set_freq.endswith(suffix):
        raise ValueError(
            f"profile gpu_setspeed_path must end with {suffix!r}; got {set_freq!r}"
        )
    devfreq_dir = set_freq[: -len(suffix)]
    return {
        "available": f"{devfreq_dir}/available_frequencies",
        "min": f"{devfreq_dir}/min_freq",
        "max": f"{devfreq_dir}/max_freq",
        "cur": f"{devfreq_dir}/cur_freq",
    }


@dataclass
class DVFSConfig:
    cpu_freq_khz: int = 0
    gpu_freq_hz: int = 0


@dataclass
class TransitionLatency:
    domain: str
    from_freq: int
    to_freq: int
    latency_ms: float


class DVFSController:
    def __init__(
        self,
        platform: Union[Platform, str] = Platform.ORIN_AGX,
        stub: Optional[bool] = None,
        cpu_paths: Optional[dict] = None,
        gpu_paths: Optional[dict] = None,
    ):
        self.profile = get_profile(platform)
        # Preserve the original-string form for backward compatibility with
        # pre-refactor callers that read ctrl.platform; surface the canonical
        # enum value so debugging stays unambiguous.
        self.platform = (
            platform.value if isinstance(platform, Platform) else str(platform)
        )

        self.cpu_paths = cpu_paths or _default_cpu_paths(self.profile)
        self.gpu_paths = gpu_paths or _default_gpu_paths(self.profile)

        if stub is None:
            stub = not Path(self.cpu_paths["available"]).exists()
        self.stub = stub

        if self.stub:
            self._stub_cpu_idx = len(self.profile.cpu_freqs_hz) - 1
            self._stub_gpu_idx = len(self.profile.gpu_freqs_hz) - 1

    def available_frequencies(self) -> dict[str, list[int]]:
        if self.stub:
            return {
                "cpu": list(self.profile.cpu_freqs_hz),
                "gpu": list(self.profile.gpu_freqs_hz),
            }
        cpu = sorted(
            int(x) for x in Path(self.cpu_paths["available"]).read_text().split()
        )
        gpu = sorted(
            int(x) for x in Path(self.gpu_paths["available"]).read_text().split()
        )
        return {"cpu": cpu, "gpu": gpu}

    def set_freq(
        self,
        cpu_idx: Optional[int] = None,
        gpu_idx: Optional[int] = None,
    ) -> DVFSConfig:
        avail = self.available_frequencies()
        cpu_freqs = avail["cpu"]
        gpu_freqs = avail["gpu"]

        if cpu_idx is not None:
            if not 0 <= cpu_idx < len(cpu_freqs):
                raise IndexError(
                    f"cpu_idx {cpu_idx} out of range [0, {len(cpu_freqs)})"
                )
            if self.stub:
                self._stub_cpu_idx = cpu_idx
            else:
                Path(self.cpu_paths["setspeed"]).write_text(str(cpu_freqs[cpu_idx]))

        if gpu_idx is not None:
            if not 0 <= gpu_idx < len(gpu_freqs):
                raise IndexError(
                    f"gpu_idx {gpu_idx} out of range [0, {len(gpu_freqs)})"
                )
            if self.stub:
                self._stub_gpu_idx = gpu_idx
            else:
                Path(self.gpu_paths["min"]).write_text(str(gpu_freqs[gpu_idx]))
                Path(self.gpu_paths["max"]).write_text(str(gpu_freqs[gpu_idx]))

        return self.current_state()

    def current_state(self) -> DVFSConfig:
        if self.stub:
            return DVFSConfig(
                cpu_freq_khz=self.profile.cpu_freqs_hz[self._stub_cpu_idx],
                gpu_freq_hz=self.profile.gpu_freqs_hz[self._stub_gpu_idx],
            )
        cpu_freq = int(Path(self.cpu_paths["cur"]).read_text().strip())
        gpu_freq = int(Path(self.gpu_paths["cur"]).read_text().strip())
        return DVFSConfig(cpu_freq_khz=cpu_freq, gpu_freq_hz=gpu_freq)

    def profile_transition_latency(
        self, domain: str = "cpu", n_iters: int = 3
    ) -> list[TransitionLatency]:
        if domain not in {"cpu", "gpu"}:
            raise ValueError("domain must be 'cpu' or 'gpu'")
        if n_iters < 1:
            raise ValueError("n_iters must be >= 1")

        avail = self.available_frequencies()
        freqs = avail[domain]
        results: list[TransitionLatency] = []

        for i, fa in enumerate(freqs):
            for j, fb in enumerate(freqs):
                if i == j:
                    continue
                latencies: list[float] = []
                for _ in range(n_iters):
                    if domain == "cpu":
                        self.set_freq(cpu_idx=i)
                    else:
                        self.set_freq(gpu_idx=i)
                    t0 = time.perf_counter()
                    if domain == "cpu":
                        self.set_freq(cpu_idx=j)
                    else:
                        self.set_freq(gpu_idx=j)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)
                results.append(
                    TransitionLatency(
                        domain=domain,
                        from_freq=fa,
                        to_freq=fb,
                        latency_ms=sum(latencies) / len(latencies),
                    )
                )
        return results
