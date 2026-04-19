"""DVFS frequency controller for NVIDIA Jetson Orin AGX (and similar).

Reads cpufreq + devfreq sysfs nodes, exposes set_freq(cpu_idx, gpu_idx),
profiles transition latency for paper figures. Mac dev: pass `stub=True`
(or rely on auto-detect when sysfs nodes are absent) to avoid touching
the filesystem.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ORIN_CPU_AVAILABLE = (
    "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies"
)
ORIN_CPU_SETSPEED = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed"
ORIN_CPU_CUR = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
ORIN_CPU_GOVERNOR = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"

ORIN_GPU_AVAILABLE = (
    "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/available_frequencies"
)
ORIN_GPU_MIN = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/min_freq"
ORIN_GPU_MAX = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/max_freq"
ORIN_GPU_CUR = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq"


# Representative Orin AGX 12-core (Cortex-A78AE) CPU + Ampere GPU table.
STUB_CPU_FREQS_KHZ: list[int] = [
    115200, 192000, 384000, 576000, 729600, 921600,
    1113600, 1305600, 1497600, 1689600, 1881600, 2073600, 2188800,
]
STUB_GPU_FREQS_HZ: list[int] = [
    114750000, 216750000, 318750000, 420750000, 510000000,
    624750000, 726750000, 828750000, 930750000, 1032750000,
    1122000000, 1224750000, 1300500000, 1377000000,
]


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
        platform: str = "orin_agx",
        stub: Optional[bool] = None,
        cpu_paths: Optional[dict] = None,
        gpu_paths: Optional[dict] = None,
    ):
        self.platform = platform
        self.cpu_paths = cpu_paths or {
            "available": ORIN_CPU_AVAILABLE,
            "setspeed": ORIN_CPU_SETSPEED,
            "cur": ORIN_CPU_CUR,
            "governor": ORIN_CPU_GOVERNOR,
        }
        self.gpu_paths = gpu_paths or {
            "available": ORIN_GPU_AVAILABLE,
            "min": ORIN_GPU_MIN,
            "max": ORIN_GPU_MAX,
            "cur": ORIN_GPU_CUR,
        }

        if stub is None:
            stub = not Path(self.cpu_paths["available"]).exists()
        self.stub = stub

        if self.stub:
            self._stub_cpu_idx = len(STUB_CPU_FREQS_KHZ) - 1
            self._stub_gpu_idx = len(STUB_GPU_FREQS_HZ) - 1

    def available_frequencies(self) -> dict[str, list[int]]:
        if self.stub:
            return {
                "cpu": list(STUB_CPU_FREQS_KHZ),
                "gpu": list(STUB_GPU_FREQS_HZ),
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
                cpu_freq_khz=STUB_CPU_FREQS_KHZ[self._stub_cpu_idx],
                gpu_freq_hz=STUB_GPU_FREQS_HZ[self._stub_gpu_idx],
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
