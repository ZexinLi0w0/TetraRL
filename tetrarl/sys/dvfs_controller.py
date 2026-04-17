"""DVFS frequency controller for NVIDIA Jetson platforms.

Enumerates available CPU, GPU, and EMC frequency points on the target
Jetson board and provides an API for setting frequencies.  Supports
super-block decision granularity (SparseDVFS 2025): DVFS adjustments
occur only every N training steps to amortize transition overhead.
"""

# TODO: Week 5 — enumerate frequency tables for Orin AGX / Xavier NX /
#       Nano; measure per-pair transition latencies; implement
#       super-block granularity with configurable N.

from dataclasses import dataclass


@dataclass
class DVFSState:
    """Current DVFS frequency configuration."""

    cpu_freq_mhz: int = 0
    gpu_freq_mhz: int = 0
    emc_freq_mhz: int = 0


class DVFSController:
    """Controls CPU/GPU/EMC frequencies on Jetson platforms."""

    def __init__(self, platform: str = "orin_agx", super_block_n: int = 10):
        self.platform = platform
        self.super_block_n = super_block_n
        self._step_counter = 0

    def available_frequencies(self) -> dict[str, list[int]]:
        """Return available frequency points for cpu, gpu, and emc.

        @return  Dict mapping domain name to sorted list of frequencies (MHz).
        """
        raise NotImplementedError

    def set_freq(self, cpu_freq: int | None = None,
                 gpu_freq: int | None = None) -> DVFSState:
        """Set CPU and/or GPU frequency.

        @param cpu_freq  Target CPU frequency in MHz, or None to keep current.
        @param gpu_freq  Target GPU frequency in MHz, or None to keep current.
        @return          The new DVFSState after the transition.
        """
        raise NotImplementedError

    def current_state(self) -> DVFSState:
        """Read back the current DVFS configuration from sysfs."""
        raise NotImplementedError
