"""Asynchronous tegrastats sensor daemon for NVIDIA Jetson platforms.

Reads tegrastats at 100 Hz and dispatches smoothed readings (EMA-filtered,
alpha=0.1) to the RL agent at 10 Hz.  Follows the kernel/user split
pattern from DVFS-DRL-Multitask (2024).
"""

# TODO: Week 5 — implement async sensor daemon with 100 Hz sampling,
#       10 Hz dispatch, and EMA filtering; validate 1-hour stability
#       on Orin AGX with < 1 MB memory drift.

import threading
from dataclasses import dataclass


@dataclass
class TegrastatsReading:
    """A single tegrastats sensor snapshot."""

    gpu_freq_mhz: int = 0
    cpu_freq_mhz: int = 0
    emc_freq_mhz: int = 0
    power_mw: int = 0
    gpu_util_pct: float = 0.0
    mem_used_mb: int = 0
    mem_total_mb: int = 0
    temp_gpu_c: float = 0.0
    temp_cpu_c: float = 0.0


class TegrastatsDaemon:
    """Async daemon that continuously reads tegrastats on Jetson hardware."""

    def __init__(self, sample_hz: float = 100.0, dispatch_hz: float = 10.0,
                 ema_alpha: float = 0.1):
        self.sample_hz = sample_hz
        self.dispatch_hz = dispatch_hz
        self.ema_alpha = ema_alpha
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background sampling thread."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop the background sampling thread."""
        raise NotImplementedError

    def latest(self) -> TegrastatsReading:
        """Return the most recent EMA-filtered reading."""
        raise NotImplementedError
