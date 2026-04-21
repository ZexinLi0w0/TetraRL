"""Async tegrastats sensor daemon.

Samples at sample_hz (default 100), dispatches EMA-filtered readings to a
callback at dispatch_hz (default 10). Mac-friendly: pass `source="file:<path>"`
to read from a captured tegrastats fixture instead of the binary.

Per-platform layout knobs (tegrastats power-rail field names, EMA defaults)
live in ``tetrarl.sys.platforms``. Pass ``platform=Platform.NANO`` (or the
string ``"nano"``) to drive a Jetson Nano; the default remains Orin AGX.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional, Union

from tetrarl.sys.platforms import Platform, get_profile

_RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
_CPU_RE = re.compile(r"CPU\s+\[([^\]]+)\]")
_GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%(?:@\[?(\d+)\]?)?")
_GPU_TEMP_RE = re.compile(r"GPU@([\d.]+)C")
_VDD_GPU_RE = re.compile(r"VDD_GPU_SOC\s+(\d+)mW")
_VDD_CPU_RE = re.compile(r"VDD_CPU_CV\s+(\d+)mW")
_POM_GPU_RE = re.compile(r"POM_5V_GPU\s+(\d+)/\d+")
_POM_CPU_RE = re.compile(r"POM_5V_CPU\s+(\d+)/\d+")
_EMC_RE = re.compile(r"EMC_FREQ\s+(\d+)%@(\d+)")


@dataclass
class TegrastatsReading:
    ts_monotonic: float = 0.0
    ram_used_mb: int = 0
    ram_total_mb: int = 0
    cpu_util_per_core: list[float] = field(default_factory=list)
    cpu_freq_mhz: int = 0
    gr3d_freq_pct: float = 0.0
    gpu_freq_mhz: int = 0
    emc_freq_pct: float = 0.0
    emc_freq_mhz: int = 0
    gpu_temp_c: float = 0.0
    vdd_gpu_soc_mw: int = 0
    vdd_cpu_cv_mw: int = 0


def parse_tegrastats_line(
    line: str, layout: Literal["orin", "nano"] = "orin"
) -> Optional[TegrastatsReading]:
    """Parse a single tegrastats line.

    The ``layout`` argument selects the power-rail regex set:
    - "orin": ``VDD_GPU_SOC`` / ``VDD_CPU_CV`` (Jetson Orin AGX/NX/Nano-Orin).
    - "nano": ``POM_5V_GPU`` / ``POM_5V_CPU`` (legacy Jetson Nano L4T 32.x).
    Both layouts populate the same ``vdd_gpu_soc_mw`` / ``vdd_cpu_cv_mw``
    fields so downstream consumers see uniform names regardless of platform.
    """
    if not line or not line.strip():
        return None

    r = TegrastatsReading(ts_monotonic=time.monotonic())
    matched = False

    m = _RAM_RE.search(line)
    if m:
        r.ram_used_mb = int(m.group(1))
        r.ram_total_mb = int(m.group(2))
        matched = True

    m = _CPU_RE.search(line)
    if m:
        utils: list[float] = []
        freqs: list[int] = []
        for chunk in m.group(1).split(","):
            try:
                util_str, freq_str = chunk.split("@")
                utils.append(float(util_str.replace("%", "")))
                freqs.append(int(freq_str))
            except (ValueError, IndexError):
                continue
        if utils:
            r.cpu_util_per_core = utils
            r.cpu_freq_mhz = freqs[0] if freqs else 0
            matched = True

    m = _GR3D_RE.search(line)
    if m:
        r.gr3d_freq_pct = float(m.group(1))
        r.gpu_freq_mhz = int(m.group(2)) if m.group(2) is not None else 0
        matched = True

    m = _EMC_RE.search(line)
    if m:
        r.emc_freq_pct = float(m.group(1))
        r.emc_freq_mhz = int(m.group(2))

    m = _GPU_TEMP_RE.search(line)
    if m:
        r.gpu_temp_c = float(m.group(1))

    if layout == "nano":
        m = _POM_GPU_RE.search(line)
        if m:
            r.vdd_gpu_soc_mw = int(m.group(1))

        m = _POM_CPU_RE.search(line)
        if m:
            r.vdd_cpu_cv_mw = int(m.group(1))
    else:
        m = _VDD_GPU_RE.search(line)
        if m:
            r.vdd_gpu_soc_mw = int(m.group(1))

        m = _VDD_CPU_RE.search(line)
        if m:
            r.vdd_cpu_cv_mw = int(m.group(1))

    return r if matched else None


def _ema_blend(
    prev: TegrastatsReading, new: TegrastatsReading, alpha: float
) -> TegrastatsReading:
    def b(p: float, n: float) -> float:
        return alpha * n + (1.0 - alpha) * p

    n_cores = max(len(prev.cpu_util_per_core), len(new.cpu_util_per_core))
    p_util = list(prev.cpu_util_per_core) + [0.0] * (
        n_cores - len(prev.cpu_util_per_core)
    )
    n_util = list(new.cpu_util_per_core) + [0.0] * (
        n_cores - len(new.cpu_util_per_core)
    )
    blended = [b(p, n) for p, n in zip(p_util, n_util)]

    return TegrastatsReading(
        ts_monotonic=new.ts_monotonic,
        ram_used_mb=int(b(prev.ram_used_mb, new.ram_used_mb)),
        ram_total_mb=new.ram_total_mb or prev.ram_total_mb,
        cpu_util_per_core=blended,
        cpu_freq_mhz=int(b(prev.cpu_freq_mhz, new.cpu_freq_mhz)),
        gr3d_freq_pct=b(prev.gr3d_freq_pct, new.gr3d_freq_pct),
        gpu_freq_mhz=int(b(prev.gpu_freq_mhz, new.gpu_freq_mhz)),
        emc_freq_pct=b(prev.emc_freq_pct, new.emc_freq_pct),
        emc_freq_mhz=int(b(prev.emc_freq_mhz, new.emc_freq_mhz)),
        gpu_temp_c=b(prev.gpu_temp_c, new.gpu_temp_c),
        vdd_gpu_soc_mw=int(b(prev.vdd_gpu_soc_mw, new.vdd_gpu_soc_mw)),
        vdd_cpu_cv_mw=int(b(prev.vdd_cpu_cv_mw, new.vdd_cpu_cv_mw)),
    )


class TegrastatsDaemon:
    """Async tegrastats sensor daemon with EMA filtering.

    source: "auto" picks "binary" if `tegrastats` is on PATH else "noop";
            "binary" runs the tegrastats CLI;
            "file:<path>" cycles through lines of a captured fixture.
    """

    def __init__(
        self,
        sample_hz: float = 100.0,
        dispatch_hz: float = 10.0,
        ema_alpha: Optional[float] = None,
        source: str = "auto",
        on_dispatch: Optional[Callable[[TegrastatsReading], None]] = None,
        tegrastats_binary: str = "tegrastats",
        platform: Union[Platform, str] = Platform.ORIN_AGX,
    ):
        self.profile = get_profile(platform)
        self.platform_name = self.profile.name

        if ema_alpha is None:
            ema_alpha = self.profile.default_ema_alpha
        else:
            if not 0.0 < ema_alpha <= 1.0:
                raise ValueError("ema_alpha must be in (0, 1]")
        if dispatch_hz > sample_hz:
            raise ValueError("dispatch_hz must be <= sample_hz")

        self.sample_hz = sample_hz
        self.dispatch_hz = dispatch_hz
        self.ema_alpha = ema_alpha
        self.source = source
        self.on_dispatch = on_dispatch
        self.tegrastats_binary = tegrastats_binary

        self._sample_period = 1.0 / sample_hz
        self._dispatch_every = max(1, round(sample_hz / dispatch_hz))

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[TegrastatsReading] = None
        self._proc: Optional[subprocess.Popen] = None
        self._fixture_lines: Optional[list[str]] = None

    def start(self) -> None:
        if self._running:
            return
        resolved = self._resolve_source()
        if resolved.startswith("file:"):
            path = resolved[5:]
            self._fixture_lines = Path(path).read_text().strip().splitlines()
        elif resolved == "binary":
            interval_ms = max(1, int(self._sample_period * 1000))
            self._proc = subprocess.Popen(
                [self.tegrastats_binary, "--interval", str(interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        # "noop" leaves both fixture_lines and _proc as None.
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        self._fixture_lines = None

    def latest(self) -> Optional[TegrastatsReading]:
        with self._lock:
            return self._latest

    def _resolve_source(self) -> str:
        if self.source != "auto":
            return self.source
        if shutil.which(self.tegrastats_binary):
            return "binary"
        return "noop"

    def _read_one_line(self, idx: int) -> Optional[str]:
        if self._fixture_lines:
            return self._fixture_lines[idx % len(self._fixture_lines)]
        if self._proc is not None and self._proc.stdout is not None:
            line = self._proc.stdout.readline()
            return line.strip() if line else None
        return None

    def _loop(self) -> None:
        tick = 0
        idx = 0
        next_t = time.monotonic()
        layout = self.profile.tegrastats_field_layout
        while self._running:
            line = self._read_one_line(idx)
            idx += 1
            if line:
                reading = parse_tegrastats_line(line, layout=layout)
                if reading is not None:
                    with self._lock:
                        if self._latest is None:
                            self._latest = reading
                        else:
                            self._latest = _ema_blend(
                                self._latest, reading, self.ema_alpha
                            )
                        snapshot = self._latest
                    if (
                        tick % self._dispatch_every == 0
                        and self.on_dispatch is not None
                    ):
                        try:
                            self.on_dispatch(snapshot)
                        except Exception:
                            pass
                    tick += 1

            next_t += self._sample_period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()
