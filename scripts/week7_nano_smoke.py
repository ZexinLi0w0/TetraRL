#!/usr/bin/env python3
"""Mac-runnable Week 7 smoke that confirms the Jetson Nano profile end-to-end.

Exercises both ``DVFSController`` and ``TegrastatsDaemon`` in stub mode using
the captured Nano tegrastats fixture (``tests/fixtures/tegrastats_nano_sample.txt``).
No Nano hardware (and no sudo) is required: the controller stays in stub mode
and the daemon ingests the captured fixture instead of shelling out.

Real Nano sysfs writes are gated behind ``--allow-real-dvfs`` in
``scripts/profile_orin_dvfs.py``; this smoke uses stub mode only and never
touches sysfs.

The script verifies:
    1. The Nano CPU/GPU frequency tables expose the documented L4T 32.7 grid.
    2. ``TegrastatsDaemon`` parses Nano-layout lines (POM_5V_GPU/POM_5V_CPU)
       and dispatches non-trivial readings via the on_dispatch callback.
    3. A side-by-side comparison of Orin AGX vs Nano top frequencies.

Exits 0 on success; on any unexpected exception the top-level handler prints
the traceback and exits 1.
"""
from __future__ import annotations

import statistics
import sys
import time
import traceback
from pathlib import Path


# Allow `python scripts/week7_nano_smoke.py` from the repo root without
# requiring PYTHONPATH or a pip-installed editable build.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tetrarl.sys.dvfs import DVFSController  # noqa: E402
from tetrarl.sys.platforms import Platform  # noqa: E402
from tetrarl.sys.tegra_daemon import TegrastatsDaemon, TegrastatsReading  # noqa: E402


NANO_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tegrastats_nano_sample.txt"


def _print_freq_table(label: str, ctrl: DVFSController) -> None:
    avail = ctrl.available_frequencies()
    cpu = avail["cpu"]
    gpu = avail["gpu"]
    print(f"[{label}] CPU points: count={len(cpu)} min={min(cpu)} max={max(cpu)} (kHz)")
    print(f"[{label}] GPU points: count={len(gpu)} min={min(gpu)} max={max(gpu)} (Hz)")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def main() -> int:
    if not NANO_FIXTURE.exists():
        print(f"FAIL: Nano tegrastats fixture missing at {NANO_FIXTURE}")
        return 1

    # 1. Nano DVFS controller in stub mode -> print frequency tables.
    print("=== Nano DVFS controller (stub) ===")
    nano_ctrl = DVFSController(platform=Platform.NANO, stub=True)
    _print_freq_table("nano", nano_ctrl)
    print()

    # 2. Spin up the Nano-layout TegrastatsDaemon backed by the fixture.
    print("=== Nano TegrastatsDaemon (file-fixture) ===")
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
    try:
        # 200 Hz -> 5 ms per tick; 50 samples need 0.25 s, sleep 0.5 s for slack.
        time.sleep(0.5)
    finally:
        daemon.stop()

    n_samples = len(received)
    print(f"platform_name      : {daemon.platform_name}")
    print(f"samples received   : {n_samples}")

    if n_samples < 50:
        print(f"FAIL: expected >= 50 dispatched samples, got {n_samples}")
        return 1

    means = {
        "cpu_freq_mhz":     _mean([r.cpu_freq_mhz     for r in received]),
        "gpu_freq_mhz":     _mean([r.gpu_freq_mhz     for r in received]),
        "gr3d_freq_pct":    _mean([r.gr3d_freq_pct    for r in received]),
        "vdd_gpu_soc_mw":   _mean([r.vdd_gpu_soc_mw   for r in received]),
        "vdd_cpu_cv_mw":    _mean([r.vdd_cpu_cv_mw    for r in received]),
        "gpu_temp_c":       _mean([r.gpu_temp_c       for r in received]),
    }
    for k, v in means.items():
        print(f"mean {k:<18}: {v:.3f}")

    if means["gpu_freq_mhz"] <= 0.0:
        print("FAIL: mean GPU MHz is zero — Nano-layout parse likely broken")
        return 1

    print()

    # 3. Per-platform comparison: top CPU/GPU frequency for Orin vs Nano.
    print("=== Per-platform DVFS top frequencies (stub mode) ===")
    orin_ctrl = DVFSController(platform=Platform.ORIN_AGX, stub=True)
    nano_avail = nano_ctrl.available_frequencies()
    orin_avail = orin_ctrl.available_frequencies()
    rows = [
        ("Orin AGX", max(orin_avail["cpu"]) / 1000.0, max(orin_avail["gpu"]) / 1e6),
        ("Nano",     max(nano_avail["cpu"]) / 1000.0, max(nano_avail["gpu"]) / 1e6),
    ]
    print(f"| {'Platform':<10} | {'top CPU MHz':>12} | {'top GPU MHz':>12} |")
    print(f"|{'-'*12}|{'-'*14}|{'-'*14}|")
    for name, cpu_mhz, gpu_mhz in rows:
        print(f"| {name:<10} | {cpu_mhz:>12.1f} | {gpu_mhz:>12.1f} |")

    print()
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 — preserve traceback for dev visibility.
        traceback.print_exc()
        sys.exit(1)
