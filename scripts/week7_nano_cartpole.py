"""Week 7 Task 7: Jetson Nano CartPole physical override-OOM validation.

Drives ``TetraRLFramework`` on CartPole-v1 under simulated memory
pressure to validate that the platform-aware DVFS module
(``Platform.NANO``) and the ``OverrideLayer`` cooperate on real
hardware: the override layer must fire >= 1 time and OOM events must
stay at 0.

Mac dev: ``--no-real-dvfs --no-real-tegrastats`` (default if hardware
absent) so the smoke runs on Mac without sudo or sysfs writes; the
companion ``tests/test_nano_cartpole_smoke.py`` exercises the same
``run_nano_cartpole`` entry point with stub telemetry.

Nano hardware: drop both ``--no-*`` flags. The driver will start a
real ``TegrastatsDaemon`` and (best-effort) a real DVFS controller;
if sysfs writes fail because the Nano profile paths don't match the
target board (e.g. Orin-Nano vs legacy Nano-4G), the driver logs the
failure and falls back to stub DVFS so training still completes.

Example::

    python3 scripts/week7_nano_cartpole.py \\
        --platform nano \\
        --memory-pressure-mb 1500 \\
        --n-episodes 200 \\
        --with-override \\
        --out-dir runs/w7_nano_cartpole/
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Make the repository root importable so we can pull in tetrarl.* even
# when invoked as ``python scripts/week7_nano_cartpole.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tetrarl.core.framework import (  # noqa: E402
    ResourceManager,
    StaticPreferencePlane,
    TetraRLFramework,
)
from tetrarl.morl.native.override import (  # noqa: E402
    HardwareTelemetry,
    OverrideLayer,
    OverrideThresholds,
)
from tetrarl.sys.dvfs import DVFSController  # noqa: E402
from tetrarl.sys.platforms import Platform  # noqa: E402


@dataclass
class _MemReading:
    ram_used_mb: int
    ram_total_mb: int
    latency_ema_ms: float
    energy_remaining_j: float

    @property
    def memory_util(self) -> float:
        return self.ram_used_mb / max(1, self.ram_total_mb)


def _read_meminfo_linux() -> tuple[int, int]:
    """Return (used_mb, total_mb) from /proc/meminfo (Linux only)."""
    total_kb = avail_kb = 0
    with open("/proc/meminfo", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                avail_kb = int(line.split()[1])
            if total_kb and avail_kb:
                break
    used_kb = max(0, total_kb - avail_kb)
    return used_kb // 1024, total_kb // 1024


def _read_meminfo_mac() -> tuple[int, int]:
    """Return (used_mb, total_mb) from `vm_stat` (Mac fallback)."""
    out = subprocess.check_output(["vm_stat"], text=True)
    page_size = 4096
    free = active = wired = inactive = 0
    for line in out.splitlines():
        if line.startswith("Pages free:"):
            free = int(line.split(":")[1].strip().rstrip("."))
        elif line.startswith("Pages active:"):
            active = int(line.split(":")[1].strip().rstrip("."))
        elif line.startswith("Pages wired down:"):
            wired = int(line.split(":")[1].strip().rstrip("."))
        elif line.startswith("Pages inactive:"):
            inactive = int(line.split(":")[1].strip().rstrip("."))
    used_mb = ((active + wired) * page_size) // (1024 * 1024)
    total_mb = ((free + active + wired + inactive) * page_size) // (1024 * 1024)
    return used_mb, max(total_mb, 1)


class PsutilTelemetrySource:
    """Memory telemetry from /proc/meminfo (Linux) or vm_stat (Mac).

    Avoids the heavyweight ``TegrastatsDaemon`` for Mac CI and for
    Jetson hosts where tegrastats is unavailable. Tests use this
    source with ``memory_pressure_mb`` configured so the override
    layer's threshold reliably trips.
    """

    def __init__(self, initial_energy_j: float = 1000.0):
        self._latency = 0.0
        self._energy = float(initial_energy_j)
        self._is_linux = Path("/proc/meminfo").exists()

    def _read(self) -> _MemReading:
        if self._is_linux:
            used, total = _read_meminfo_linux()
        else:
            used, total = _read_meminfo_mac()
        return _MemReading(
            ram_used_mb=used,
            ram_total_mb=total,
            latency_ema_ms=self._latency,
            energy_remaining_j=self._energy,
        )

    def update(self, latency_ms: float, energy_remaining_j: float) -> None:
        self._latency = float(latency_ms)
        self._energy = float(energy_remaining_j)

    def latest(self) -> _MemReading:
        return self._read()


class TegraTelemetrySource:
    """Real-hardware telemetry: TegrastatsDaemon + injected latency/energy."""

    def __init__(self, platform: Any, initial_energy_j: float = 1000.0):
        from tetrarl.sys.tegra_daemon import TegrastatsDaemon

        self._latency = 0.0
        self._energy = float(initial_energy_j)
        self.daemon = TegrastatsDaemon(platform=platform, source="auto")
        self.daemon.start()

    def update(self, latency_ms: float, energy_remaining_j: float) -> None:
        self._latency = float(latency_ms)
        self._energy = float(energy_remaining_j)

    def latest(self) -> _MemReading:
        r = self.daemon.latest()
        ram_used = getattr(r, "ram_used_mb", 0) if r is not None else 0
        ram_total = getattr(r, "ram_total_mb", 1) if r is not None else 1
        return _MemReading(
            ram_used_mb=int(ram_used),
            ram_total_mb=int(max(1, ram_total)),
            latency_ema_ms=self._latency,
            energy_remaining_j=self._energy,
        )

    def stop(self) -> None:
        self.daemon.stop()


def _telemetry_to_hw(reading: _MemReading) -> HardwareTelemetry:
    return HardwareTelemetry(
        latency_ema_ms=reading.latency_ema_ms,
        energy_remaining_j=reading.energy_remaining_j,
        memory_util=reading.memory_util,
    )


class _RandomArbiter:
    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = int(n_actions)
        self._rng = random.Random(seed)

    def act(self, state: Any, omega: np.ndarray) -> int:
        return self._rng.randint(0, self.n_actions - 1)


def _build_dvfs(platform: str, use_real_dvfs: bool) -> tuple[DVFSController, Optional[str]]:
    """Build a DVFSController; on real-DVFS sysfs failure fall back to stub.

    Returns (controller, deferred_reason_or_None).
    """
    deferred: Optional[str] = None
    if not use_real_dvfs:
        return DVFSController(platform=platform, stub=True), None
    try:
        ctrl = DVFSController(platform=platform, stub=False)
        # Sanity check: read available frequencies. This will raise
        # FileNotFoundError if the sysfs paths don't exist on the host.
        ctrl.available_frequencies()
        return ctrl, None
    except (FileNotFoundError, PermissionError, OSError) as exc:
        deferred = (
            f"real-DVFS unavailable ({type(exc).__name__}: {exc!s}); "
            "falling back to stub mode. The platform sysfs paths in "
            "tetrarl/sys/platforms.py likely need adjustment for this "
            "board variant. Documented as deferred follow-up."
        )
        return DVFSController(platform=platform, stub=True), deferred


def make_nano_framework(
    n_actions: int,
    seed: int,
    platform: str,
    max_memory_util: float,
    with_override: bool,
    use_real_dvfs: bool,
    use_real_tegrastats: bool,
) -> tuple[TetraRLFramework, Any, OverrideLayer, Optional[str]]:
    pref = StaticPreferencePlane(np.array([0.5, 0.5], dtype=np.float32))
    rm = ResourceManager()

    thresholds = OverrideThresholds(
        max_latency_ms=10000.0 if with_override else None,
        min_energy_j=0.5 if with_override else None,
        max_memory_util=float(max_memory_util) if with_override else None,
    )
    override = OverrideLayer(
        thresholds=thresholds,
        fallback_action=0,
        cooldown_steps=0,
    )

    if use_real_tegrastats:
        telemetry: Any = TegraTelemetrySource(platform=platform)
    else:
        telemetry = PsutilTelemetrySource()

    dvfs, deferred = _build_dvfs(platform=platform, use_real_dvfs=use_real_dvfs)

    fw = TetraRLFramework(
        preference_plane=pref,
        rl_arbiter=_RandomArbiter(n_actions=n_actions, seed=seed),
        resource_manager=rm,
        override_layer=override,
        telemetry_source=telemetry,
        telemetry_adapter=_telemetry_to_hw,
        dvfs_controller=dvfs,
    )
    return fw, telemetry, override, deferred


def run_nano_cartpole(
    n_episodes: int = 200,
    memory_pressure_mb: int = 0,
    with_override: bool = True,
    use_real_dvfs: bool = False,
    use_real_tegrastats: bool = False,
    out_dir: Optional[str] = None,
    seed: int = 0,
    max_memory_util: float = 0.95,
    platform: str = "nano",
) -> dict:
    """Run a Nano CartPole episode loop and return a summary dict.

    Allocates ``memory_pressure_mb`` of dummy bytes alongside training to
    push memory_util toward the override threshold. OOM events are
    counted (caught from MemoryError) so we can verify the override
    layer prevents catastrophic failure.
    """
    import gymnasium as gym  # lazy import keeps test collection cheap

    env = gym.make("CartPole-v1")
    n_actions = int(env.action_space.n)
    fw, telemetry, override, deferred = make_nano_framework(
        n_actions=n_actions,
        seed=seed,
        platform=platform,
        max_memory_util=max_memory_util,
        with_override=with_override,
        use_real_dvfs=use_real_dvfs,
        use_real_tegrastats=use_real_tegrastats,
    )

    pressure_blocks: list[bytearray] = []
    oom_events = 0
    if memory_pressure_mb > 0:
        try:
            pressure_blocks.append(bytearray(int(memory_pressure_mb) * 1024 * 1024))
        except (MemoryError, OverflowError):
            oom_events += 1

    out_file = None
    jsonl_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    if out_dir is not None:
        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_root / "trace.jsonl"
        summary_path = out_root / "summary.json"
        out_file = open(jsonl_path, "w", encoding="utf-8")

    energy_remaining = 1000.0
    framework_step_times_ms: list[float] = []
    total_steps = 0
    episode_returns: list[float] = []

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            episode_step = 0
            while not done:
                t_fw0 = time.perf_counter()
                try:
                    record = fw.step(obs)
                except MemoryError:
                    oom_events += 1
                    done = True
                    break
                t_fw1 = time.perf_counter()
                fw_dt_ms = (t_fw1 - t_fw0) * 1000.0
                framework_step_times_ms.append(fw_dt_ms)
                action = int(record["action"])

                t_env0 = time.perf_counter()
                obs, reward, terminated, truncated, _ = env.step(action)
                t_env1 = time.perf_counter()
                env_dt_ms = (t_env1 - t_env0) * 1000.0

                latency_ms = env_dt_ms + fw_dt_ms
                energy_j = 1e-3 * (action + 1)
                energy_remaining = max(0.0, energy_remaining - energy_j)

                record["latency_ms"] = float(latency_ms)
                record["energy_j"] = float(energy_j)
                fw.observe_reward(float(reward))
                telemetry.update(
                    latency_ms=latency_ms,
                    energy_remaining_j=energy_remaining,
                )

                if out_file is not None:
                    serialisable = {
                        "episode": ep,
                        "step": episode_step,
                        "action": action,
                        "override_fired": bool(record["override_fired"]),
                        "reward": float(record["reward"]),
                        "latency_ms": float(record["latency_ms"]),
                        "memory_util": float(record["memory_util"] or 0.0),
                        "framework_step_ms": fw_dt_ms,
                    }
                    out_file.write(json.dumps(serialisable) + "\n")

                ep_return += float(reward)
                episode_step += 1
                total_steps += 1
                done = bool(terminated or truncated)
            episode_returns.append(ep_return)
    finally:
        if out_file is not None:
            out_file.close()
        env.close()
        if hasattr(telemetry, "stop"):
            try:
                telemetry.stop()
            except Exception:
                pass
        del pressure_blocks
        gc.collect()

    summary = {
        "platform": platform,
        "n_episodes": n_episodes,
        "memory_pressure_mb": int(memory_pressure_mb),
        "with_override": bool(with_override),
        "use_real_dvfs": bool(use_real_dvfs),
        "use_real_tegrastats": bool(use_real_tegrastats),
        "max_memory_util_threshold": float(max_memory_util),
        "total_steps": int(total_steps),
        "override_fire_count": int(override.fire_count),
        "oom_events": int(oom_events),
        "mean_episode_return": (
            float(sum(episode_returns) / len(episode_returns))
            if episode_returns
            else 0.0
        ),
        "mean_framework_step_ms": (
            float(sum(framework_step_times_ms) / len(framework_step_times_ms))
            if framework_step_times_ms
            else 0.0
        ),
        "max_framework_step_ms": (
            float(max(framework_step_times_ms))
            if framework_step_times_ms
            else 0.0
        ),
        "deferred_dvfs_reason": deferred,
        "trace_path": str(jsonl_path) if jsonl_path else None,
    }

    if summary_path is not None:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_path"] = str(summary_path)

    return summary


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--platform",
        default="nano",
        choices=["nano", "orin_agx"],
    )
    p.add_argument("--memory-pressure-mb", type=int, default=0)
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument(
        "--with-override",
        dest="with_override",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-override",
        dest="with_override",
        action="store_false",
    )
    p.add_argument(
        "--no-real-dvfs",
        action="store_true",
        help="Force DVFS stub mode (don't write sysfs).",
    )
    p.add_argument(
        "--no-real-tegrastats",
        action="store_true",
        help="Don't spawn the real tegrastats binary; "
        "use /proc/meminfo (Linux) or vm_stat (Mac).",
    )
    p.add_argument(
        "--max-memory-util",
        type=float,
        default=0.85,
        help="OverrideLayer max_memory_util threshold; "
        "set lower to make the override fire under pressure.",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    summary = run_nano_cartpole(
        n_episodes=args.n_episodes,
        memory_pressure_mb=args.memory_pressure_mb,
        with_override=args.with_override,
        use_real_dvfs=not args.no_real_dvfs,
        use_real_tegrastats=not args.no_real_tegrastats,
        out_dir=args.out_dir,
        seed=args.seed,
        max_memory_util=args.max_memory_util,
        platform=args.platform,
    )

    print(json.dumps(summary, indent=2))
    print()
    print(f"override_fire_count = {summary['override_fire_count']}")
    print(f"oom_events          = {summary['oom_events']}")
    if summary.get("deferred_dvfs_reason"):
        print(f"deferred            = {summary['deferred_dvfs_reason']}")

    # Acceptance: spec says fire_count >= 1 AND oom_events == 0
    ok = summary["override_fire_count"] >= 1 and summary["oom_events"] == 0
    if ok:
        print("ACCEPTANCE: PASS")
        return 0
    print("ACCEPTANCE: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
