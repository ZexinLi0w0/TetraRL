"""TetraRL 4-component framework orchestrator.

Wires the four runtime components into a single per-step pipeline:
  (i)   Preference Plane  -> emits omega weights for the current step
  (ii)  RL Arbiter        -> proposes an action conditioned on (state, omega)
  (iii) Resource Manager  -> picks a DVFS target index from telemetry
  (iv)  Hardware Override -> may veto the arbiter's action under hard limits

Per-step dataflow inside step():
    omega = preference_plane.get()
    hw    = telemetry_adapter(telemetry_source.latest())
    proposed = rl_arbiter.act(state, omega)
    fired, fallback = override_layer.step(hw)
    action = fallback if fired else proposed
    dvfs_idx = resource_manager.decide_dvfs(hw, n_levels) if dvfs_controller else None

When ``concurrent_decision`` is provided (Week 7 / DVFO Zhang TMC 2023),
the DVFS decision computation is moved to a background thread and the
per-step body becomes:

    apply_latest()  # use the decision computed during step t-1
    submit(hw)      # kick off the decision for step t+1 in background
    arbiter.act()   # foreground forward pass overlaps with the worker

The concurrent path takes precedence over the in-loop ResourceManager +
DVFSController calls; passing both is fine — the in-loop path is simply
skipped. When ``concurrent_decision`` is ``None`` (default), behaviour
is identical to the pre-Week-7 sequential pipeline (BACKWARD COMPATIBLE).

The framework treats each component as a black box; we do not retrain
or re-derive anything here. Records (one dict per step) are appended to
self.history and exposed for offline analysis.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from tetrarl.morl.native.override import HardwareTelemetry, OverrideLayer


class StaticPreferencePlane:
    """Constant preference vector. Replaceable with a learned scheduler later."""

    def __init__(self, omega: np.ndarray):
        self._omega = np.asarray(omega, dtype=np.float32).copy()

    def get(self) -> np.ndarray:
        return self._omega.copy()


@dataclass
class ResourceManagerConfig:
    soft_latency_ms: float = 50.0
    min_energy_j: float = 50.0
    max_memory_util: float = 0.7


class ResourceManager:
    """Maps telemetry to a DVFS target index via a simple step-down rule."""

    def __init__(self, config: Optional[ResourceManagerConfig] = None):
        self.config = config or ResourceManagerConfig()

    def decide_dvfs(self, telemetry: HardwareTelemetry, n_levels: int) -> int:
        if n_levels <= 0:
            raise ValueError("n_levels must be positive")
        idx = n_levels - 1
        cfg = self.config
        if telemetry.latency_ema_ms is not None and telemetry.latency_ema_ms > cfg.soft_latency_ms:
            idx -= 1
        if telemetry.energy_remaining_j is not None and telemetry.energy_remaining_j < cfg.min_energy_j:
            idx -= 1
        if telemetry.memory_util is not None and telemetry.memory_util > cfg.max_memory_util:
            idx -= 1
        return max(0, min(n_levels - 1, idx))


class TetraRLFramework:
    """Per-step orchestrator over the 4 TetraRL components."""

    def __init__(
        self,
        preference_plane: StaticPreferencePlane,
        rl_arbiter: Any,
        resource_manager: ResourceManager,
        override_layer: OverrideLayer,
        telemetry_source: Any,
        telemetry_adapter: Callable[[Any], HardwareTelemetry],
        dvfs_controller: Any = None,
        concurrent_decision: Optional[Any] = None,
        profiler: Optional[Any] = None,
    ):
        self.preference_plane = preference_plane
        self.rl_arbiter = rl_arbiter
        self.resource_manager = resource_manager
        self.override_layer = override_layer
        self.telemetry_source = telemetry_source
        self.telemetry_adapter = telemetry_adapter
        self.dvfs_controller = dvfs_controller
        self.concurrent_decision = concurrent_decision
        self.profiler = profiler
        self.history: list[dict] = []

    def _maybe_time(self, name: str):
        """Wrap a component call in profiler.time(name) when a profiler is set.

        Returns a no-op context manager when self.profiler is None so the
        per-step body stays a single readable block instead of a tower of
        if/else gates.
        """
        if self.profiler is None:
            return contextlib.nullcontext()
        return self.profiler.time(name)

    def step(self, state) -> dict:
        with self._maybe_time("preference_plane_get"):
            omega = self.preference_plane.get()
        with self._maybe_time("tegra_daemon_sample"):
            hw = self.telemetry_adapter(self.telemetry_source.latest())

        dvfs_idx: Optional[int] = None
        concurrent_dvfs_used = False
        if self.concurrent_decision is not None:
            # Apply the decision computed during the PREVIOUS step's
            # background work (lag-by-1 per DVFO; on step 0 returns None
            # or the configured fallback without crashing).
            dvfs_idx = self.concurrent_decision.apply_latest()
            # Kick off the NEXT decision; the worker computes it while the
            # arbiter forward pass below runs on the GIL-bound foreground.
            self.concurrent_decision.submit(hw)
            concurrent_dvfs_used = True

        with self._maybe_time("rl_arbiter_act"):
            proposed = self.rl_arbiter.act(state, omega)
        with self._maybe_time("override_layer_step"):
            override_fired, fallback = self.override_layer.step(hw)
        action = fallback if override_fired else proposed

        if self.concurrent_decision is None and self.dvfs_controller is not None:
            n_levels = len(self.dvfs_controller.available_frequencies()["gpu"])
            with self._maybe_time("resource_manager_decide"):
                dvfs_idx = self.resource_manager.decide_dvfs(hw, n_levels=n_levels)
            with self._maybe_time("dvfs_controller_set"):
                self.dvfs_controller.set_freq(gpu_idx=dvfs_idx)

        record: dict = {
            "action": action,
            "proposed_action": proposed,
            "omega": omega,
            "override_fired": bool(override_fired),
            "reward": None,
            "latency_ms": hw.latency_ema_ms,
            "energy_j": hw.energy_remaining_j,
            "memory_util": hw.memory_util,
            "dvfs_idx": dvfs_idx,
            "concurrent_dvfs_used": concurrent_dvfs_used,
        }
        self.history.append(record)
        if self.profiler is not None:
            self.profiler.step_marker()
        return record

    def observe_reward(self, reward: float) -> None:
        if not self.history:
            raise RuntimeError("observe_reward called before step()")
        self.history[-1]["reward"] = float(reward)

    def reset(self) -> None:
        self.history = []
        self.override_layer.reset()
