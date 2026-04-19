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

The framework treats each component as a black box; we do not retrain
or re-derive anything here. Records (one dict per step) are appended to
self.history and exposed for offline analysis.
"""
from __future__ import annotations

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
    ):
        self.preference_plane = preference_plane
        self.rl_arbiter = rl_arbiter
        self.resource_manager = resource_manager
        self.override_layer = override_layer
        self.telemetry_source = telemetry_source
        self.telemetry_adapter = telemetry_adapter
        self.dvfs_controller = dvfs_controller
        self.history: list[dict] = []

    def step(self, state) -> dict:
        omega = self.preference_plane.get()
        hw = self.telemetry_adapter(self.telemetry_source.latest())
        proposed = self.rl_arbiter.act(state, omega)
        override_fired, fallback = self.override_layer.step(hw)
        action = fallback if override_fired else proposed

        dvfs_idx: Optional[int] = None
        if self.dvfs_controller is not None:
            n_levels = len(self.dvfs_controller.available_frequencies()["gpu"])
            dvfs_idx = self.resource_manager.decide_dvfs(hw, n_levels=n_levels)
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
        }
        self.history.append(record)
        return record

    def observe_reward(self, reward: float) -> None:
        if not self.history:
            raise RuntimeError("observe_reward called before step()")
        self.history[-1]["reward"] = float(reward)

    def reset(self) -> None:
        self.history = []
        self.override_layer.reset()
