"""Hardware-emergency override layer.

Monitors a hardware telemetry snapshot (latency_ema_ms, energy_remaining_j,
memory_util) against configurable thresholds. When any threshold is
violated, returns a conservative fallback action and `override_active=True`.
Otherwise returns `(False, None)` and the executor uses the policy's action.

The override is decoupled from the policy gradient: train sees the policy's
proposed action and reward, while the executor swaps in `fallback_action`.
This keeps learning unaffected and lets us audit how often the override fires.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HardwareTelemetry:
    """Snapshot of runtime hardware state. All fields optional except where used."""
    latency_ema_ms: float | None = None
    energy_remaining_j: float | None = None
    memory_util: float | None = None  # 0..1


@dataclass
class OverrideThresholds:
    """If any field is set and exceeded, override fires."""
    max_latency_ms: float | None = None         # latency_ema_ms > this -> override
    min_energy_j: float | None = None           # energy_remaining_j < this -> override
    max_memory_util: float | None = None        # memory_util > this -> override


class OverrideLayer:
    """Strategy: given (state, telemetry, policy_action), return (override, fallback)."""

    def __init__(
        self,
        thresholds: OverrideThresholds,
        fallback_action: Any,
        cooldown_steps: int = 0,
    ):
        """
        fallback_action: opaque value the executor knows how to apply
            (e.g., int=0 for "lowest freq" in a discrete space, or a numpy
            array for continuous). The override layer does not interpret it.
        cooldown_steps: after override fires, keep firing for this many
            additional steps even if telemetry recovers, to avoid
            thrashing.  0 = no hysteresis.
        """
        self.thresholds = thresholds
        self.fallback_action = fallback_action
        self.cooldown_steps = int(cooldown_steps)
        self._cooldown_remaining = 0
        self.fire_count = 0
        self.last_reasons: list[str] = []

    def reset(self) -> None:
        self._cooldown_remaining = 0
        self.fire_count = 0
        self.last_reasons = []

    def _check(self, t: HardwareTelemetry) -> list[str]:
        reasons: list[str] = []
        th = self.thresholds
        if th.max_latency_ms is not None and t.latency_ema_ms is not None \
                and t.latency_ema_ms > th.max_latency_ms:
            reasons.append(f"latency {t.latency_ema_ms:.2f} > {th.max_latency_ms}")
        if th.min_energy_j is not None and t.energy_remaining_j is not None \
                and t.energy_remaining_j < th.min_energy_j:
            reasons.append(f"energy {t.energy_remaining_j:.2f} < {th.min_energy_j}")
        if th.max_memory_util is not None and t.memory_util is not None \
                and t.memory_util > th.max_memory_util:
            reasons.append(f"memory {t.memory_util:.2f} > {th.max_memory_util}")
        return reasons

    def step(self, telemetry: HardwareTelemetry) -> tuple[bool, Any | None]:
        """Returns (override_active, fallback_action_or_None).

        Logic:
          1. evaluate thresholds; if any violated, fire immediately and arm cooldown.
          2. else, if cooldown_remaining > 0, keep firing and decrement.
          3. else, no override - return (False, None).
        """
        reasons = self._check(telemetry)
        if reasons:
            self._cooldown_remaining = self.cooldown_steps
            self.fire_count += 1
            self.last_reasons = reasons
            return True, self.fallback_action
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self.fire_count += 1
            return True, self.fallback_action
        self.last_reasons = []
        return False, None
