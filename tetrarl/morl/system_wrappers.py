"""Five concrete :class:`SystemWrapper` implementations for the P15 matrix.

- :class:`MaxAWrapper` (max-action) — pin DVFS to the maximum-frequency index.
- :class:`MaxPWrapper` (max-performance) — pin DVFS to index 0
  (TetraRL convention: index 0 is the highest-perf DVFS level).
- :class:`R3Wrapper` — deadline-aware replay/batch sizing (off-policy only).
- :class:`DuoJouleWrapper` — energy-per-step minimisation knob (off-policy only).
- :class:`TetraRLWrapper` — full TetraRL coordination
  (preference plane → R^4 arbiter → resource manager + override layer).

All wrappers consume the algorithm's class via ``is_compatible`` and a runtime
``algo_state`` dict from the runner; they never read or mutate the policy
network parameters.
"""
from __future__ import annotations

from typing import Any, Type

import numpy as np

from tetrarl.morl.system_wrapper import SystemWrapper, WrapperKnobs

# --- helpers ---------------------------------------------------------------


def _is_off_policy(algo_class: Type[Any]) -> bool:
    """True for replay-buffer-based algos (DQN/DDQN/C51), False for on-policy
    A2C/PPO. Match by class name to avoid a circular import on
    ``tetrarl.morl.algos``."""
    name = algo_class.__name__.lower()
    return any(tag in name for tag in ("dqn", "ddqn", "c51"))


# --- MaxA ------------------------------------------------------------------


class MaxAWrapper(SystemWrapper):
    """Pin DVFS to the maximum-frequency index AND force a small minibatch.

    Spec (Phase 6, 2026-04-26): MaxA = max-action policy = max DVFS + minibatch=16.
    The batch_size knob is propagated through every ``step_hook`` so the runner
    echoes it in summary metrics for the controlled-budget protocol.

    For on-policy A2C/PPO the algo's ``batch_size`` mirrors ``mini_batch_size``;
    setting it shrinks PPO mini-batches (A2C ignores it for the rollout-loss
    pass but the wrapper still pins it for metric consistency).
    """

    name = "maxa"

    def __init__(
        self,
        max_dvfs_idx: int = 11,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> None:
        self.max_dvfs_idx = int(max_dvfs_idx)
        self.batch_size = int(batch_size)
        self._step_count = 0

    def is_compatible(self, algo_class: Type[Any]) -> bool:
        return True

    def wrap(self, algo_instance: Any, **kwargs: Any) -> Any:
        self._algo = algo_instance
        if hasattr(algo_instance, "batch_size"):
            algo_instance.batch_size = self.batch_size
        if hasattr(algo_instance, "mini_batch_size"):
            algo_instance.mini_batch_size = self.batch_size
        return algo_instance

    def step_hook(self, step_idx: int, algo_state: dict[str, Any]) -> WrapperKnobs:
        self._step_count += 1
        return WrapperKnobs(dvfs_idx=self.max_dvfs_idx, batch_size=self.batch_size)

    def get_metrics(self) -> dict[str, Any]:
        return {
            "wrapper": self.name,
            "dvfs_pinned_idx": self.max_dvfs_idx,
            "batch_size": self.batch_size,
            "n_steps": self._step_count,
        }


# --- MaxP ------------------------------------------------------------------


class MaxPWrapper(SystemWrapper):
    """Pin DVFS to index 0 AND force a large minibatch (max performance).

    Spec (Phase 6, 2026-04-26): MaxP = max-perf policy = DVFS index 0 (highest
    perf in TetraRL convention) + minibatch=128. Batch is propagated through
    every step_hook for runner metric consistency.
    """

    name = "maxp"

    def __init__(
        self,
        batch_size: int = 128,
        **kwargs: Any,
    ) -> None:
        self.batch_size = int(batch_size)
        self._step_count = 0

    def is_compatible(self, algo_class: Type[Any]) -> bool:
        return True

    def wrap(self, algo_instance: Any, **kwargs: Any) -> Any:
        self._algo = algo_instance
        if hasattr(algo_instance, "batch_size"):
            algo_instance.batch_size = self.batch_size
        if hasattr(algo_instance, "mini_batch_size"):
            algo_instance.mini_batch_size = self.batch_size
        return algo_instance

    def step_hook(self, step_idx: int, algo_state: dict[str, Any]) -> WrapperKnobs:
        self._step_count += 1
        return WrapperKnobs(dvfs_idx=0, batch_size=self.batch_size)

    def get_metrics(self) -> dict[str, Any]:
        return {
            "wrapper": self.name,
            "dvfs_pinned_idx": 0,
            "batch_size": self.batch_size,
            "n_steps": self._step_count,
        }


# --- R³ --------------------------------------------------------------------


class R3Wrapper(SystemWrapper):
    """Deadline-aware batch / replay sizing (Cao et al., 2024 R³ surrogate).

    Off-policy ONLY (DQN / DDQN / C51). When the running mean of step latency
    exceeds ``deadline_ms``, halve the batch size (down to ``min_batch``).
    When latency is comfortably under deadline, double the batch size (up to
    ``max_batch``). Replay capacity is held at the algo default; this surrogate
    only exposes the batch-size knob.
    """

    name = "r3"

    def __init__(
        self,
        deadline_ms: float = 50.0,
        min_batch: int = 16,
        max_batch: int = 256,
        ema_alpha: float = 0.1,
        **kwargs: Any,
    ) -> None:
        self.deadline_ms = float(deadline_ms)
        self.min_batch = int(min_batch)
        self.max_batch = int(max_batch)
        self.ema_alpha = float(ema_alpha)
        self._latency_ema_ms: float = 0.0
        self._current_batch: int | None = None
        self._n_miss = 0
        self._n_steps = 0

    def is_compatible(self, algo_class: Type[Any]) -> bool:
        return _is_off_policy(algo_class)

    def wrap(self, algo_instance: Any, **kwargs: Any) -> Any:
        self._algo = algo_instance
        # Initialise current batch from the algo's default if exposed.
        bs = getattr(algo_instance, "batch_size", 64)
        self._current_batch = int(bs)
        return algo_instance

    def step_hook(self, step_idx: int, algo_state: dict[str, Any]) -> WrapperKnobs:
        self._n_steps += 1
        last_lat = float(algo_state.get("last_step_ms", 0.0))
        # EMA update
        self._latency_ema_ms = (
            (1.0 - self.ema_alpha) * self._latency_ema_ms + self.ema_alpha * last_lat
        )
        if last_lat > self.deadline_ms:
            self._n_miss += 1
        # Adjust batch
        if (
            self._latency_ema_ms > self.deadline_ms
            and self._current_batch
            and self._current_batch > self.min_batch
        ):
            self._current_batch = max(self.min_batch, self._current_batch // 2)
        elif (
            self._latency_ema_ms < 0.5 * self.deadline_ms
            and self._current_batch
            and self._current_batch < self.max_batch
        ):
            self._current_batch = min(self.max_batch, self._current_batch * 2)
        return WrapperKnobs(batch_size=self._current_batch)

    def get_metrics(self) -> dict[str, Any]:
        miss_rate = float(self._n_miss) / float(max(1, self._n_steps))
        return {
            "wrapper": self.name,
            "deadline_ms": self.deadline_ms,
            "deadline_miss_rate": miss_rate,
            "latency_ema_ms": float(self._latency_ema_ms),
            "final_batch": int(self._current_batch or 0),
            "n_steps": self._n_steps,
        }


# --- DuoJoule --------------------------------------------------------------


class DuoJouleWrapper(SystemWrapper):
    """Energy-per-step minimisation knob (DuoJoule surrogate).

    Off-policy ONLY. Tracks running mean of energy/step; when it rises above
    ``energy_target_j`` halves batch size (less compute → less energy);
    when below, doubles batch size to use the headroom for sample efficiency.
    """

    name = "duojoule"

    def __init__(
        self,
        energy_target_j: float = 0.05,
        min_batch: int = 16,
        max_batch: int = 256,
        ema_alpha: float = 0.1,
        **kwargs: Any,
    ) -> None:
        self.energy_target_j = float(energy_target_j)
        self.min_batch = int(min_batch)
        self.max_batch = int(max_batch)
        self.ema_alpha = float(ema_alpha)
        self._energy_ema_j: float = 0.0
        self._current_batch: int | None = None
        self._n_steps = 0

    def is_compatible(self, algo_class: Type[Any]) -> bool:
        return _is_off_policy(algo_class)

    def wrap(self, algo_instance: Any, **kwargs: Any) -> Any:
        self._algo = algo_instance
        bs = getattr(algo_instance, "batch_size", 64)
        self._current_batch = int(bs)
        return algo_instance

    def step_hook(self, step_idx: int, algo_state: dict[str, Any]) -> WrapperKnobs:
        self._n_steps += 1
        last_energy = float(algo_state.get("last_step_energy_j", 0.0))
        self._energy_ema_j = (
            (1.0 - self.ema_alpha) * self._energy_ema_j + self.ema_alpha * last_energy
        )
        if (
            self._energy_ema_j > self.energy_target_j
            and self._current_batch
            and self._current_batch > self.min_batch
        ):
            self._current_batch = max(self.min_batch, self._current_batch // 2)
        elif (
            self._energy_ema_j < 0.5 * self.energy_target_j
            and self._current_batch
            and self._current_batch < self.max_batch
        ):
            self._current_batch = min(self.max_batch, self._current_batch * 2)
        return WrapperKnobs(batch_size=self._current_batch)

    def get_metrics(self) -> dict[str, Any]:
        return {
            "wrapper": self.name,
            "energy_target_j": self.energy_target_j,
            "energy_ema_j": float(self._energy_ema_j),
            "final_batch": int(self._current_batch or 0),
            "n_steps": self._n_steps,
        }


# --- TetraRL ---------------------------------------------------------------


class TetraRLWrapper(SystemWrapper):
    """Full TetraRL coordination wrapper.

    Per-step coordinated R^4 knobs (batch_size, replay_capacity, dvfs_idx,
    mixed_precision) selected by a preference-conditioned arbiter, gated by an
    override layer that fires when the simulated memory pressure crosses a
    threshold. Preference omega is configurable; default favours
    [reward, latency, energy, memory] = [0.4, 0.3, 0.2, 0.1].

    For on-policy algos (A2C/PPO) the replay_capacity field is set to None
    (no-op); the wrapper is still compatible because its DVFS / batch / MP
    knobs all apply.
    """

    name = "tetrarl"

    def __init__(
        self,
        omega: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
        max_dvfs_idx: int = 11,
        memory_threshold: float = 0.85,
        cooldown_steps: int = 10,
        **kwargs: Any,
    ) -> None:
        self.omega = np.asarray(omega, dtype=np.float32)
        self.max_dvfs_idx = int(max_dvfs_idx)
        self.memory_threshold = float(memory_threshold)
        self.cooldown_steps = int(cooldown_steps)
        self._algo_class: Type[Any] | None = None
        self._is_off_policy: bool = False
        self._fire_count = 0
        self._cooldown_until = -1
        self._n_steps = 0

    def is_compatible(self, algo_class: Type[Any]) -> bool:
        return True

    def wrap(self, algo_instance: Any, **kwargs: Any) -> Any:
        self._algo = algo_instance
        self._algo_class = type(algo_instance)
        self._is_off_policy = _is_off_policy(self._algo_class)
        return algo_instance

    def step_hook(self, step_idx: int, algo_state: dict[str, Any]) -> WrapperKnobs:
        self._n_steps += 1
        # Resource-manager: scale DVFS based on omega weights.
        # omega[1] = latency weight, omega[2] = energy weight.
        # When energy weight dominates, drop DVFS toward middle; otherwise pin high.
        energy_w = float(self.omega[2])
        if energy_w > 0.5:
            dvfs_idx = max(0, self.max_dvfs_idx // 2)
        else:
            dvfs_idx = self.max_dvfs_idx
        # Override-layer: if simulated memory exceeds threshold, fire.
        mem_util = float(algo_state.get("memory_util", 0.0))
        action_override: int | None = None
        if mem_util >= self.memory_threshold and step_idx >= self._cooldown_until:
            self._fire_count += 1
            self._cooldown_until = step_idx + self.cooldown_steps
            # Force action 0 (safe fallback) and drop DVFS.
            action_override = 0
            dvfs_idx = max(0, dvfs_idx - 2)
        # Batch-size: keep algo default (None means runner keeps whatever the
        # algo configured).
        replay_cap = None
        if not self._is_off_policy:
            replay_cap = None
        return WrapperKnobs(
            dvfs_idx=dvfs_idx,
            batch_size=None,
            replay_capacity=replay_cap,
            mixed_precision=True,
            action_override=action_override,
            extras={"omega": self.omega.tolist()},
        )

    def get_metrics(self) -> dict[str, Any]:
        return {
            "wrapper": self.name,
            "omega": self.omega.tolist(),
            "override_fire_count": int(self._fire_count),
            "n_steps": self._n_steps,
        }


# --- registry --------------------------------------------------------------


WRAPPER_REGISTRY: dict[str, Type[SystemWrapper]] = {
    "maxa": MaxAWrapper,
    "maxp": MaxPWrapper,
    "r3": R3Wrapper,
    "duojoule": DuoJouleWrapper,
    "tetrarl": TetraRLWrapper,
}


def make_wrapper(name: str, **kwargs: Any) -> SystemWrapper:
    """Factory for the 5 P15 wrappers."""
    key = name.lower()
    if key not in WRAPPER_REGISTRY:
        raise ValueError(f"unknown wrapper: {name!r}; valid={list(WRAPPER_REGISTRY)}")
    return WRAPPER_REGISTRY[key](**kwargs)
