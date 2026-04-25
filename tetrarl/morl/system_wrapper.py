"""SystemWrapper ABC — system-layer DRL training wrapper interface.

A SystemWrapper modulates *how* a DRL algorithm is trained at the system layer
(DVFS frequency, batch size, replay capacity, mixed-precision flag) without
changing the algorithm's policy. The 5 concrete wrappers in
``system_wrappers.py`` (MaxA, MaxP, R3, DuoJoule, TetraRL) all implement this
contract so the unified P15 runner can drop any (algo, wrapper) pair into the
same training loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Type


@dataclass
class WrapperKnobs:
    """System-level knobs returned by ``SystemWrapper.step_hook``.

    Any field set to ``None`` means the wrapper has no opinion this step and
    the runner should keep the previous value.
    """

    dvfs_idx: int | None = None
    batch_size: int | None = None
    replay_capacity: int | None = None
    mixed_precision: bool | None = None
    # If set, the runner overrides the algo's chosen action with this value
    # for the current env.step. Used by override-layer style wrappers
    # (e.g. TetraRL when a memory-pressure threshold fires).
    action_override: int | None = None
    # Free-form bag for wrapper-specific signals the runner can ignore.
    extras: dict[str, Any] = field(default_factory=dict)


class SystemWrapper(ABC):
    """Abstract base class for system-layer DRL wrappers.

    Concrete wrappers must implement:

    - :meth:`is_compatible` — declare which algorithm classes this wrapper can
      legally wrap. R3/DuoJoule, for example, return ``False`` for on-policy
      algos (A2C/PPO) because their core mechanism (replay buffer / off-policy
      replay ratio) does not exist in those algos.
    - :meth:`wrap` — bind to an algorithm instance (one-shot at runner start).
    - :meth:`step_hook` — called once per training step with the current step
      index and a dict view of the algo's state; returns the
      :class:`WrapperKnobs` the runner should apply.
    - :meth:`get_metrics` — wrapper-specific aggregated metrics (deadline-miss
      rate, override fire count, mean energy/step, ...) for ``summary.json``.
    """

    name: str = "abstract"

    @abstractmethod
    def is_compatible(self, algo_class: Type[Any]) -> bool: ...

    @abstractmethod
    def wrap(self, algo_instance: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def step_hook(self, step_idx: int, algo_state: dict[str, Any]) -> WrapperKnobs: ...

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]: ...
