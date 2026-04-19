"""Action masking strategies for preference-conditioned PPO.

Provides a strategy-pattern interface (`ActionMask`) that returns a per-action
boolean allow mask, plus two concrete implementations:

* `NoOpMask` -- identity mask, used when the env has no deadline structure.
* `DeadlineMask` -- DVFS-aware heuristic that masks frequency steps whose
  predicted execution time would miss the next deadline; the latency
  estimate is updated online via an EMA.

`apply_logit_mask` rewrites disallowed action logits with a large negative
constant (default -1e9) so the resulting `Categorical` distribution assigns
~0 probability to them while remaining numerically finite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class ActionMask(ABC):
    """Strategy interface returning a per-action allow mask.

    Returns shape (act_dim,) bool numpy array. True = allowed.
    Stateless across calls; impls hold any rolling state internally
    and have it updated by the system.
    """

    @abstractmethod
    def compute(self, state: np.ndarray, act_dim: int) -> np.ndarray: ...

    def as_tensor(
        self, state: np.ndarray, act_dim: int, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        """Convert the numpy mask to a bool tensor on the requested device."""
        mask = self.compute(state, act_dim)
        return torch.as_tensor(mask, dtype=torch.bool, device=device)


class NoOpMask(ActionMask):
    """Allow everything. Safe default for envs without deadlines."""

    def compute(self, state: np.ndarray, act_dim: int) -> np.ndarray:
        return np.ones(act_dim, dtype=bool)


class DeadlineMask(ActionMask):
    """Heuristic: mask DVFS actions whose predicted exec time exceeds the deadline.

    Actions are ordered slowest -> fastest by `freq_scale` (action a runs
    at relative speed freq_scale[a]).  Predicted time = latency_ms / freq_scale[a].
    The system updates `latency_ms` via `update_latency()` after each step
    using an EMA. If every action would miss the deadline, the fastest action
    (argmax of freq_scale) stays allowed so the policy is never handed an
    empty mask.

    Args:
        freq_scale: list/array of positive floats, length == act_dim
        deadline_ms: budget per step
        ema_alpha: smoothing for latency EMA (0..1)
        initial_latency_ms: starting EMA value
    """

    def __init__(
        self,
        freq_scale,
        deadline_ms: float,
        ema_alpha: float = 0.2,
        initial_latency_ms: float = 1.0,
    ) -> None:
        scale = np.asarray(freq_scale, dtype=np.float64)
        if scale.ndim != 1 or scale.size == 0 or np.any(scale <= 0):
            raise ValueError(
                "freq_scale must be a non-empty 1-D array of positive floats"
            )
        self.freq_scale = scale
        self.deadline_ms = float(deadline_ms)
        self.ema_alpha = float(ema_alpha)
        self.latency_ms = float(initial_latency_ms)

    def update_latency(self, observed_ms: float) -> None:
        a = self.ema_alpha
        self.latency_ms = (1.0 - a) * self.latency_ms + a * float(observed_ms)

    def compute(self, state: np.ndarray, act_dim: int) -> np.ndarray:
        if act_dim != self.freq_scale.size:
            raise ValueError(
                f"act_dim {act_dim} does not match freq_scale length {self.freq_scale.size}"
            )
        predicted = self.latency_ms / self.freq_scale
        mask = predicted <= self.deadline_ms
        if not mask.any():
            mask[int(np.argmax(self.freq_scale))] = True
        return mask.astype(bool)


def apply_logit_mask(
    logits: torch.Tensor, mask: torch.Tensor, fill_value: float = -1e9
) -> torch.Tensor:
    """Replace masked logits with `fill_value` (default -1e9, NOT -inf to keep softmax finite).

    Shapes must broadcast. Cast non-bool masks to bool first. Returns a new tensor.
    Use torch.where rather than in-place masking so autograd works.
    """
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    fill = torch.full_like(logits, fill_value)
    return torch.where(mask, logits, fill)
