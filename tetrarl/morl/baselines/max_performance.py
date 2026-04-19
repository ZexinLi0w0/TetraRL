"""MAX-P arbiter (always emit action 0 — max performance).

Behavioural surrogate baseline: always returns ``0``. In the TetraRL
DVFS action space, action index 0 corresponds to the highest-frequency
(max-performance) DVFS level.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class MaxPerformanceArbiter:
    """Always returns ``0`` (max-performance DVFS index)."""

    def __init__(self, n_actions: int, seed: int = 0, **kwargs: Any):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.n_actions = int(n_actions)
        self.seed = int(seed)

    def act(self, state: Any, omega: np.ndarray) -> int:
        return 0
