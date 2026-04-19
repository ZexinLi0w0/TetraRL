"""MAX-A arbiter (always emit the top action index).

Behavioural surrogate baseline: always returns ``n_actions - 1``.
Provides an upper-action-index reference for HV comparison.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class MaxActionArbiter:
    """Always returns the top action index (``n_actions - 1``)."""

    def __init__(self, n_actions: int, seed: int = 0, **kwargs: Any):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.n_actions = int(n_actions)
        self.seed = int(seed)

    def act(self, state: Any, omega: np.ndarray) -> int:
        return int(self.n_actions - 1)
