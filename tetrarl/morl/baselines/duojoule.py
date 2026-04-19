"""DuoJoule arbiter (behavioural surrogate).

Surrogate of the prior-group DuoJoule dual-objective energy/perf gate.
A simple deterministic threshold rule on the energy weight: when
``omega[3] > 0.5`` (energy-dominant), pick the low-energy action 0;
otherwise pick the high-perf action ``n_actions - 1``.

The constructor accepts ``seed`` for interface uniformity but the
policy itself is deterministic (no RNG calls).
"""
from __future__ import annotations

from typing import Any

import numpy as np


class DuoJouleArbiter:
    """Behavioural surrogate of DuoJoule's energy/perf threshold gate."""

    def __init__(self, n_actions: int, seed: int = 0, **kwargs: Any):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.n_actions = int(n_actions)
        self.seed = int(seed)

    def act(self, state: Any, omega: np.ndarray) -> int:
        omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
        w_energy = float(omega_arr[3]) if omega_arr.shape[0] >= 4 else 0.0
        if w_energy > 0.5:
            return 0
        return int(self.n_actions - 1)
