"""FOCOPS arbiter (behavioural surrogate).

Surrogate of FOCOPS — Zhang, Vuong & Ross (NeurIPS 2020), "First-Order
Constrained Optimization in Policy Space". The original method does a
KL-projection of an unconstrained policy onto the constraint set via a
first-order update.

For TetraRL: a stochastic policy similar to PPO-Lag but with a
KL-smoothed (concave) bias on the action index. Logits:
``omega[0] * a - 0.5 * omega[3] * a**2 / max(1, n - 1)``. Softmax with
beta=1.5, sampled from a seeded RNG.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class FOCOPSArbiter:
    """Behavioural surrogate of FOCOPS (Zhang 2020) inference."""

    def __init__(self, n_actions: int, seed: int = 0, **kwargs: Any):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.n_actions = int(n_actions)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._beta = 1.5

    def act(self, state: Any, omega: np.ndarray) -> int:
        omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
        n = self.n_actions
        w_reward = float(omega_arr[0]) if omega_arr.shape[0] >= 1 else 0.0
        w_energy = float(omega_arr[3]) if omega_arr.shape[0] >= 4 else 0.0
        idxs = np.arange(n, dtype=np.float64)
        denom = float(max(1, n - 1))
        logits = w_reward * idxs - 0.5 * w_energy * (idxs * idxs) / denom
        scaled = self._beta * logits
        scaled = scaled - scaled.max()
        exp = np.exp(scaled)
        probs = exp / exp.sum()
        return int(self._rng.choice(n, p=probs))
