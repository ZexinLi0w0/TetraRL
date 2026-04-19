"""Envelope MORL arbiter (behavioural surrogate).

Surrogate of the envelope-based update from Yang, Sun & Narasimhan
(NeurIPS 2019), "A Generalized Algorithm for Multi-Objective
Reinforcement Learning and Policy Adaptation". The original method
trains a single network whose Q-values are projected through an
omega-conditioned envelope max operator.

For TetraRL's HV-vs-baseline comparison we only need the *inference*
behaviour: a deterministic, omega-conditioned greedy choice over a
fixed per-action feature matrix. Not a re-training; documented as a
behavioural surrogate.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class EnvelopeMORLArbiter:
    """Behavioural surrogate of Yang 2019 envelope-MO-RL inference.

    At construction we draw a deterministic per-action feature matrix
    ``F[a] in R^4`` from ``np.random.default_rng(seed)``. ``act`` then
    returns ``argmax_a (F @ omega)`` — pure deterministic, no RNG calls
    during ``act``.
    """

    def __init__(self, n_actions: int, seed: int = 0, **kwargs: Any):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.n_actions = int(n_actions)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        # 4-D feature matrix matches the 4-objective omega used across
        # the Week 10 HV evaluation matrix.
        self._features = rng.standard_normal((self.n_actions, 4))

    def act(self, state: Any, omega: np.ndarray) -> int:
        omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
        if omega_arr.shape[0] != self._features.shape[1]:
            # Pad / truncate so 2-D legacy omegas (e.g. DEFAULT_OMEGA)
            # still produce a valid scalar score per action.
            target = self._features.shape[1]
            if omega_arr.shape[0] < target:
                pad = np.zeros(target - omega_arr.shape[0], dtype=np.float64)
                omega_arr = np.concatenate([omega_arr, pad])
            else:
                omega_arr = omega_arr[:target]
        scores = self._features @ omega_arr
        return int(np.argmax(scores))
