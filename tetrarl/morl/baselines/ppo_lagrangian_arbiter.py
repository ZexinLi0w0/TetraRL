"""PPO-Lagrangian arbiter (behavioural surrogate).

Surrogate of PPO with a Lagrangian dual update on a constraint cost
(Achiam et al. 2017 / Stooke et al. 2020 style). The original method
trains a stochastic policy with a learned dual multiplier on a cost
constraint. We only need its *inference* behaviour for HV comparison.

For TetraRL: a stochastic policy whose logits are biased by the reward
weight ``omega[0]`` (favouring the top action) and the energy/cost
weight ``omega[3]`` (favouring the low-cost action). Sampled from a
seeded RNG so the action sequence is deterministic given (seed, omega).
"""
from __future__ import annotations

from typing import Any

import numpy as np


class PPOLagrangianArbiter:
    """Behavioural surrogate of PPO-Lagrangian inference.

    Reward weight pushes toward action ``a = n_actions - 1``; energy /
    cost weight pushes toward action ``a = 0``. Logits:
    ``omega[0] * a + omega[3] * (n_actions - 1 - a)``.
    """

    def __init__(self, n_actions: int, seed: int = 0, **kwargs: Any):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.n_actions = int(n_actions)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def act(self, state: Any, omega: np.ndarray) -> int:
        omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
        n = self.n_actions
        w_reward = float(omega_arr[0]) if omega_arr.shape[0] >= 1 else 0.0
        w_energy = float(omega_arr[3]) if omega_arr.shape[0] >= 4 else 0.0
        idxs = np.arange(n, dtype=np.float64)
        logits = w_reward * idxs + w_energy * (n - 1 - idxs)
        logits = logits - logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        return int(self._rng.choice(n, p=probs))
