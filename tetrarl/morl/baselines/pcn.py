"""PCN arbiter (behavioural surrogate).

Surrogate of Pareto Conditioned Networks — Reymond, Bargiacchi & Nowe
(AAMAS 2022), "Pareto Conditioned Networks". The original method
trains a single network conditioned on a desired return vector;
inference samples actions from that network. PCN is *discrete-action
only* — the arbiter validates this at construction time.

For TetraRL: at construction we draw two reference action distributions
``A`` and ``B`` from the constructor RNG. ``act`` blends them via
``p = omega[0] * A + (1 - omega[0]) * B``, normalises, and samples
from a separately seeded RNG (``seed + 1``) so the deterministic test
that builds two arbiters with the same seed sees identical sequences.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class PCNArbiter:
    """Behavioural surrogate of Pareto-Conditioned Networks inference.

    Discrete-action only. ``__init__`` raises ``ValueError`` when
    ``n_actions <= 0`` — this also covers the test path where we pass
    ``n_actions=0`` to simulate a continuous-action environment.
    """

    def __init__(self, n_actions: int, seed: int = 0, **kwargs: Any):
        if n_actions <= 0:
            raise ValueError(
                "PCN requires a discrete action space with n_actions > 0"
            )
        self.n_actions = int(n_actions)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        a_logits = rng.standard_normal(self.n_actions)
        b_logits = rng.standard_normal(self.n_actions)
        self._a_dist = self._softmax(a_logits)
        self._b_dist = self._softmax(b_logits)
        # Separate act-time RNG so two arbiters with the same seed
        # produce identical act() sequences (the deterministic test).
        self._act_rng = np.random.default_rng(self.seed + 1)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def act(self, state: Any, omega: np.ndarray) -> int:
        omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
        w0 = float(omega_arr[0]) if omega_arr.shape[0] >= 1 else 0.5
        w0 = max(0.0, min(1.0, w0))
        p = w0 * self._a_dist + (1.0 - w0) * self._b_dist
        p = p / p.sum()
        return int(self._act_rng.choice(self.n_actions, p=p))
