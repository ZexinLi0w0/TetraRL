"""DVFS-DRL-Multitask baseline (Algorithm 3 from "DVFS-DRL-Multitask" 2024).

Soft-deadline reward shaping plus a small omega-conditioned categorical
arbiter used as a Week 9 baseline against the TetraRL stack.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def soft_deadline_reward_shape(
    r_base: float,
    latency_ms: float,
    deadline_ms: float,
    lambda_: float = 1.0,
) -> float:
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    if deadline_ms < 0:
        raise ValueError("deadline_ms must be non-negative")
    excess = max(0.0, float(latency_ms) - float(deadline_ms))
    return float(r_base) - float(lambda_) * excess * excess


class DVFSDRLMultitaskArbiter:
    def __init__(
        self,
        n_actions: int,
        seed: int = 0,
        deadline_ms: float = 50.0,
        lambda_: float = 1.0,
    ):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.n_actions = int(n_actions)
        self.deadline_ms = float(deadline_ms)
        self.lambda_ = float(lambda_)
        self._rng = np.random.default_rng(int(seed))

    def act(self, state: Any, omega: np.ndarray) -> int:
        n = self.n_actions
        logits = np.zeros(n, dtype=np.float64)
        if n >= 1:
            logits[0] = float(omega[0])
        if n >= 2:
            logits[n - 1] = float(omega[1])
        beta = 1.0 + float(self.lambda_)
        scaled = beta * logits
        scaled -= scaled.max()
        exp = np.exp(scaled)
        probs = exp / exp.sum()
        return int(self._rng.choice(n, p=probs))
