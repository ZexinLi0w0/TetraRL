"""Gymnasium env wrappers used by the unified eval runner.

``MOAggregateWrapper`` scalarises a multi-objective env's vector reward
via an inner product with a fixed preference vector (omega). The
unified eval loop in :mod:`tetrarl.eval.runner` consumes scalar rewards,
so wrapping the multi-objective env at the boundary keeps the loop
unchanged while letting the matrix YAML drive different Pareto-front
corners by changing only ``EvalConfig.extra["omega"]``.

The original 4-vector is preserved in ``info["reward_vec"]`` so the
analysis layer can recover per-objective signals after the fact.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class MOAggregateWrapper(gym.Wrapper):
    """Scalarise a vector-reward env via ``omega @ reward_vec``.

    Parameters
    ----------
    env:
        The wrapped Gymnasium env. Must return an array-like reward of
        the same length as ``omega`` from ``step``.
    omega:
        Preference vector. Stored as a float32 ``np.ndarray``.
    """

    def __init__(self, env: gym.Env, omega: np.ndarray):
        super().__init__(env)
        self._omega = np.asarray(omega, dtype=np.float32).copy()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward_vec, terminated, truncated, info = self.env.step(action)
        r_vec = np.asarray(reward_vec, dtype=np.float32)
        scalar = float(np.dot(self._omega, r_vec))
        out_info = dict(info) if info else {}
        out_info["reward_vec"] = [float(x) for x in r_vec]
        return obs, scalar, bool(terminated), bool(truncated), out_info
