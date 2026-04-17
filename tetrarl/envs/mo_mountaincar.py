"""Multi-objective MountainCarContinuous environment.

Wraps gymnasium MountainCarContinuous-v0 with a 2-objective reward vector:
  - Objective 0 (position): +100 on reaching the goal, 0 otherwise.
  - Objective 1 (energy efficiency): -|action|^2 per step.

These objectives conflict: reaching the goal fast requires high-energy
actions, while energy efficiency demands minimal action magnitude.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class MOMountainCarContinuous(gym.Env):
    """MountainCarContinuous with vectorial reward [position, energy]."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    REFERENCE_POINT = np.array([0.0, -200.0], dtype=np.float64)

    APPROX_PARETO_FRONT = np.array(
        [
            [100.0, -75.0],
            [100.0, -50.0],
            [100.0, -30.0],
            [100.0, -20.0],
            [100.0, -15.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self._env = gym.make("MountainCarContinuous-v0", render_mode=render_mode)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.reward_dim = 2
        self.render_mode = render_mode

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        obs, scalar_reward, terminated, truncated, info = self._env.step(action)

        position_reward = 100.0 if terminated and obs[0] >= 0.45 else 0.0
        energy_penalty = -float(np.sum(action ** 2))

        reward_vec = np.array([position_reward, energy_penalty], dtype=np.float32)
        info["scalar_reward"] = scalar_reward
        return obs, reward_vec, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
