"""Deep Sea Treasure (DST) environment for multi-objective RL.

A standard MORL benchmark (Vamplew et al., 2011) with two objectives:
treasure value (to maximize) and time penalty (to minimize, -1 per step).
The agent navigates an 11x10 grid to reach one of 10 treasures, each
at a different depth and with a different value.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DeepSeaTreasure(gym.Env):
    """Deep Sea Treasure environment with vectorial reward.

    The grid is 11 columns x 10 rows. The agent starts at (0, 0) and
    can move in 4 cardinal directions. Each treasure is located at a
    specific (row, col) position with a known value.

    Reward vector: [treasure_value, time_penalty].
    - treasure_value: value of collected treasure (0 if none).
    - time_penalty: -1 per step.
    """

    metadata = {"render_modes": ["ansi"]}

    TREASURE_MAP: dict[tuple[int, int], float] = {
        (1, 0): 1,
        (2, 1): 2,
        (3, 2): 3,
        (4, 3): 5,
        (4, 4): 8,
        (4, 5): 16,
        (7, 6): 24,
        (7, 7): 50,
        (9, 8): 74,
        (10, 9): 124,
    }

    PARETO_OPTIMAL_RETURNS: list[tuple[float, float]] = [
        (1, -1),
        (2, -3),
        (3, -5),
        (5, -7),
        (8, -8),
        (16, -9),
        (24, -13),
        (50, -14),
        (74, -17),
        (124, -19),
    ]

    GRID_ROWS = 11
    GRID_COLS = 10
    MAX_STEPS = 200

    _sea_map: np.ndarray

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.GRID_ROWS - 1, self.GRID_COLS - 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)
        self.reward_dim = 2
        self._build_sea_map()
        self._pos = np.array([0, 0])
        self._step_count = 0

    def _build_sea_map(self) -> None:
        self._sea_map = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.int8)
        treasure_cols = sorted(set(c for _, c in self.TREASURE_MAP))
        for c in treasure_cols:
            max_row = max(r for (r, cc) in self.TREASURE_MAP if cc == c)
            for r in range(max_row + 1):
                self._sea_map[r, c] = 1
        self._sea_map[0, :] = 1

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._pos = np.array([0, 0])
        self._step_count = 0
        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[action]
        new_r = np.clip(self._pos[0] + dr, 0, self.GRID_ROWS - 1)
        new_c = np.clip(self._pos[1] + dc, 0, self.GRID_COLS - 1)

        if self._sea_map[new_r, new_c] == 1:
            self._pos = np.array([new_r, new_c])

        self._step_count += 1

        treasure_val = self.TREASURE_MAP.get(tuple(self._pos), 0.0)
        reward_vec = np.array([treasure_val, -1.0], dtype=np.float32)
        terminated = treasure_val > 0
        truncated = self._step_count >= self.MAX_STEPS

        return self._get_obs(), reward_vec, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        return self._pos.astype(np.float32).copy()

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        rows = []
        for r in range(self.GRID_ROWS):
            row_str = ""
            for c in range(self.GRID_COLS):
                if np.array_equal(self._pos, [r, c]):
                    row_str += "A "
                elif (r, c) in self.TREASURE_MAP:
                    row_str += "T "
                elif self._sea_map[r, c]:
                    row_str += ". "
                else:
                    row_str += "# "
            rows.append(row_str)
        grid = "\n".join(rows)
        print(grid)
        return grid
