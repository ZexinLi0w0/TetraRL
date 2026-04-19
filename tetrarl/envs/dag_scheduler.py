"""Synthetic DAG scheduling environment with vectorial reward.

A small Gymnasium env designed to validate the GCN feature extractor in
`tetrarl/morl/native/gnn_extractor.py`. Observations are graph-structured
(node features + edge index + ready mask), actions pick which task to
schedule next, and the reward is by default a 4-vector
``[throughput, -energy_step, -peak_memory_delta, -energy_normalized_step]``.

The 4th component is the per-step energy contribution
``(task_compute_cost * dvfs_scaling_factor) / max_energy`` (negated), where
``max_energy`` is the maximum cumulative energy possible in the episode at
``dvfs=1.0`` (i.e. the sum of all task compute costs). Summing the 4th
component over a full episode at ``dvfs=1.0`` therefore yields exactly
``-1.0``. A backward-compat constructor flag ``reward_dim=3`` drops the
4th component for legacy callers.

Topology generation follows a simple Erdos-Renyi style rule on a
topologically-ordered node set: for every ordered pair (u, v) with u < v,
add edge u -> v with probability ``density``. This guarantees acyclicity
without needing a separate DAG-ification pass.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tetrarl.morl.native.masking import ActionMask


def generate_random_dag(
    n_tasks: int,
    density: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a random DAG and per-task cost vector.

    Args:
        n_tasks: number of nodes in the DAG.
        density: probability in [0, 1] of including each topologically-
            valid edge ``u -> v`` with ``u < v``.
        rng: numpy ``Generator`` for reproducibility.

    Returns:
        edge_index: int64 array of shape ``(2, E)`` containing edges
            ``(u, v)`` with ``u < v``.
        node_costs: float32 array of shape ``(n_tasks, 3)`` with columns
            ``(compute_cost, memory_cost, deadline_window)``, each
            strictly positive.
    """
    sources: list[int] = []
    targets: list[int] = []
    for u in range(n_tasks):
        for v in range(u + 1, n_tasks):
            if rng.random() < density:
                sources.append(u)
                targets.append(v)
    if sources:
        edge_index = np.asarray([sources, targets], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    compute = rng.uniform(0.5, 1.5, size=n_tasks).astype(np.float32)
    memory = rng.uniform(0.5, 1.5, size=n_tasks).astype(np.float32)
    deadline = rng.uniform(2.0, 5.0, size=n_tasks).astype(np.float32)
    node_costs = np.stack([compute, memory, deadline], axis=1).astype(np.float32)
    return edge_index, node_costs


class DAGSchedulerEnv(gym.Env):
    """Schedule tasks from a synthetic DAG one-at-a-time.

    Each step the policy picks a task index. If the task is *ready*
    (not done and all predecessors done) it is executed and contributes
    a positive throughput reward, a negative energy term proportional to
    its compute cost, and a negative peak-memory increment when the
    cumulative live memory crosses a new high-water mark. Invalid actions
    are no-ops that still consume a step.

    Reward vector layout (default ``reward_dim=4``):
    ``[throughput, -energy_step, -peak_memory_delta, -energy_normalized_step]``.
    With ``reward_dim=3`` the 4th component is dropped for backward compat.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_tasks: int = 8,
        density: float = 0.3,
        max_steps: int | None = None,
        seed: int = 0,
        *,
        reward_dim: int = 4,
        dvfs_scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if n_tasks <= 0:
            raise ValueError("n_tasks must be positive")
        if reward_dim not in (3, 4):
            raise ValueError("reward_dim must be 3 or 4")
        if dvfs_scaling_factor <= 0:
            raise ValueError("dvfs_scaling_factor must be > 0")
        self.n_tasks = int(n_tasks)
        self.density = float(density)
        self.max_steps = int(max_steps) if max_steps is not None else 4 * self.n_tasks
        self.reward_dim = int(reward_dim)
        self.dvfs_scaling_factor = float(dvfs_scaling_factor)

        self.e_max = self.n_tasks * (self.n_tasks - 1) // 2

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.n_tasks, 4),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=-1,
                    high=self.n_tasks,
                    shape=(2, max(self.e_max, 1)),
                    dtype=np.int64,
                ),
                "num_edges": spaces.Box(
                    low=0,
                    high=max(self.e_max, 1),
                    shape=(),
                    dtype=np.int64,
                ),
                "valid_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_tasks,),
                    dtype=np.int8,
                ),
            }
        )
        self.action_space = spaces.Discrete(self.n_tasks)

        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

        # Per-episode state, populated by reset().
        self._edge_index: np.ndarray = np.zeros((2, 0), dtype=np.int64)
        self._node_costs: np.ndarray = np.zeros((self.n_tasks, 3), dtype=np.float32)
        self._done_mask: np.ndarray = np.zeros(self.n_tasks, dtype=bool)
        self._step_count: int = 0
        self._peak_mem: float = 0.0
        self._max_energy: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)

        self._edge_index, self._node_costs = generate_random_dag(
            self.n_tasks, self.density, self._rng
        )
        self._done_mask = np.zeros(self.n_tasks, dtype=bool)
        self._step_count = 0
        self._peak_mem = 0.0
        self._max_energy = float(self._node_costs[:, 0].sum())
        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], np.ndarray, bool, bool, dict[str, Any]]:
        action = int(action)
        valid = self._compute_valid_mask()

        if not (0 <= action < self.n_tasks) or not bool(valid[action]):
            reward_vec = np.zeros(self.reward_dim, dtype=np.float32)
        else:
            self._done_mask[action] = True
            compute_cost = float(self._node_costs[action, 0])
            energy_delta = -compute_cost
            current_mem = float(self._node_costs[self._done_mask, 1].sum())
            prior_peak = self._peak_mem
            if current_mem > prior_peak:
                memory_delta = -(current_mem - prior_peak)
                self._peak_mem = current_mem
            else:
                memory_delta = 0.0
            # Defensive zero-division guard: degenerate DAGs may have zero cost.
            if self._max_energy > 0.0:
                energy_norm_delta = -(compute_cost * self.dvfs_scaling_factor) / self._max_energy
            else:
                energy_norm_delta = 0.0
            full_vec = [1.0, energy_delta, memory_delta, energy_norm_delta]
            reward_vec = np.asarray(full_vec[: self.reward_dim], dtype=np.float32)

        self._step_count += 1
        terminated = bool(self._done_mask.all())
        truncated = bool(self._step_count >= self.max_steps)
        return self._get_obs(), reward_vec, terminated, truncated, {}

    def close(self) -> None:  # pragma: no cover - trivial
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_valid_mask(self) -> np.ndarray:
        """Return a per-task boolean mask of *ready* tasks."""
        ready = ~self._done_mask
        if self._edge_index.shape[1] > 0:
            srcs = self._edge_index[0]
            dsts = self._edge_index[1]
            # A task t is blocked if any predecessor is not yet done.
            for u, v in zip(srcs.tolist(), dsts.tolist()):
                if not self._done_mask[u]:
                    ready[v] = False
        return ready

    def _get_obs(self) -> dict[str, np.ndarray]:
        node_feats = np.zeros((self.n_tasks, 4), dtype=np.float32)
        node_feats[:, 0] = self._node_costs[:, 0]
        node_feats[:, 1] = self._node_costs[:, 1]
        node_feats[:, 2] = self._node_costs[:, 2]
        node_feats[:, 3] = self._done_mask.astype(np.float32)

        padded = np.full((2, max(self.e_max, 1)), -1, dtype=np.int64)
        n_edges = int(self._edge_index.shape[1])
        if n_edges > 0:
            padded[:, :n_edges] = self._edge_index

        valid_mask = self._compute_valid_mask().astype(np.int8)

        return {
            "node_features": node_feats,
            "edge_index": padded,
            "num_edges": np.asarray(n_edges, dtype=np.int64),
            "valid_mask": valid_mask,
        }


class DAGReadyMask(ActionMask):
    """Action mask that reads the precomputed ``valid_mask`` from a dict obs.

    Useful for plugging the env into preference-conditioned PPO without
    having to re-derive readiness from the graph at every forward pass.
    """

    def compute(self, state: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
        if not isinstance(state, dict) or "valid_mask" not in state:
            raise ValueError(
                "DAGReadyMask expects a dict observation containing 'valid_mask'"
            )
        mask = np.asarray(state["valid_mask"], dtype=bool)
        if mask.shape != (act_dim,):
            raise ValueError(
                f"valid_mask shape {mask.shape} does not match act_dim {act_dim}"
            )
        return mask
