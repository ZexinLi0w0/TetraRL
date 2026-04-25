"""DuoJoule (Yan et al. RTSS'24) re-implementation.

Re-implementation following Yan et al. RTSS'24 description; original code
not publicly available. DuoJoule wraps a base RL agent and switches its
hyperparameters (batch size B and replay ratio R) at episode boundaries
using a greedy efficiency-score controller.

Efficiency score combines an energy proxy (per-update FLOP surrogate) and
a latency proxy (measured wall-clock per update). On each ``end_episode``
call, the controller scores the just-finished episode under the current
(B, R). If the score worsened versus the previous accepted score it
reverts; otherwise it explores a random axis-aligned neighbour. DVFS is
a stub on DST since DST is too small to drive real hardware DVFS.

The legacy :class:`DuoJouleArbiter` is kept for back-compat with the eval
runner registry (``agent_type='duojoule'``) and continues to expose the
omega-threshold gate that downstream Week-10 tests already depend on.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np

from tetrarl.morl.agents.pd_morl import PDMORLAgent, Transition

# Three Linear layers in MOQNetwork (state+omega -> hidden -> hidden -> head).
_FLOP_LAYERS = 3


class DuoJouleAgent:
    """Trainable DuoJoule controller wrapping a :class:`PDMORLAgent`.

    The wrapper is transparent for ``act``/``store``: it delegates to the
    base agent. The interesting bits are ``update`` (which fires the base
    update R times per call to honour the replay ratio while tracking
    wall-clock and FLOP proxies) and ``end_episode`` (which runs the
    greedy controller).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_objectives: int = 2,
        hidden_dim: int = 256,
        lr: float = 5e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        seed: int = 0,
        batch_sizes: tuple[int, ...] = (32, 64, 128, 256),
        replay_ratios: tuple[int, ...] = (1, 2, 4),
        initial_batch_idx: int = 1,
        initial_replay_idx: int = 0,
        w_energy: float = 0.5,
        w_latency: float = 0.5,
        **base_kwargs: Any,
    ):
        if not batch_sizes or not replay_ratios:
            raise ValueError("batch_sizes and replay_ratios must be non-empty")
        if not (0 <= initial_batch_idx < len(batch_sizes)):
            raise ValueError("initial_batch_idx out of range")
        if not (0 <= initial_replay_idx < len(replay_ratios)):
            raise ValueError("initial_replay_idx out of range")

        self.batch_sizes = tuple(int(b) for b in batch_sizes)
        self.replay_ratios = tuple(int(r) for r in replay_ratios)
        self.w_energy = float(w_energy)
        self.w_latency = float(w_latency)
        self.hidden_dim = int(hidden_dim)
        self.n_objectives = int(n_objectives)
        self.seed = int(seed)
        self._rng = random.Random(int(seed))

        # Controller state.
        self._B_idx = int(initial_batch_idx)
        self._R_idx = int(initial_replay_idx)
        self._prev_idx: tuple[int, int] = (self._B_idx, self._R_idx)
        self._prev_score: float = math.inf
        self._last_action: str = "accept"

        # Per-episode telemetry accumulators.
        self._ep_wallclock_sum: float = 0.0
        self._ep_flop_sum: float = 0.0
        self._ep_update_calls: int = 0

        self.base = PDMORLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            n_objectives=n_objectives,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            batch_size=self.batch_sizes[self._B_idx],
            device=device,
            **base_kwargs,
        )

    # -- delegated agent surface --------------------------------------------

    def act(self, state: np.ndarray, omega: np.ndarray, explore: bool = True) -> int:
        return int(self.base.act(state, omega, explore=explore))

    def store(self, transition: Transition) -> None:
        self.base.store(transition)

    def update(self) -> dict:
        """Run the base update ``R`` times, tracking wall-clock and FLOPs.

        DuoJoule's replay ratio R means R gradient steps per env step;
        we expose that here so the controller's latency/energy proxies
        actually reflect the chosen knobs.
        """
        R = self.replay_ratios[self._R_idx]
        last_metrics: dict = {}
        for _ in range(R):
            t0 = time.perf_counter()
            metrics = self.base.update()
            elapsed = time.perf_counter() - t0
            self._ep_wallclock_sum += elapsed
            # FLOP proxy: B * hidden_dim^2 per linear layer (n_layers=3).
            self._ep_flop_sum += (
                self.base.batch_size * self.hidden_dim * self.hidden_dim * _FLOP_LAYERS
            )
            self._ep_update_calls += 1
            if metrics:
                last_metrics = metrics
        return last_metrics

    def end_episode(self) -> None:
        """Greedy controller step. Accept / revert / explore."""
        score = self._compute_efficiency_score(self._B_idx, self._R_idx)
        improved = score < self._prev_score

        if improved:
            self._prev_score = score
            self._prev_idx = (self._B_idx, self._R_idx)
            new_idx = self._random_neighbour(self._B_idx, self._R_idx)
            if new_idx == (self._B_idx, self._R_idx):
                self._last_action = "accept"
            else:
                self._B_idx, self._R_idx = new_idx
                self._last_action = "explore"
        else:
            # Revert to last known-good config; keep prev_score as the bar.
            self._B_idx, self._R_idx = self._prev_idx
            self._last_action = "revert"

        self.base.batch_size = self.batch_sizes[self._B_idx]
        self._reset_episode_telemetry()

    # -- controller internals ----------------------------------------------

    def _compute_efficiency_score(self, B_idx: int, R_idx: int) -> float:
        """Lower is better. Combines latency (wall-clock) and energy (FLOP).

        When no updates fired this episode (e.g. buffer not warm yet) we
        fall back to the static FLOP estimate alone so the controller
        still gets a monotonic signal in B and R.
        """
        B = self.batch_sizes[B_idx]
        R = self.replay_ratios[R_idx]
        if self._ep_update_calls > 0:
            mean_wallclock = self._ep_wallclock_sum / self._ep_update_calls
            mean_flops = self._ep_flop_sum / self._ep_update_calls
        else:
            mean_wallclock = 0.0
            mean_flops = float(B * R * self.hidden_dim * self.hidden_dim * _FLOP_LAYERS)
        return self.w_energy * mean_flops + self.w_latency * mean_wallclock

    def _random_neighbour(self, B_idx: int, R_idx: int) -> tuple[int, int]:
        """Pick a random ±1 neighbour on exactly one axis (or stay put)."""
        candidates: list[tuple[int, int]] = []
        if B_idx > 0:
            candidates.append((B_idx - 1, R_idx))
        if B_idx < len(self.batch_sizes) - 1:
            candidates.append((B_idx + 1, R_idx))
        if R_idx > 0:
            candidates.append((B_idx, R_idx - 1))
        if R_idx < len(self.replay_ratios) - 1:
            candidates.append((B_idx, R_idx + 1))
        if not candidates:
            return (B_idx, R_idx)
        return self._rng.choice(candidates)

    def _reset_episode_telemetry(self) -> None:
        self._ep_wallclock_sum = 0.0
        self._ep_flop_sum = 0.0
        self._ep_update_calls = 0

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        import torch

        self.base.save(path)
        torch.save(
            {
                "B_idx": self._B_idx,
                "R_idx": self._R_idx,
                "prev_idx": self._prev_idx,
                "prev_score": self._prev_score,
                "last_action": self._last_action,
                "rng_state": self._rng.getstate(),
            },
            path + ".duojoule",
        )

    def load(self, path: str) -> None:
        import torch

        self.base.load(path)
        ctrl = torch.load(path + ".duojoule", map_location="cpu", weights_only=False)
        self._B_idx = int(ctrl["B_idx"])
        self._R_idx = int(ctrl["R_idx"])
        self._prev_idx = tuple(ctrl["prev_idx"])  # type: ignore[assignment]
        self._prev_score = float(ctrl["prev_score"])
        self._last_action = str(ctrl["last_action"])
        self._rng.setstate(ctrl["rng_state"])
        self.base.batch_size = self.batch_sizes[self._B_idx]

    # -- properties ---------------------------------------------------------

    @property
    def current_batch_size(self) -> int:
        return self.batch_sizes[self._B_idx]

    @property
    def current_replay_ratio(self) -> int:
        return self.replay_ratios[self._R_idx]

    @property
    def step_count(self) -> int:
        return int(self.base.step_count)


class DuoJouleArbiter:
    """Behavioural surrogate of DuoJoule's energy/perf threshold gate.

    Kept for back-compat with the eval runner registry. ``omega[3] > 0.5``
    (energy-dominant) selects the low-energy action 0; otherwise selects
    the high-perf action ``n_actions - 1``. Constructor accepts ``seed``
    for interface uniformity but the policy is deterministic.
    """

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
