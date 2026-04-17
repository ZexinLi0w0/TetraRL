"""Preference-Directed Multi-Objective RL (PD-MORL) agent.

Implements the MO-DDQN-HER variant for discrete action spaces following
Basaklar et al. (ICLR 2023). The agent receives a preference vector
omega and optimizes the cosine-similarity-weighted multi-objective
Q-function via Hindsight Experience Replay over preferences.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tetrarl.morl.operators import action_selection, envelope_operator


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward_vec: np.ndarray
    next_state: np.ndarray
    done: bool
    omega: np.ndarray


class MOQNetwork(nn.Module):
    """Multi-objective Q-network: [state, omega] -> Q(s, *, omega).

    Output shape: (batch, n_actions, n_objectives).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_objectives: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_objectives, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * n_objectives),
        )

    def forward(self, state: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, omega], dim=-1)
        out = self.net(x)
        return out.view(-1, self.action_dim, self.n_objectives)


class ReplayBuffer:
    """Simple replay buffer for multi-objective transitions."""

    def __init__(self, capacity: int = 100_000):
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(list(self._buffer), min(batch_size, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)


class PDMORLAgent:
    """Preference-conditioned multi-objective DDQN agent with HER."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_objectives: int = 2,
        hidden_dim: int = 256,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        buffer_capacity: int = 100_000,
        n_relabel: int = 4,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_relabel = n_relabel
        self.device = torch.device(device)

        self.q_net = MOQNetwork(state_dim, action_dim, n_objectives, hidden_dim).to(
            self.device
        )
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.step_count = 0

    def _epsilon(self) -> float:
        progress = min(1.0, self.step_count / self.epsilon_decay)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def act(self, state: np.ndarray, omega: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self._epsilon():
            return random.randrange(self.action_dim)

        with torch.no_grad():
            s = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            w = torch.tensor(
                omega, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q = self.q_net(s, w)
            return action_selection(w, q).item()

    def store(self, transition: Transition) -> None:
        self.buffer.push(transition)
        for _ in range(self.n_relabel):
            new_omega = np.random.dirichlet(np.ones(self.n_objectives))
            relabeled = Transition(
                state=transition.state,
                action=transition.action,
                reward_vec=transition.reward_vec,
                next_state=transition.next_state,
                done=transition.done,
                omega=new_omega.astype(np.float32),
            )
            self.buffer.push(relabeled)

    def update(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size)
        states = torch.tensor(
            np.array([t.state for t in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            np.array([t.reward_vec for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [float(t.done) for t in batch], dtype=torch.float32, device=self.device
        )
        omegas = torch.tensor(
            np.array([t.omega for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )

        q_all = self.q_net(states, omegas)
        q_taken = q_all[torch.arange(self.batch_size, device=self.device), actions]

        with torch.no_grad():
            q_next_online = self.q_net(next_states, omegas)
            next_actions = action_selection(omegas, q_next_online)
            q_next_target = self.target_net(next_states, omegas)
            q_next_selected = q_next_target[
                torch.arange(self.batch_size, device=self.device), next_actions
            ]
            targets = rewards + self.gamma * (1 - dones.unsqueeze(-1)) * q_next_selected

        td_error = targets - q_taken
        mse_loss = (td_error ** 2).mean()

        env_op = envelope_operator(omegas, q_taken)
        env_op_target = envelope_operator(omegas, targets)
        envelope_loss = ((env_op_target.detach() - env_op) ** 2).mean()

        loss = mse_loss + envelope_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {
            "loss": loss.item(),
            "mse_loss": mse_loss.item(),
            "envelope_loss": envelope_loss.item(),
            "epsilon": self._epsilon(),
            "buffer_size": len(self.buffer),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = ckpt["step_count"]
