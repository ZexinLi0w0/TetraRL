"""Multi-Objective SAC with HER preference replay (MO-SAC-HER).

Ports the PD-MORL preference-directed framework from DQN to SAC for
continuous action spaces. Combines entropy-regularized policy optimization
with cosine-similarity envelope critic loss and directional regularization.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from tetrarl.morl.loss import (
    cosine_similarity_envelope_loss,
    directional_regularization,
)
from tetrarl.morl.operators import cosine_similarity_scalarization

LOG_STD_MIN = -20
LOG_STD_MAX = 2
ALPHA_MIN = 0.01
ALPHA_MAX = 1.0


@dataclass
class MOTransition:
    state: np.ndarray
    action: np.ndarray
    reward_vec: np.ndarray
    next_state: np.ndarray
    done: bool
    omega: np.ndarray


class MOReplayBuffer:
    def __init__(self, capacity: int = 1_000_000):
        self._buf: deque[MOTransition] = deque(maxlen=capacity)

    def push(self, transition: MOTransition) -> None:
        self._buf.append(transition)

    def sample(self, batch_size: int) -> list[MOTransition]:
        return random.sample(list(self._buf), min(batch_size, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


class MOGaussianPolicy(nn.Module):
    """Preference-conditioned Gaussian policy: [s, omega] -> action distribution."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_objectives: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = state_dim + n_objectives
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, state: torch.Tensor, omega: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, omega], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, state: torch.Tensor, omega: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state, omega)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(state, omega)
        return torch.tanh(mean)


class MOTwinQNetwork(nn.Module):
    """Preference-conditioned twin Q-network outputting Q-vectors."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_objectives: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = state_dim + action_dim + n_objectives
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objectives),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objectives),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, omega: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action, omega], dim=-1)
        return self.q1(x), self.q2(x)


class MOSACHERAgent:
    """Multi-Objective SAC with Hindsight Experience Replay over preferences."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_objectives: int = 2,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_lr: float = 3e-4,
        batch_size: int = 256,
        buffer_capacity: int = 1_000_000,
        initial_alpha: float = 0.2,
        n_relabel: int = 4,
        dir_reg_coeff: float = 0.1,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_relabel = n_relabel
        self.dir_reg_coeff = dir_reg_coeff
        self.device = torch.device(device)

        self.policy = MOGaussianPolicy(
            state_dim, action_dim, n_objectives, hidden_dim
        ).to(self.device)
        self.critic = MOTwinQNetwork(
            state_dim, action_dim, n_objectives, hidden_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(
            np.log(initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.buffer = MOReplayBuffer(buffer_capacity)
        self.step_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(ALPHA_MIN, ALPHA_MAX)

    def act(
        self,
        state: np.ndarray,
        omega: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        with torch.no_grad():
            s = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            w = torch.tensor(
                omega, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            if deterministic:
                action = self.policy.deterministic(s, w)
            else:
                action, _ = self.policy.sample(s, w)
            return action.cpu().numpy().squeeze(0)

    def store(self, transition: MOTransition) -> None:
        self.buffer.push(transition)
        for _ in range(self.n_relabel):
            new_omega = np.random.dirichlet(
                np.ones(self.n_objectives)
            ).astype(np.float32)
            relabeled = MOTransition(
                state=transition.state,
                action=transition.action,
                reward_vec=transition.reward_vec,
                next_state=transition.next_state,
                done=transition.done,
                omega=new_omega,
            )
            self.buffer.push(relabeled)

    def _scalarize_q(self, omega: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Scalarize Q-vector via cosine-similarity envelope for min selection."""
        sc = cosine_similarity_scalarization(omega, q)
        linear = (omega * q).sum(dim=-1)
        return sc * linear

    def update(self) -> dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size)
        dev = self.device
        states = torch.tensor(
            np.array([t.state for t in batch]),
            dtype=torch.float32, device=dev,
        )
        actions = torch.tensor(
            np.array([t.action for t in batch]),
            dtype=torch.float32, device=dev,
        )
        rewards = torch.tensor(
            np.array([t.reward_vec for t in batch]),
            dtype=torch.float32, device=dev,
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]),
            dtype=torch.float32, device=dev,
        )
        dones = torch.tensor(
            [float(t.done) for t in batch],
            dtype=torch.float32, device=dev,
        )
        omegas = torch.tensor(
            np.array([t.omega for t in batch]),
            dtype=torch.float32, device=dev,
        )

        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states, omegas)
            q1_next, q2_next = self.critic_target(next_states, next_actions, omegas)
            # Min over twin critics using scalarized values for selection
            scalar_q1 = self._scalarize_q(omegas, q1_next)
            scalar_q2 = self._scalarize_q(omegas, q2_next)
            use_q1 = (scalar_q1 <= scalar_q2).unsqueeze(-1).float()
            q_next = use_q1 * q1_next + (1 - use_q1) * q2_next
            # Entropy bonus subtracted as scalar, broadcast across objectives
            target_q = rewards + self.gamma * (1 - dones.unsqueeze(-1)) * (
                q_next - self.alpha.detach() * next_log_probs
            )

        q1, q2 = self.critic(states, actions, omegas)
        mse_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        envelope_loss = (
            cosine_similarity_envelope_loss(omegas, q1, target_q)
            + cosine_similarity_envelope_loss(omegas, q2, target_q)
        )
        critic_loss = mse_loss + envelope_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        # --- Actor update ---
        new_actions, log_probs = self.policy.sample(states, omegas)
        q1_new, q2_new = self.critic(states, new_actions, omegas)
        scalar_q1_new = self._scalarize_q(omegas, q1_new)
        scalar_q2_new = self._scalarize_q(omegas, q2_new)
        q_scalar_new = torch.min(scalar_q1_new, scalar_q2_new)

        # Use q1_new for directional regularization
        dir_reg = directional_regularization(omegas, q1_new)
        actor_loss = (self.alpha.detach() * log_probs.squeeze(-1) - q_scalar_new).mean()
        actor_loss = actor_loss + self.dir_reg_coeff * dir_reg

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.policy_optimizer.step()

        # --- Alpha update ---
        alpha_loss = -(
            self.log_alpha
            * (log_probs.detach().squeeze(-1) + self.target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Clamp log_alpha to enforce alpha bounds
        with torch.no_grad():
            self.log_alpha.clamp_(np.log(ALPHA_MIN), np.log(ALPHA_MAX))

        # --- Soft target update ---
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self.step_count += 1
        return {
            "critic_loss": critic_loss.item(),
            "mse_loss": mse_loss.item(),
            "envelope_loss": envelope_loss.item(),
            "actor_loss": actor_loss.item(),
            "dir_reg": dir_reg.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "step_count": self.step_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(True)
        lr = self.alpha_optimizer.defaults["lr"]
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=lr
        )
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        self.step_count = ckpt["step_count"]
