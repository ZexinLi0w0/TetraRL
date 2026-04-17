"""Preference-Directed Multi-Objective RL (PD-MORL) agent.

Implements the MO-DQN-HER variant for discrete action spaces following
Basaklar et al. (ICLR 2023).  The agent receives a preference vector
omega and optimizes the cosine-similarity-weighted multi-objective
Q-function via Hindsight Experience Replay over preferences.
"""

# TODO: Week 1 — implement MO-DQN-HER with cosine-similarity envelope
#       operator; validate on Deep Sea Treasure (HV >= 229).

import torch
import torch.nn as nn


class PDMORLAgent:
    """Preference-conditioned multi-objective DQN agent with HER."""

    def __init__(self, state_dim: int, action_dim: int, n_objectives: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives

    def act(self, state: torch.Tensor, omega: torch.Tensor) -> int:
        """Select an action given state and preference vector.

        @param state  Current environment observation.
        @param omega  Preference vector in the (n_objectives-1)-simplex.
        @return       Discrete action index.
        """
        raise NotImplementedError

    def update(self, batch: dict) -> dict:
        """Perform a single gradient update on a minibatch.

        @param batch  Dictionary with keys: state, action, reward_vec,
                      next_state, done, omega.
        @return       Dictionary of training metrics.
        """
        raise NotImplementedError
