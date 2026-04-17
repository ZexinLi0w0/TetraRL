"""Cosine-similarity envelope operator for multi-objective Q-learning.

Implements the scalarization operator Sc(omega, Q) from PD-MORL
(Basaklar et al., ICLR 2023), which uses cosine similarity between the
preference vector and the Q-vector to guide the critic toward
Pareto-optimal solutions.
"""

import torch


def cosine_similarity_scalarization(
    omega: torch.Tensor, q_values: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Compute cosine similarity between preference and Q-vectors.

    Sc(omega, Q) = (omega . Q) / (||omega|| * ||Q||)

    @param omega     Preference vector, shape (batch, n_obj).
    @param q_values  Multi-objective Q-values, shape (batch, n_obj).
    @param eps       Small constant to avoid division by zero.
    @return          Cosine similarity, shape (batch,).
    """
    dot = (omega * q_values).sum(dim=-1)
    norm_omega = omega.norm(dim=-1).clamp(min=eps)
    norm_q = q_values.norm(dim=-1).clamp(min=eps)
    return dot / (norm_omega * norm_q)


def envelope_operator(
    omega: torch.Tensor, q_values: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Apply the PD-MORL envelope operator: Sc(omega, Q) * (omega^T Q).

    @param omega     Preference vector, shape (batch, n_obj).
    @param q_values  Multi-objective Q-values, shape (batch, n_obj).
    @param eps       Small constant for numerical stability.
    @return          Scalar envelope value, shape (batch,).
    """
    sc = cosine_similarity_scalarization(omega, q_values, eps=eps)
    linear = (omega * q_values).sum(dim=-1)
    return sc * linear


def action_selection(
    omega: torch.Tensor, q_all_actions: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Select actions by maximizing the envelope operator over actions.

    @param omega          Preference vector, shape (batch, n_obj).
    @param q_all_actions  Q-values for all actions, shape (batch, n_actions, n_obj).
    @param eps            Small constant for numerical stability.
    @return               Selected action indices, shape (batch,).
    """
    batch, n_actions, n_obj = q_all_actions.shape
    omega_expanded = omega.unsqueeze(1).expand_as(q_all_actions)
    omega_flat = omega_expanded.reshape(batch * n_actions, n_obj)
    q_flat = q_all_actions.reshape(batch * n_actions, n_obj)
    scores = envelope_operator(omega_flat, q_flat, eps=eps)
    scores = scores.view(batch, n_actions)
    return scores.argmax(dim=-1)
