"""Cosine-similarity envelope operator for multi-objective Q-learning.

Implements the scalarization operator Sc(omega_p, Q) from PD-MORL
(Basaklar et al., ICLR 2023), which uses cosine similarity between the
preference vector and the Q-vector to guide the critic toward
Pareto-optimal solutions.
"""

# TODO: Week 1 — implement cosine_similarity_scalarization and
#       envelope_operator; unit-test in tests/test_operators.py.

import torch


def cosine_similarity_scalarization(
    omega: torch.Tensor, q_values: torch.Tensor
) -> torch.Tensor:
    """Compute cosine-similarity-weighted scalarization.

    @param omega     Preference vector, shape (batch, n_obj).
    @param q_values  Multi-objective Q-values, shape (batch, n_obj).
    @return          Scalarized Q-values, shape (batch,).
    """
    raise NotImplementedError


def envelope_operator(
    omega: torch.Tensor, q_values: torch.Tensor
) -> torch.Tensor:
    """Apply the PD-MORL envelope operator: Sc(omega, Q) * (omega^T Q).

    @param omega     Preference vector, shape (batch, n_obj).
    @param q_values  Multi-objective Q-values, shape (batch, n_obj).
    @return          Scalar envelope value, shape (batch,).
    """
    raise NotImplementedError
