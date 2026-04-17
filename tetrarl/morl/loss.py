"""Loss functions for preference-directed multi-objective RL.

Extracts the cosine-similarity envelope critic loss and the directional
regularization term from the PD-MORL formulation (Basaklar et al.,
ICLR 2023) into reusable components for both DQN and SAC backbones.
"""

from __future__ import annotations

import torch

from tetrarl.morl.operators import cosine_similarity_scalarization


def cosine_similarity_envelope_loss(
    omega: torch.Tensor,
    q: torch.Tensor,
    target_q: torch.Tensor,
) -> torch.Tensor:
    """Cosine-similarity envelope critic loss.

    L_env = E[ (Sc(omega, target_Q) * (omega^T target_Q)
                - Sc(omega, Q) * (omega^T Q))^2 ]

    @param omega     Preference vector, shape (batch, n_obj).
    @param q         Predicted Q-vector, shape (batch, n_obj).
    @param target_q  Target Q-vector (detached), shape (batch, n_obj).
    @return          Scalar loss.
    """
    sc_q = cosine_similarity_scalarization(omega, q)
    linear_q = (omega * q).sum(dim=-1)
    env_q = sc_q * linear_q

    sc_tgt = cosine_similarity_scalarization(omega, target_q)
    linear_tgt = (omega * target_q).sum(dim=-1)
    env_tgt = sc_tgt * linear_tgt

    return ((env_tgt.detach() - env_q) ** 2).mean()


def directional_regularization(
    omega: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Directional regularization: g(omega, Q) = 1 - Sc(omega, Q).

    Penalizes misalignment between the preference vector and the
    Q-vector, encouraging the critic to produce Q-values whose
    direction matches the requested preference.

    @param omega  Preference vector, shape (batch, n_obj).
    @param q      Q-vector, shape (batch, n_obj).
    @param eps    Numerical stability constant.
    @return       Scalar regularization loss (mean 1 - Sc).
    """
    sc = cosine_similarity_scalarization(omega, q, eps=eps)
    return (1.0 - sc).mean()
