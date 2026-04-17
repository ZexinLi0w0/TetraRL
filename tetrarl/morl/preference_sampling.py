"""Preference sampling and HER-style preference relabeling for PD-MORL.

Provides Dirichlet-based preference sampling and hindsight experience
replay (HER) over preference vectors, following Basaklar et al. (ICLR 2023)
and the HER mechanism from Envelope MORL (Yang et al., 2019).
"""

import numpy as np


def sample_preference(
    num_objectives: int, batch_size: int = 1, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Sample preference vectors from a symmetric Dirichlet(1) distribution.

    @param num_objectives  Number of objective dimensions.
    @param batch_size      Number of preference vectors to sample.
    @param rng             Optional numpy random generator for reproducibility.
    @return                Array of shape (batch_size, num_objectives),
                           each row sums to 1.
    """
    if rng is None:
        rng = np.random.default_rng()
    raw = rng.dirichlet(np.ones(num_objectives), size=batch_size)
    return raw.astype(np.float32)


def sample_anchor_preferences(num_objectives: int) -> np.ndarray:
    """Generate deterministic anchor preferences: corners + center.

    @param num_objectives  Number of objective dimensions.
    @return                Array of shape (num_objectives + 1, num_objectives).
    """
    corners = np.eye(num_objectives, dtype=np.float32)
    center = np.ones((1, num_objectives), dtype=np.float32) / num_objectives
    return np.concatenate([corners, center], axis=0)


def her_preference_relabel(
    transitions: list[dict],
    num_objectives: int,
    n_relabel: int = 4,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Augment transitions with HER-style preference relabeling.

    For each original transition, sample n_relabel new preference vectors
    and create copies of the transition with these new preferences.
    The reward vector remains unchanged; only omega is replaced.

    @param transitions     List of transition dicts with keys:
                           state, action, reward_vec, next_state, done, omega.
    @param num_objectives  Number of objective dimensions.
    @param n_relabel       Number of relabeled copies per transition.
    @param rng             Optional numpy random generator.
    @return                Original transitions + relabeled copies.
    """
    if rng is None:
        rng = np.random.default_rng()
    augmented = list(transitions)
    for t in transitions:
        new_omegas = sample_preference(num_objectives, n_relabel, rng=rng)
        for i in range(n_relabel):
            relabeled = dict(t)
            relabeled["omega"] = new_omegas[i]
            augmented.append(relabeled)
    return augmented
