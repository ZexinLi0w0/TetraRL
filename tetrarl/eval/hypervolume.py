"""Hypervolume indicator computation for multi-objective evaluation.

Computes the dominated hypervolume of a Pareto front approximation
relative to a reference point.  Used as the primary aggregated metric
for comparing multi-objective methods across the R-four dimensions.
"""

# TODO: Week 1 — implement exact HV for <= 4 objectives (WFG algorithm
#       or scipy-based); validate against known DST Pareto front.

import numpy as np


def hypervolume(pareto_front: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute the hypervolume indicator.

    @param pareto_front  Array of shape (n_points, n_objectives).
    @param ref_point     Reference point, shape (n_objectives,).
    @return              Dominated hypervolume (scalar).
    """
    raise NotImplementedError


def pareto_filter(points: np.ndarray) -> np.ndarray:
    """Filter a set of points to the non-dominated (Pareto) subset.

    @param points  Array of shape (n_points, n_objectives), maximization.
    @return        Non-dominated subset, shape (n_pareto, n_objectives).
    """
    raise NotImplementedError
