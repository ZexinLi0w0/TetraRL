"""Hypervolume indicator computation for multi-objective evaluation.

Computes the dominated hypervolume of a Pareto front approximation
relative to a reference point. Used as the primary aggregated metric
for comparing multi-objective methods across the R-four dimensions.
"""

import numpy as np


def pareto_filter(points: np.ndarray) -> np.ndarray:
    """Filter a set of points to the non-dominated (Pareto) subset.

    Assumes maximization on all objectives.

    @param points  Array of shape (n_points, n_objectives).
    @return        Non-dominated subset, shape (n_pareto, n_objectives).
    """
    if len(points) == 0:
        return points
    is_pareto = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if not is_pareto[i]:
            continue
        for j in range(i + 1, len(points)):
            if not is_pareto[j]:
                continue
            if np.all(points[j] >= p) and np.any(points[j] > p):
                is_pareto[i] = False
                break
            if np.all(p >= points[j]) and np.any(p > points[j]):
                is_pareto[j] = False
    return points[is_pareto]


def hypervolume(pareto_front: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute the hypervolume indicator.

    For 2-D objectives, uses an efficient sweep-line algorithm.
    For higher dimensions, falls back to an inclusion-exclusion method.

    Assumes maximization: points should dominate the reference point.

    @param pareto_front  Array of shape (n_points, n_objectives).
    @param ref_point     Reference point, shape (n_objectives,).
    @return              Dominated hypervolume (scalar).
    """
    if len(pareto_front) == 0:
        return 0.0
    ref_point = np.asarray(ref_point, dtype=np.float64)
    front = np.asarray(pareto_front, dtype=np.float64)

    dominated = np.all(front > ref_point, axis=1)
    front = front[dominated]
    if len(front) == 0:
        return 0.0

    n_obj = front.shape[1]
    if n_obj == 2:
        return _hv_2d(front, ref_point)
    return _hv_nd(front, ref_point)


def _hv_2d(front: np.ndarray, ref: np.ndarray) -> float:
    """Sweep-line hypervolume for 2 objectives (maximization)."""
    sorted_front = front[front[:, 0].argsort()[::-1]]
    hv = 0.0
    y_bound = ref[1]
    for point in sorted_front:
        if point[1] > y_bound:
            hv += (point[0] - ref[0]) * (point[1] - y_bound)
            y_bound = point[1]
    return hv


def _hv_nd(front: np.ndarray, ref: np.ndarray) -> float:
    """Recursive inclusion-exclusion hypervolume for N-D (small N)."""
    if front.shape[1] == 1:
        return float(np.max(front[:, 0]) - ref[0])
    if len(front) == 1:
        return float(np.prod(front[0] - ref))

    sorted_front = front[front[:, 0].argsort()[::-1]]
    hv = 0.0
    for i, point in enumerate(sorted_front):
        x_width = point[0] - (
            sorted_front[i + 1, 0] if i + 1 < len(sorted_front) else ref[0]
        )
        remaining = sorted_front[: i + 1, 1:]
        remaining_ref = ref[1:]
        dominated_remaining = np.all(remaining > remaining_ref, axis=1)
        remaining = remaining[dominated_remaining]
        if len(remaining) > 0:
            slice_hv = (
                _hv_nd(pareto_filter(remaining), remaining_ref)
                if remaining.shape[1] > 1
                else float(np.max(remaining[:, 0]) - remaining_ref[0])
            )
            hv += x_width * slice_hv
    return hv
