"""Tail-latency CDF analysis utilities.

Provides functions for computing and plotting cumulative distribution
functions of per-step and per-episode latencies, with emphasis on tail
percentiles (95th, 99th, 99.9th) as required by the evaluation norms.
"""

# TODO: Week 7 — implement CDF computation and plotting; integrate with
#       the evaluation harness for automated figure generation.

import numpy as np


def compute_cdf(latencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the empirical CDF of a latency array.

    @param latencies  1-D array of latency measurements.
    @return           Tuple of (sorted_values, cdf_values).
    """
    raise NotImplementedError


def tail_percentiles(
    latencies: np.ndarray,
    percentiles: tuple[float, ...] = (50.0, 95.0, 99.0, 99.9),
) -> dict[float, float]:
    """Compute tail-latency percentiles.

    @param latencies    1-D array of latency measurements.
    @param percentiles  Percentile levels to compute.
    @return             Dict mapping percentile to latency value.
    """
    raise NotImplementedError
