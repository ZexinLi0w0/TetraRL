"""NeuOS LAG metric (Bateni & Liu, RTAS 2020) as an RL state feature.

LAG (Latency Above Ground-truth) is the per-DNN ratio of the current
inference latency to its target latency: ``lag = latency / target``.
NeuOS uses this as an additional state feature for multi-DNN co-running
schedulers; when ``lag > 1`` the corunner is missing its deadline and
the policy should de-prioritize new arrivals.

For TetraRL's single-task case we approximate using the framework's
``telemetry.latency_ema_ms / soft_latency_ms`` ratio; the multi-corunner
case accepts an explicit list of corunner latencies and returns one
ratio per corunner.

This module is a pure-numpy state-feature extractor and does NOT alter
the framework or the arbiter. The training script wires it in via
``extractor.append_to_state(state, telemetry)`` (Task 3 scope).
"""
from __future__ import annotations

import numpy as np

from tetrarl.morl.native.override import HardwareTelemetry


class LAGFeatureExtractor:
    """Extract NeuOS LAG ratios as a fixed-width float32 feature vector."""

    def __init__(
        self,
        soft_latency_ms: float = 50.0,
        n_corunners: int = 1,
        clip_max: float | None = 10.0,
    ):
        if not soft_latency_ms > 0.0:
            raise ValueError(
                f"soft_latency_ms must be > 0, got {soft_latency_ms!r}"
            )
        if n_corunners < 1:
            raise ValueError(f"n_corunners must be >= 1, got {n_corunners!r}")
        self.soft_latency_ms = float(soft_latency_ms)
        self.n_corunners = int(n_corunners)
        self.clip_max = float(clip_max) if clip_max is not None else None

    @property
    def feature_dim(self) -> int:
        """Width of the LAG vector this extractor returns."""
        return self.n_corunners

    def extract(
        self,
        telemetry: HardwareTelemetry,
        corunner_latencies_ms: list[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the per-corunner LAG ratio vector as float32.

        Single-task semantics (``corunner_latencies_ms is None``):
          - LAG = ``telemetry.latency_ema_ms / soft_latency_ms``.
          - If ``telemetry.latency_ema_ms is None`` returns zeros.

        Multi-corunner semantics:
          - Each corunner latency is divided by ``soft_latency_ms``.
          - ``len(corunner_latencies_ms) != n_corunners`` raises ``ValueError``.

        The returned ratios are clipped to ``[0, clip_max]`` when
        ``clip_max`` is not None.
        """
        if corunner_latencies_ms is None:
            lat = telemetry.latency_ema_ms
            if lat is None:
                return np.zeros(self.n_corunners, dtype=np.float32)
            arr = np.full(self.n_corunners, float(lat), dtype=np.float32)
        else:
            arr = np.asarray(corunner_latencies_ms, dtype=np.float32)
            if arr.shape != (self.n_corunners,):
                raise ValueError(
                    f"corunner_latencies_ms length {arr.shape} != n_corunners "
                    f"({self.n_corunners},)"
                )
        ratios = arr / np.float32(self.soft_latency_ms)
        if self.clip_max is not None:
            np.clip(ratios, 0.0, self.clip_max, out=ratios)
        return ratios

    def append_to_state(
        self,
        state: np.ndarray,
        telemetry: HardwareTelemetry,
        corunner_latencies_ms: list[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``concat([state, extract(...)])`` along the last axis."""
        lag = self.extract(telemetry, corunner_latencies_ms)
        base = np.asarray(state, dtype=np.float32)
        return np.concatenate([base, lag], axis=-1)
