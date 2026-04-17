"""Proximal Policy Optimization (PPO) — on-policy non-replay.

Reference: Schulman et al., 'Proximal Policy Optimization Algorithms', arXiv:1707.06347,
2017.

Paradigm: on-policy non-replay.
Action space: discrete or continuous.
R^4 knobs exposed to the runtime coordinator: n_steps, n_envs, n_epochs,
mini_batch_size, KL_coef, DVFS, mixed_precision, gradient_checkpointing.

Scope within TetraRL: Primary on-policy candidate for the non-replay extension. The
Lagrangian PPO variant (Spoor 2025) integrates soft constraints on memory and energy;
PI-controller dual variable update with K_P=K_I=1e-4, K_D=0, anti-windup.

Status: stub (skeleton only).
TODO: full implementation scheduled for Week 4-6 (Lagrangian PPO) and Week 7-8 (full
integration) per docs/action-plan-weekly.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PPOAgentConfig:
    """Configuration for PPOAgent.

    All R^4 runtime knobs that the resource manager may modify at runtime
    are exposed here. The runtime coordinator updates this configuration
    in-place; the agent re-reads relevant fields on each training step.
    """

    # Algorithm-specific hyperparameters (Week 4-6 / Week 7-8).
    extra: dict[str, Any] = field(default_factory=dict)


class PPOAgent:
    """Proximal Policy Optimization (PPO) — on-policy non-replay.

    This is an interface stub. The training loop and loss formulation will
    be implemented during Week 4-6 (Lagrangian PPO) and Week 7-8 (full integration). The
    class signature is finalized to allow
    upstream modules (resource manager, eval harness) to import the type.
    """

    _STUB_MSG = (
        "Stub method; full implementation scheduled for "
        "Week 4-6 (Lagrangian PPO) and Week 7-8 (full integration)."
    )

    def __init__(self, config: "PPOAgentConfig | None" = None) -> None:
        self.config = config or PPOAgentConfig()
        self._initialized = False

    def act(self, observation: Any, *, deterministic: bool = False) -> Any:
        """Select an action given an observation. Stub — see class docstring."""
        raise NotImplementedError(self._STUB_MSG)

    def update(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Run one optimization step. Stub — see class docstring."""
        raise NotImplementedError(self._STUB_MSG)

    def save(self, path: str) -> None:
        """Persist agent state. Stub — see class docstring."""
        raise NotImplementedError(self._STUB_MSG)

    def load(self, path: str) -> None:
        """Restore agent state. Stub — see class docstring."""
        raise NotImplementedError(self._STUB_MSG)
