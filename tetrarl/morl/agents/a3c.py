"""Asynchronous Advantage Actor-Critic (A3C) — on-policy non-replay (async).

Reference: Mnih et al., 'Asynchronous Methods for Deep Reinforcement Learning', ICML
2016.

Paradigm: on-policy non-replay (async).
Action space: discrete or continuous.
R^4 knobs exposed to the runtime coordinator: n_steps, n_envs, n_workers, DVFS,
mixed_precision.

Scope within TetraRL: Asynchronous variant introducing the n_workers knob. Useful for
stress-testing the Resource Primitives mapper under inter-worker contention on Jetson
devices.

Status: stub (skeleton only).
TODO: full implementation scheduled for Week 7-8 per docs/action-plan-weekly.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class A3CAgentConfig:
    """Configuration for A3CAgent.

    All R^4 runtime knobs that the resource manager may modify at runtime
    are exposed here. The runtime coordinator updates this configuration
    in-place; the agent re-reads relevant fields on each training step.
    """

    # Algorithm-specific hyperparameters (filled in Week 7-8).
    extra: dict[str, Any] = field(default_factory=dict)


class A3CAgent:
    """Asynchronous Advantage Actor-Critic (A3C) — on-policy non-replay (async).

    This is an interface stub. The training loop and loss formulation will
    be implemented during Week 7-8. The class signature is finalized to allow
    upstream modules (resource manager, eval harness) to import the type.
    """

    _STUB_MSG = "Stub method; full implementation scheduled for Week 7-8."

    def __init__(self, config: "A3CAgentConfig | None" = None) -> None:
        self.config = config or A3CAgentConfig()
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
