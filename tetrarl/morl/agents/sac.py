"""Soft Actor-Critic (SAC) — replay-based off-policy actor-critic.

Reference: Haarnoja et al., 'Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with
a Stochastic Actor', ICML 2018.

Paradigm: off-policy replay-based (actor-critic).
Action space: continuous.
R^4 knobs exposed to the runtime coordinator: batch_size, replay_buffer_size, DVFS,
mixed_precision, target_entropy.

Scope within TetraRL: Continuous-control baseline new to TetraRL coverage (R^3/DuoJoule
supported only discrete agents). Will be lifted to MO-SAC-HER following PD-MORL
methodology in MuJoCo evaluations (Week 5+).

Status: stub (skeleton only).
TODO: full implementation scheduled for Week 4-6 per docs/action-plan-weekly.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SACAgentConfig:
    """Configuration for SACAgent.

    All R^4 runtime knobs that the resource manager may modify at runtime
    are exposed here. The runtime coordinator updates this configuration
    in-place; the agent re-reads relevant fields on each training step.
    """

    # Algorithm-specific hyperparameters (filled in Week 4-6).
    extra: dict[str, Any] = field(default_factory=dict)


class SACAgent:
    """Soft Actor-Critic (SAC) — replay-based off-policy actor-critic.

    This is an interface stub. The training loop and loss formulation will
    be implemented during Week 4-6. The class signature is finalized to allow
    upstream modules (resource manager, eval harness) to import the type.
    """

    _STUB_MSG = "Stub method; full implementation scheduled for Week 4-6."

    def __init__(self, config: "SACAgentConfig | None" = None) -> None:
        self.config = config or SACAgentConfig()
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
