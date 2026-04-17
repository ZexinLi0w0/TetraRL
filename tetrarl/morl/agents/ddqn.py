"""Double DQN (DDQN) — replay-based off-policy.

Reference: Hasselt, Guez, Silver, 'Deep Reinforcement Learning with Double Q-learning',
AAAI 2016.

Paradigm: off-policy replay-based.
Action space: discrete.
R^4 knobs exposed to the runtime coordinator: batch_size, replay_buffer_size, DVFS,
mixed_precision.

Scope within TetraRL: Decouples action selection from value evaluation to mitigate
maximization bias. The MO-DDQN-HER variant (Basaklar 2023, ICLR) is implemented in
pd_morl.py; this file provides the single-objective baseline.

Status: stub (skeleton only).
TODO: full implementation scheduled for Week 7-8 per docs/action-plan-weekly.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DDQNAgentConfig:
    """Configuration for DDQNAgent.

    All R^4 runtime knobs that the resource manager may modify at runtime
    are exposed here. The runtime coordinator updates this configuration
    in-place; the agent re-reads relevant fields on each training step.
    """

    # Algorithm-specific hyperparameters (filled in Week 7-8).
    extra: dict[str, Any] = field(default_factory=dict)


class DDQNAgent:
    """Double DQN (DDQN) — replay-based off-policy.

    This is an interface stub. The training loop and loss formulation will
    be implemented during Week 7-8. The class signature is finalized to allow
    upstream modules (resource manager, eval harness) to import the type.
    """

    _STUB_MSG = "Stub method; full implementation scheduled for Week 7-8."

    def __init__(self, config: "DDQNAgentConfig | None" = None) -> None:
        self.config = config or DDQNAgentConfig()
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
