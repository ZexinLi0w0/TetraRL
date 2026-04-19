"""TetraRLNativeAgent — high-level wrapper for preference-conditioned PPO.

Mirrors the CMORLAgent API (tetrarl/morl/c_morl_agent.py) for drop-in
comparison between C-MORL (cloud, multi-process) and TetraRL native
(edge, single-process).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from tetrarl.morl.native.preference_ppo import (
    PreferenceNetwork,
    PreferencePPOConfig,
    evaluate_policy,
    train_preference_ppo,
)


@dataclass
class NativeAgentConfig:
    """Configuration for TetraRLNativeAgent."""

    env_name: str = "dst"
    obj_num: int = 2
    ref_point: list[float] = field(
        default_factory=lambda: [0.0, -25.0]
    )
    total_timesteps: int = 100_000
    num_steps: int = 256
    hidden_dim: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    seed: int = 0
    eval_interval: int = 10
    eval_episodes: int = 3
    n_eval_interior: int = 10
    device: str = "cpu"


class TetraRLNativeAgent:
    """Preference-conditioned multi-objective PPO agent.

    Single-process, edge-device-friendly alternative to CMORLAgent.

    Usage::

        agent = TetraRLNativeAgent(
            env_name="dst",
            obj_num=2,
            ref_point=[0.0, -25.0],
        )
        agent.train()
        front = agent.get_pareto_front()
        obj = agent.evaluate(np.array([0.5, 0.5]))
    """

    def __init__(
        self,
        env_name: str = "dst",
        obj_num: int = 2,
        ref_point: list[float] | None = None,
        *,
        device: str = "cpu",
        total_timesteps: int = 100_000,
        num_steps: int = 256,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        seed: int = 0,
        eval_interval: int = 10,
        eval_episodes: int = 3,
        n_eval_interior: int = 10,
        **kwargs: Any,
    ) -> None:
        if ref_point is None:
            ref_point = [0.0] * obj_num

        self.config = NativeAgentConfig(
            env_name=env_name,
            obj_num=obj_num,
            ref_point=ref_point,
            total_timesteps=total_timesteps,
            num_steps=num_steps,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            seed=seed,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            n_eval_interior=n_eval_interior,
            device=device,
        )
        self._network: PreferenceNetwork | None = None
        self._pareto_front: np.ndarray | None = None
        self._results: dict[str, Any] | None = None
        self._env_fn = self._make_env_fn()

    def _make_env_fn(self) -> Any:
        env_name = self.config.env_name
        if env_name == "dst":
            from tetrarl.envs.dst import DeepSeaTreasure

            return lambda: DeepSeaTreasure()
        else:
            import mo_gymnasium

            return lambda: mo_gymnasium.make(env_name)

    def train(
        self, verbose: bool = True, **overrides: Any
    ) -> dict[str, Any]:
        """Run full preference-conditioned PPO training."""
        ppo_config = PreferencePPOConfig(
            n_objectives=self.config.obj_num,
            total_timesteps=self.config.total_timesteps,
            num_steps=self.config.num_steps,
            hidden_dim=self.config.hidden_dim,
            lr=self.config.lr,
            gamma=self.config.gamma,
            seed=self.config.seed,
            eval_interval=self.config.eval_interval,
            eval_episodes=self.config.eval_episodes,
            n_eval_interior=self.config.n_eval_interior,
            ref_point=self.config.ref_point,
        )
        for k, v in overrides.items():
            if hasattr(ppo_config, k):
                setattr(ppo_config, k, v)

        results = train_preference_ppo(
            ppo_config,
            self._env_fn,
            device=self.config.device,
            verbose=verbose,
        )

        self._network = results["network"]
        self._pareto_front = results["pareto_front"]
        self._results = results
        return results

    def get_pareto_front(self) -> dict[str, Any]:
        """Return discovered Pareto front and hypervolume metrics."""
        if self._pareto_front is None or self._results is None:
            raise RuntimeError(
                "No Pareto front available. Run train() first."
            )
        return {
            "objectives": self._pareto_front,
            "hv": self._results["best_hv"],
            "hv_history": self._results["hv_history"],
        }

    def evaluate(
        self,
        preference_vector: np.ndarray,
        n_episodes: int = 5,
    ) -> np.ndarray:
        """Evaluate the trained policy at a specific preference."""
        if self._network is None:
            raise RuntimeError(
                "No trained network. Run train() first."
            )
        env = self._env_fn()
        result = evaluate_policy(
            self._network,
            env,
            preference_vector,
            n_episodes=n_episodes,
            device=self.config.device,
        )
        env.close()
        return result

    def save(self, path: str | Path) -> None:
        """Persist trained network and Pareto front to disk."""
        if self._network is None:
            raise RuntimeError(
                "No trained network. Run train() first."
            )
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._network.state_dict(), path / "network.pt")
        if self._pareto_front is not None:
            np.savetxt(
                path / "pareto_front.csv",
                self._pareto_front,
                delimiter=",",
            )

    def load(self, path: str | Path) -> None:
        """Restore network and Pareto front from disk."""
        path = Path(path)
        env = self._env_fn()
        obs_dim = int(np.prod(env.observation_space.shape))
        continuous = isinstance(env.action_space, gym.spaces.Box)
        act_dim = (
            env.action_space.shape[0]
            if continuous
            else env.action_space.n
        )
        env.close()

        self._network = PreferenceNetwork(
            obs_dim,
            act_dim,
            self.config.obj_num,
            self.config.hidden_dim,
            continuous,
        ).to(self.config.device)
        self._network.load_state_dict(
            torch.load(
                path / "network.pt",
                map_location=self.config.device,
                weights_only=True,
            )
        )
        pf_path = path / "pareto_front.csv"
        if pf_path.exists():
            self._pareto_front = np.loadtxt(pf_path, delimiter=",")
