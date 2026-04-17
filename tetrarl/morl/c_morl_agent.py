"""Clean wrapper around vendored C-MORL for TetraRL integration.

Provides CMORLAgent — a thin API over the two-stage Pareto front
discovery algorithm from 'Efficient Discovery of Pareto Front for
Multi-Objective Reinforcement Learning' (ICLR 2025).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


_CMORL_DIR = Path(__file__).resolve().parent / "c_morl"


def _ensure_cmorl_paths() -> None:
    """Add vendored C-MORL directories to sys.path (idempotent)."""
    paths = [
        str(_CMORL_DIR),
        str(_CMORL_DIR / "externals" / "baselines"),
        str(_CMORL_DIR / "externals" / "pytorch-a2c-ppo-acktr-gail"),
    ]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


@dataclass
class CMORLConfig:
    """Configuration mirroring C-MORL's argparse defaults."""

    env_name: str = "MO-HalfCheetah-v2"
    obj_num: int = 2
    ref_point: list[float] = field(default_factory=lambda: [0.0, 0.0])
    num_time_steps: int = 2_500_000
    num_init_steps: int = 1_500_000
    seed: int = 0

    # Weight grid
    min_weight: float = 0.0
    max_weight: float = 1.0
    delta_weight: float = 0.2
    eval_delta_weight: float = 0.01

    # EP / selection
    num_select: int = 5
    policy_buffer: int = 200

    # Extension stage
    update_iter: int = 20
    beta: float = 0.9
    t: float = 20.0
    update_method: str = "cmorl-ipo"

    # PPO hyper-parameters
    algo: str = "ppo"
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    entropy_coef: float = 0.0
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_steps: int = 2048
    num_processes: int = 4
    ppo_epoch: int = 10
    num_mini_batch: int = 32
    clip_param: float = 0.2

    # Evaluation
    eval_num: int = 10
    eval_gamma: float = 0.99
    rl_eval_interval: int = 10
    rl_log_interval: int = 10

    # Flags
    use_linear_lr_decay: bool = False
    lr_decay_ratio: float = 1.0
    use_gae: bool = False
    use_proper_time_limits: bool = False
    ob_rms: bool = False
    obj_rms: bool = False
    raw: bool = False
    layernorm: bool = False
    cost_objective: bool = False

    # Misc
    num_tasks: int = 6
    save_dir: str = "./trained_models/"
    selection_method: str = "prediction-guided"
    pbuffer_num: int = 200
    pbuffer_size: int = 2
    num_weight_candidates: int = 7
    sparsity: float = 1.0
    warmup_iter: int = 80
    num_env_steps: float = 5e6

    def to_namespace(self) -> Any:
        """Convert to argparse.Namespace for C-MORL consumption."""
        import argparse
        return argparse.Namespace(**self.__dict__)


class CMORLAgent:
    """Thin wrapper around vendored C-MORL two-stage Pareto discovery.

    Usage::

        agent = CMORLAgent(
            env_name="building_3d",
            obj_num=3,
            ref_point=[0.0, 0.0, 0.0],
        )
        agent.train()
        front = agent.get_pareto_front()
    """

    def __init__(
        self,
        env_name: str,
        obj_num: int,
        ref_point: list[float],
        *,
        num_time_steps: int = 2_500_000,
        num_init_steps: int = 1_500_000,
        save_dir: str = "./trained_models/",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.config = CMORLConfig(
            env_name=env_name,
            obj_num=obj_num,
            ref_point=ref_point,
            num_time_steps=num_time_steps,
            num_init_steps=num_init_steps,
            save_dir=save_dir,
            seed=seed,
            **kwargs,
        )
        self._ep = None
        self._trained = False

    def train(self, **overrides: Any) -> None:
        """Run the full two-stage C-MORL training pipeline.

        Stage 1 (Init): Parallel policy training with weighted scalarization.
        Stage 2 (Extension): CPO/IPO constrained optimization to fill
                             Pareto front gaps.
        """
        _ensure_cmorl_paths()
        import morl as _cmorl_module

        cfg = CMORLConfig(**{**self.config.__dict__, **overrides})
        args = cfg.to_namespace()
        os.makedirs(args.save_dir, exist_ok=True)

        _cmorl_module.run(args)
        self._trained = True

    def get_pareto_front(self) -> dict[str, np.ndarray]:
        """Return discovered Pareto front objectives from the last run.

        Returns a dict with key ``"objectives"`` (N x obj_num array).
        Reads from the saved results directory.
        """
        objs_path = os.path.join(self.config.save_dir, "final", "objs.txt")
        if not os.path.exists(objs_path):
            raise FileNotFoundError(
                f"No Pareto front results found at {objs_path}. "
                "Run train() first."
            )
        objectives = np.loadtxt(objs_path, delimiter=",")
        return {"objectives": objectives}

    def evaluate(self, preference_vector: np.ndarray) -> np.ndarray:
        """Evaluate the Pareto front under a given preference vector.

        Returns the objective values of the policy whose weighted-sum
        utility is highest for the given preference.
        """
        front = self.get_pareto_front()["objectives"]
        utilities = front @ preference_vector
        best_idx = np.argmax(utilities)
        return front[best_idx]

    @staticmethod
    def vendored_path() -> Path:
        """Return the path to the vendored C-MORL source."""
        return _CMORL_DIR
