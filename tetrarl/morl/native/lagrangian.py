"""PPO-Lagrangian for TetraRL — constrained-RL training driver.

Per-step reward is shaped by Lagrangian multipliers that are updated with
the Spoor (2025) PI dual rule:

    error_i        = max(0, measured_i - target_i)              # one-sided
    integral_i    += error_i        # clamped to +/- integral_max (anti-windup I)
    delta_i        = Kp * error_i + Ki * integral_i
    lambda_i       = clip(lambda_i + delta_i, lambda_min, lambda_max)  # anti-windup II

The shaped scalar reward fed into the rollout buffer is

    r_shaped = r_scalar - sum(lambda_i * violation_i)

We keep the override layer decoupled from the policy gradient (same pattern
as preference_ppo.py): when the override fires we substitute the env-step
action with a safe fallback, but the policy's own action+logprob is what
gets stored in the rollout buffer and trained on.

Currently only the wrapper is implemented for the `knob_mapper` argument
(closed-loop coupling between lambdas and PPO knobs is deferred to a later
week's deliverable; see docs/week7_ppo_lagrangian_design.md).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from tetrarl.morl.native.override import (
    HardwareTelemetry,
    OverrideLayer,
)

# ---------------------------------------------------------------------------
# Lagrangian multipliers
# ---------------------------------------------------------------------------


@dataclass
class LagrangianConfig:
    """Hyperparameters for the Lagrangian dual update.

    Fields
    ------
    n_constraints : number of constraints (default 3 = latency, energy, memory)
    targets       : threshold value per constraint (i.e. the upper bound the
                    measured value must stay below)
    init_lambdas  : initial multiplier values; defaults to zeros
    lambda_lr_p   : Kp gain in the Spoor PI dual update
    lambda_lr_i   : Ki gain in the Spoor PI dual update
    lambda_max    : anti-windup upper clamp on lambdas
    lambda_min    : anti-windup lower clamp on lambdas (one-sided -> 0)
    integral_max  : anti-windup clamp on the integral accumulator
    """

    n_constraints: int = 3
    targets: list[float] = field(
        default_factory=lambda: [30.0, 5.0, 0.85]
    )
    init_lambdas: list[float] | None = None
    lambda_lr_p: float = 0.05
    lambda_lr_i: float = 0.01
    lambda_max: float = 100.0
    lambda_min: float = 0.0
    integral_max: float = 50.0


class LagrangianDual:
    """PI-controlled dual variables (one per constraint).

    See ``LagrangianConfig`` for the update rule. The class holds both
    the multipliers and the integral accumulator; ``reset()`` returns
    both to their initial values.
    """

    def __init__(self, config: LagrangianConfig) -> None:
        self.config = config
        self._init_lambdas: np.ndarray = (
            np.zeros(config.n_constraints, dtype=np.float64)
            if config.init_lambdas is None
            else np.asarray(config.init_lambdas, dtype=np.float64).copy()
        )
        if self._init_lambdas.shape[0] != config.n_constraints:
            raise ValueError(
                "init_lambdas length must equal n_constraints "
                f"(got {self._init_lambdas.shape[0]} vs {config.n_constraints})"
            )
        self._lambdas: np.ndarray = self._init_lambdas.copy()
        self._integral_accum: np.ndarray = np.zeros(
            config.n_constraints, dtype=np.float64
        )

    def update(self, violations: np.ndarray) -> np.ndarray:
        """Spoor 2025 PI dual update.

        Parameters
        ----------
        violations : array of shape (n_constraints,) where each entry is
                     ``max(0, measured - target)`` (i.e. the over-shoot;
                     negative or zero means the constraint is satisfied).

        Returns the new lambdas.
        """
        v = np.asarray(violations, dtype=np.float64)
        if v.shape[0] != self.config.n_constraints:
            raise ValueError(
                f"violations length {v.shape[0]} != n_constraints "
                f"{self.config.n_constraints}"
            )
        # Anti-windup I: clamp the integral accumulator.
        self._integral_accum = np.clip(
            self._integral_accum + v,
            -self.config.integral_max,
            self.config.integral_max,
        )
        delta = self.config.lambda_lr_p * v + self.config.lambda_lr_i * self._integral_accum
        # Anti-windup II: clamp the multipliers themselves.
        self._lambdas = np.clip(
            self._lambdas + delta,
            self.config.lambda_min,
            self.config.lambda_max,
        )
        return self._lambdas.copy()

    def get_lambdas(self) -> np.ndarray:
        return self._lambdas.copy()

    def reset(self) -> None:
        self._lambdas = self._init_lambdas.copy()
        self._integral_accum = np.zeros(
            self.config.n_constraints, dtype=np.float64
        )


def shaped_reward(
    scalar_reward: float,
    violations: np.ndarray,
    lambdas: np.ndarray,
) -> float:
    """Lagrangian-shaped reward: ``r - sum_i lambda_i * violation_i``."""
    v = np.asarray(violations, dtype=np.float64)
    lam = np.asarray(lambdas, dtype=np.float64)
    if v.shape != lam.shape:
        raise ValueError(
            f"violations shape {v.shape} != lambdas shape {lam.shape}"
        )
    return float(scalar_reward) - float(np.dot(lam, v))


# ---------------------------------------------------------------------------
# Actor-critic (no preference conditioning)
# ---------------------------------------------------------------------------


def _layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class _ActorCritic(nn.Module):
    """Plain actor-critic for PPO-Lagrangian (no omega conditioning).

    Discrete -> Categorical, Continuous -> diagonal Gaussian. Same MLP
    layout as ``PreferenceNetwork`` so behaviour matches preference_ppo.py
    when the preference dim is zero.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
        continuous: bool = False,
    ) -> None:
        super().__init__()
        self.continuous = continuous
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        if continuous:
            self.actor_mean = nn.Sequential(
                _layer_init(nn.Linear(obs_dim, hidden_dim)),
                nn.Tanh(),
                _layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                _layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        else:
            self.actor_logits = nn.Sequential(
                _layer_init(nn.Linear(obs_dim, hidden_dim)),
                nn.Tanh(),
                _layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                _layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
            )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.continuous:
            mean = self.actor_mean(obs)
            logstd = self.actor_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = Normal(mean, std)
            if action is None:
                action = probs.sample()
            return (
                action,
                probs.log_prob(action).sum(-1),
                probs.entropy().sum(-1),
                self.critic(obs),
            )
        else:
            logits = self.actor_logits(obs)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return (
                action,
                probs.log_prob(action),
                probs.entropy(),
                self.critic(obs),
            )


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


class _JsonlWriter:
    """Tiny buffered JSONL writer; flushes on every iteration boundary."""

    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = str(path) if path is not None else None
        self._buf: list[str] = []
        if self.path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
            # Truncate/create the file so callers don't accumulate stale data.
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("")

    def append(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        # numpy scalars are not JSON-serializable; coerce defensively.
        clean: dict[str, Any] = {}
        for k, v in record.items():
            if isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            else:
                clean[k] = v
        self._buf.append(json.dumps(clean))

    def flush(self) -> None:
        if self.path is None or not self._buf:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(self._buf) + "\n")
        self._buf.clear()


# ---------------------------------------------------------------------------
# PPO-Lagrangian training loop
# ---------------------------------------------------------------------------


@dataclass
class PPOLagrangianConfig:
    """Hyperparameters for the PPO loop used inside ``train_ppo_lagrangian``.

    Mirrors the relevant subset of ``PreferencePPOConfig`` minus preference
    conditioning. We expose ``num_steps`` / ``num_minibatches`` /
    ``update_epochs`` directly because the spec's `knob_mapper` arg names
    these knobs.
    """

    num_steps: int = 256
    hidden_dim: int = 64
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    seed: int = 0

    @property
    def batch_size(self) -> int:
        return self.num_steps

    @property
    def minibatch_size(self) -> int:
        return max(1, self.batch_size // self.num_minibatches)


_KNOB_CHOICES = ("n_steps", "n_epochs", "mini_batch_size")


def train_ppo_lagrangian(
    env_fn: Callable[[], gym.Env],
    lagrangian_config: LagrangianConfig,
    ppo_config: PPOLagrangianConfig,
    *,
    telemetry_fn: Callable[[float], HardwareTelemetry] | Callable[[], HardwareTelemetry],
    override: OverrideLayer | None = None,
    total_steps: int = 100_000,
    knob_mapper: str = "n_steps",
    with_override: bool = False,
    log_jsonl_path: str | os.PathLike[str] | None = None,
    verbose: bool = True,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """PPO-Lagrangian training driver.

    Parameters
    ----------
    env_fn            : callable returning a fresh gymnasium env. The env must
                        return scalar OR vector reward; vector rewards are
                        sum-reduced to a scalar before Lagrangian shaping.
    lagrangian_config : ``LagrangianConfig`` controlling the dual update.
    ppo_config        : ``PPOLagrangianConfig`` with PPO hyperparameters.
    telemetry_fn      : callable that returns the latest ``HardwareTelemetry``.
                        For backward compatibility we accept either a 0-arg
                        callable or a 1-arg callable that takes the most
                        recent step-latency in milliseconds; we'll detect
                        which signature it has.
    override          : ``OverrideLayer`` instance (only consulted when
                        ``with_override=True``).
    total_steps       : approximate number of env steps to roll out
                        (rounded up to the nearest rollout multiple).
    knob_mapper       : one of ``{"n_steps", "n_epochs", "mini_batch_size"}``;
                        recorded in the result dict (closed-loop coupling
                        deferred).
    with_override     : if True, the override layer is consulted every step.
    log_jsonl_path    : optional path for per-step JSONL logging.
    verbose           : print rollout-end progress.
    device            : torch device.

    Returns dict with keys: ``network``, ``lambdas_history``,
    ``violation_rate_history``, ``override_fire_count``, ``total_steps``,
    ``knob_mapper``, ``mean_violations``.
    """
    if knob_mapper not in _KNOB_CHOICES:
        raise ValueError(
            f"knob_mapper must be one of {_KNOB_CHOICES}, got {knob_mapper!r}"
        )

    # Telemetry signature detection: we want to support both 0-arg and
    # 1-arg telemetry callables without forcing all call sites to change.
    def _call_telemetry(latency_ms: float) -> HardwareTelemetry:
        try:
            return telemetry_fn(latency_ms)  # type: ignore[call-arg]
        except TypeError:
            return telemetry_fn()  # type: ignore[call-arg]

    np.random.default_rng(ppo_config.seed)
    torch.manual_seed(ppo_config.seed)
    np.random.seed(ppo_config.seed)

    env = env_fn()
    continuous = isinstance(env.action_space, gym.spaces.Box)
    if continuous:
        act_dim = env.action_space.shape[0]
    else:
        act_dim = int(env.action_space.n)
    obs_dim = int(np.prod(env.observation_space.shape))

    network = _ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=ppo_config.hidden_dim,
        continuous=continuous,
    ).to(device)
    optimizer = optim.Adam(
        network.parameters(), lr=ppo_config.lr, eps=1e-5
    )

    # Rollout buffers
    obs_buf = torch.zeros((ppo_config.num_steps, obs_dim), device=device)
    if continuous:
        act_buf = torch.zeros(
            (ppo_config.num_steps, act_dim), device=device
        )
    else:
        act_buf = torch.zeros(
            ppo_config.num_steps, dtype=torch.long, device=device
        )
    logprob_buf = torch.zeros(ppo_config.num_steps, device=device)
    reward_buf = torch.zeros(ppo_config.num_steps, device=device)
    done_buf = torch.zeros(ppo_config.num_steps, device=device)
    value_buf = torch.zeros(ppo_config.num_steps, device=device)

    dual = LagrangianDual(lagrangian_config)
    n_constraints = lagrangian_config.n_constraints
    targets = np.asarray(lagrangian_config.targets, dtype=np.float64)
    if targets.shape[0] != n_constraints:
        env.close()
        raise ValueError(
            f"len(targets)={targets.shape[0]} != n_constraints={n_constraints}"
        )

    use_override = bool(with_override and override is not None)
    override_fire_count = 0

    writer = _JsonlWriter(log_jsonl_path)

    obs, _ = env.reset(seed=ppo_config.seed)
    next_done = 0.0

    global_step = 0
    start_time = time.time()
    num_iterations = max(1, total_steps // ppo_config.num_steps)

    lambdas_history: list[tuple[int, list[float]]] = [
        (0, dual.get_lambdas().tolist())
    ]
    violation_rate_history: list[tuple[int, list[float]]] = []
    # running violation tally for final mean
    violation_sum = np.zeros(n_constraints, dtype=np.float64)
    violation_steps = 0

    for iteration in range(1, num_iterations + 1):
        if ppo_config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * ppo_config.lr

        rollout_violations = np.zeros(
            (ppo_config.num_steps, n_constraints), dtype=np.float64
        )
        rollout_violation_flags = np.zeros(
            (ppo_config.num_steps, n_constraints), dtype=np.bool_
        )

        # === Rollout ===
        for step in range(ppo_config.num_steps):
            global_step += 1
            done_buf[step] = next_done

            obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
            obs_buf[step] = torch.from_numpy(obs_np).to(device)

            with torch.no_grad():
                obs_t = obs_buf[step].unsqueeze(0)
                action, logprob, _, value = network.get_action_and_value(
                    obs_t
                )
                value_buf[step] = value.flatten()[0]
            act_buf[step] = action.squeeze(0)
            logprob_buf[step] = logprob

            if continuous:
                action_np = action.squeeze(0).cpu().numpy()
                # Clip continuous actions to env bounds for safety.
                low = env.action_space.low
                high = env.action_space.high
                action_np = np.clip(action_np, low, high)
            else:
                action_np = int(action.item())

            # Hardware override (decoupled from policy gradient): swap in
            # the safe fallback before stepping the env, but still train on
            # the policy's proposed (action, logprob, value).
            step_action = action_np
            override_fired = False
            if use_override:
                pre_tele = _call_telemetry(0.0)
                fired, fallback = override.step(pre_tele)
                if fired:
                    override_fired = True
                    override_fire_count += 1
                    step_action = fallback

            # Measure step latency around env.step.
            t0 = time.perf_counter()
            next_obs, reward, terminated, truncated, _ = env.step(step_action)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            done = bool(terminated or truncated)

            # Scalar reward: vector envs are sum-reduced; scalar envs pass
            # through. PPO-Lagrangian doesn't condition on omega, so a
            # plain sum is the simplest faithful scalarization.
            reward_arr = np.asarray(reward, dtype=np.float64)
            r_scalar = float(reward_arr.sum()) if reward_arr.ndim > 0 else float(reward_arr)

            tele = _call_telemetry(latency_ms)
            measured = np.array(
                [
                    float(tele.latency_ema_ms or 0.0),
                    # Energy semantics: violation occurs when energy USED
                    # exceeds the budget. We treat the telemetry's
                    # energy_remaining_j as a budget indicator: if it falls
                    # BELOW the target we report a positive over-shoot
                    # equal to (target - remaining). For synthetic test
                    # telemetry that reports energy directly, callers can
                    # set energy_remaining_j = used_so_far instead.
                    float(tele.energy_remaining_j or 0.0),
                    float(tele.memory_util or 0.0),
                ],
                dtype=np.float64,
            )
            # One-sided over-shoot per constraint.
            violations = np.maximum(0.0, measured - targets)
            rollout_violations[step] = violations
            rollout_violation_flags[step] = violations > 0

            r_shaped = shaped_reward(r_scalar, violations, dual.get_lambdas())
            reward_buf[step] = r_shaped

            lam = dual.get_lambdas()
            writer.append(
                {
                    "step": global_step,
                    "iteration": iteration,
                    "latency_ms": float(measured[0]),
                    "energy_j": float(measured[1]),
                    "memory_util": float(measured[2]),
                    "reward_raw": float(r_scalar),
                    "reward_shaped": float(r_shaped),
                    "lambda_T": float(lam[0]) if n_constraints > 0 else 0.0,
                    "lambda_E": float(lam[1]) if n_constraints > 1 else 0.0,
                    "lambda_M": float(lam[2]) if n_constraints > 2 else 0.0,
                    "override_fired": bool(override_fired),
                    "violation_T": float(violations[0]) if n_constraints > 0 else 0.0,
                    "violation_E": float(violations[1]) if n_constraints > 1 else 0.0,
                    "violation_M": float(violations[2]) if n_constraints > 2 else 0.0,
                }
            )

            if done:
                next_done = 1.0
                obs, _ = env.reset()
            else:
                next_done = 0.0
                obs = next_obs

        # === GAE ===
        with torch.no_grad():
            next_obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
            next_t = torch.from_numpy(next_obs_np).unsqueeze(0).to(device)
            next_value = network.get_value(next_t).flatten()[0]
            advantages = torch.zeros(ppo_config.num_steps, device=device)
            lastgaelam = 0.0
            for t in reversed(range(ppo_config.num_steps)):
                if t == ppo_config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1].item()
                    nextvalues = value_buf[t + 1]
                delta = (
                    reward_buf[t]
                    + ppo_config.gamma * nextvalues * nextnonterminal
                    - value_buf[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + ppo_config.gamma
                    * ppo_config.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + value_buf

        # === PPO update ===
        b_inds = np.arange(ppo_config.batch_size)
        pg_loss = torch.tensor(0.0)
        v_loss = torch.tensor(0.0)
        approx_kl = torch.tensor(0.0)

        for epoch in range(ppo_config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(
                0, ppo_config.batch_size, ppo_config.minibatch_size
            ):
                end = start + ppo_config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = network.get_action_and_value(
                    obs_buf[mb_inds], act_buf[mb_inds]
                )
                logratio = newlogprob - logprob_buf[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = advantages[mb_inds]
                if mb_advantages.numel() > 1:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()
                    ) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - ppo_config.clip_coef,
                    1 + ppo_config.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if ppo_config.clip_vloss:
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = value_buf[mb_inds] + torch.clamp(
                        newvalue - value_buf[mb_inds],
                        -ppo_config.clip_coef,
                        ppo_config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss = (
                        0.5
                        * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    )
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - ppo_config.ent_coef * entropy_loss
                    + v_loss * ppo_config.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    network.parameters(), ppo_config.max_grad_norm
                )
                optimizer.step()

            if (
                ppo_config.target_kl is not None
                and approx_kl > ppo_config.target_kl
            ):
                break

        # === Dual update on rollout-mean violation ===
        mean_violation = rollout_violations.mean(axis=0)
        new_lambdas = dual.update(mean_violation)
        lambdas_history.append((global_step, new_lambdas.tolist()))

        per_constraint_rate = rollout_violation_flags.mean(axis=0)
        violation_rate_history.append(
            (global_step, per_constraint_rate.tolist())
        )
        violation_sum += rollout_violations.sum(axis=0)
        violation_steps += ppo_config.num_steps

        writer.flush()

        if verbose:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0
            lam = dual.get_lambdas()
            print(
                f"[iter {iteration}/{num_iterations}] step={global_step} "
                f"lambdas=[{', '.join(f'{x:.3f}' for x in lam)}] "
                f"mean_viol=[{', '.join(f'{x:.3f}' for x in mean_violation)}] "
                f"pg={pg_loss.item():.4f} vl={v_loss.item():.4f} "
                f"override_fires={override_fire_count} SPS={sps}"
            )

    env.close()
    writer.flush()

    mean_violations = (
        violation_sum / max(1, violation_steps)
    ).tolist()

    return {
        "network": network,
        "lambdas_history": lambdas_history,
        "violation_rate_history": violation_rate_history,
        "override_fire_count": override_fire_count,
        "total_steps": global_step,
        "knob_mapper": knob_mapper,
        "mean_violations": mean_violations,
        "final_lambdas": dual.get_lambdas().tolist(),
    }
