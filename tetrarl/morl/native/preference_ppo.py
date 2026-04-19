"""Preference-conditioned PPO for multi-objective RL.

Single-process PPO that conditions on a preference vector omega to learn
policies across the entire preference space. Built on cleanrl's PPO
architecture (vendored in ppo_base.py) with three key modifications:

1. Observation augmentation: obs_aug = concat(obs, omega)
2. Reward scalarization: r_scalar = omega . r_vec (linear scalarization)
3. Periodic Pareto front evaluation across diverse omega samples

Reference architecture: cleanrl ppo_continuous_action.py (Huang et al., 2022)
"""

from __future__ import annotations

import copy
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

from tetrarl.eval.hypervolume import hypervolume, pareto_filter
from tetrarl.morl.preference_sampling import (
    sample_anchor_preferences,
    sample_preference,
)


def layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


@dataclass
class PreferencePPOConfig:
    """Hyperparameters for preference-conditioned PPO."""

    n_objectives: int = 2
    total_timesteps: int = 100_000
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

    eval_interval: int = 10
    eval_episodes: int = 3
    n_eval_interior: int = 10
    ref_point: list[float] = field(default_factory=lambda: [0.0, -25.0])

    seed: int = 0

    @property
    def batch_size(self) -> int:
        return self.num_steps

    @property
    def minibatch_size(self) -> int:
        return max(1, self.batch_size // self.num_minibatches)

    @property
    def num_iterations(self) -> int:
        return self.total_timesteps // self.num_steps


class PreferenceNetwork(nn.Module):
    """Actor-critic conditioned on (observation, preference vector)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        pref_dim: int,
        hidden_dim: int = 64,
        continuous: bool = False,
    ):
        super().__init__()
        input_dim = obs_dim + pref_dim
        self.continuous = continuous
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pref_dim = pref_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        if continuous:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(input_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        else:
            self.actor_logits = nn.Sequential(
                layer_init(nn.Linear(input_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
            )

    def get_value(self, obs_aug: torch.Tensor) -> torch.Tensor:
        return self.critic(obs_aug)

    def get_action_and_value(
        self, obs_aug: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.continuous:
            action_mean = self.actor_mean(obs_aug)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return (
                action,
                probs.log_prob(action).sum(-1),
                probs.entropy().sum(-1),
                self.critic(obs_aug),
            )
        else:
            logits = self.actor_logits(obs_aug)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return (
                action,
                probs.log_prob(action),
                probs.entropy(),
                self.critic(obs_aug),
            )

    def get_deterministic_action(
        self, obs_aug: torch.Tensor
    ) -> torch.Tensor:
        if self.continuous:
            return self.actor_mean(obs_aug)
        else:
            return self.actor_logits(obs_aug).argmax(-1)


def evaluate_policy(
    network: PreferenceNetwork,
    env: gym.Env,
    omega: np.ndarray,
    n_episodes: int = 3,
    device: str | torch.device = "cpu",
    deterministic: bool = True,
) -> np.ndarray:
    """Evaluate current policy at a specific preference vector.

    Returns the mean multi-objective return across episodes.
    """
    all_returns: list[np.ndarray] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_rewards: list[np.ndarray] = []
        done = False
        while not done:
            obs_aug = np.concatenate([obs, omega]).astype(np.float32)
            obs_t = torch.from_numpy(obs_aug).unsqueeze(0).to(device)
            with torch.no_grad():
                if deterministic:
                    action = network.get_deterministic_action(obs_t)
                else:
                    action, _, _, _ = network.get_action_and_value(obs_t)
            if isinstance(env.action_space, gym.spaces.Discrete):
                act = int(action.item())
            else:
                act = action.squeeze(0).cpu().numpy()
            obs, reward_vec, terminated, truncated, _ = env.step(act)
            episode_rewards.append(np.asarray(reward_vec, dtype=np.float64))
            done = terminated or truncated
        all_returns.append(np.sum(episode_rewards, axis=0))
    return np.mean(all_returns, axis=0)


def _build_eval_preferences(
    n_obj: int, n_interior: int, rng: np.random.Generator
) -> np.ndarray:
    """Corner preferences + random interior samples."""
    anchors = sample_anchor_preferences(n_obj)
    interior = sample_preference(n_obj, n_interior, rng=rng)
    return np.concatenate([anchors, interior], axis=0)


def train_preference_ppo(
    config: PreferencePPOConfig,
    env_fn: Callable[[], gym.Env],
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """Run preference-conditioned PPO training.

    Returns dict with keys: network, pareto_front, hv_history,
    best_hv, global_step, config.
    """
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = env_fn()
    obs_dim = int(np.prod(env.observation_space.shape))
    continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = (
        env.action_space.shape[0] if continuous else env.action_space.n
    )
    n_obj = config.n_objectives

    network = PreferenceNetwork(
        obs_dim, act_dim, n_obj, config.hidden_dim, continuous
    ).to(device)
    optimizer = optim.Adam(network.parameters(), lr=config.lr, eps=1e-5)

    # Rollout buffers
    aug_dim = obs_dim + n_obj
    obs_buf = torch.zeros((config.num_steps, aug_dim), device=device)
    if continuous:
        act_buf = torch.zeros(
            (config.num_steps, act_dim), device=device
        )
    else:
        act_buf = torch.zeros(
            config.num_steps, dtype=torch.long, device=device
        )
    logprob_buf = torch.zeros(config.num_steps, device=device)
    reward_buf = torch.zeros(config.num_steps, device=device)
    done_buf = torch.zeros(config.num_steps, device=device)
    value_buf = torch.zeros(config.num_steps, device=device)

    obs, _ = env.reset(seed=config.seed)
    omega = sample_preference(n_obj, 1, rng=rng)[0]
    next_done = 0.0

    global_step = 0
    start_time = time.time()
    hv_history: list[tuple[int, float, int]] = []
    pareto_front = np.empty((0, n_obj))
    best_hv = 0.0
    best_network_state: dict[str, Any] | None = None

    eval_env = env_fn()

    for iteration in range(1, config.num_iterations + 1):
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.num_iterations
            optimizer.param_groups[0]["lr"] = frac * config.lr

        # === Collect rollout ===
        for step in range(config.num_steps):
            global_step += 1

            obs_aug = np.concatenate([obs, omega]).astype(np.float32)
            obs_buf[step] = torch.from_numpy(obs_aug).to(device)
            done_buf[step] = next_done

            with torch.no_grad():
                obs_t = obs_buf[step].unsqueeze(0)
                action, logprob, _, value = (
                    network.get_action_and_value(obs_t)
                )
                value_buf[step] = value.flatten()[0]

            act_buf[step] = action.squeeze(0)
            logprob_buf[step] = logprob

            if continuous:
                action_np = action.squeeze(0).cpu().numpy()
            else:
                action_np = int(action.item())

            next_obs, reward_vec, terminated, truncated, _ = env.step(
                action_np
            )
            done = terminated or truncated

            r_scalar = float(np.dot(omega, reward_vec))
            reward_buf[step] = r_scalar

            if done:
                next_done = 1.0
                obs, _ = env.reset()
                omega = sample_preference(n_obj, 1, rng=rng)[0]
            else:
                next_done = 0.0
                obs = next_obs

        # === GAE ===
        with torch.no_grad():
            next_aug = np.concatenate([obs, omega]).astype(np.float32)
            next_t = torch.from_numpy(next_aug).unsqueeze(0).to(device)
            next_value = network.get_value(next_t).flatten()[0]

            advantages = torch.zeros(config.num_steps, device=device)
            lastgaelam = 0.0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1].item()
                    nextvalues = value_buf[t + 1]
                delta = (
                    reward_buf[t]
                    + config.gamma * nextvalues * nextnonterminal
                    - value_buf[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + config.gamma
                    * config.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + value_buf

        # === PPO update ===
        b_inds = np.arange(config.batch_size)
        clipfracs: list[float] = []
        pg_loss = torch.tensor(0.0)
        v_loss = torch.tensor(0.0)
        entropy_loss = torch.tensor(0.0)
        approx_kl = torch.tensor(0.0)

        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(
                0, config.batch_size, config.minibatch_size
            ):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = (
                    network.get_action_and_value(
                        obs_buf[mb_inds], act_buf[mb_inds]
                    )
                )
                logratio = newlogprob - logprob_buf[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > config.clip_coef)
                        .float()
                        .mean()
                        .item()
                    )

                mb_advantages = advantages[mb_inds]
                mb_advantages = (
                    mb_advantages - mb_advantages.mean()
                ) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - config.clip_coef,
                    1 + config.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (
                        newvalue - returns[mb_inds]
                    ) ** 2
                    v_clipped = value_buf[mb_inds] + torch.clamp(
                        newvalue - value_buf[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (
                        v_clipped - returns[mb_inds]
                    ) ** 2
                    v_loss = (
                        0.5
                        * torch.max(v_loss_unclipped, v_loss_clipped)
                        .mean()
                    )
                else:
                    v_loss = (
                        0.5
                        * ((newvalue - returns[mb_inds]) ** 2).mean()
                    )

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - config.ent_coef * entropy_loss
                    + v_loss * config.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    network.parameters(), config.max_grad_norm
                )
                optimizer.step()

            if (
                config.target_kl is not None
                and approx_kl > config.target_kl
            ):
                break

        # === Periodic evaluation ===
        if (
            iteration % config.eval_interval == 0
            or iteration == config.num_iterations
        ):
            eval_prefs = _build_eval_preferences(
                n_obj, config.n_eval_interior, rng
            )
            eval_returns: list[np.ndarray] = []
            for omega_eval in eval_prefs:
                obj_return = evaluate_policy(
                    network,
                    eval_env,
                    omega_eval,
                    config.eval_episodes,
                    device,
                )
                eval_returns.append(obj_return)

            eval_arr = np.array(eval_returns)
            ref = np.array(config.ref_point, dtype=np.float64)
            pf = pareto_filter(eval_arr)
            hv = hypervolume(pf, ref)
            hv_history.append((global_step, hv, len(pf)))

            if hv >= best_hv:
                best_hv = hv
                pareto_front = pf.copy()
                best_network_state = copy.deepcopy(
                    network.state_dict()
                )

            if verbose:
                elapsed = time.time() - start_time
                sps = int(global_step / elapsed) if elapsed > 0 else 0
                print(
                    f"[iter {iteration}/{config.num_iterations}] "
                    f"step={global_step} HV={hv:.2f} |PF|={len(pf)} "
                    f"pg={pg_loss.item():.4f} "
                    f"vl={v_loss.item():.4f} "
                    f"SPS={sps}"
                )

    env.close()
    eval_env.close()

    if best_network_state is not None:
        network.load_state_dict(best_network_state)

    return {
        "network": network,
        "pareto_front": pareto_front,
        "hv_history": hv_history,
        "best_hv": best_hv,
        "global_step": global_step,
        "config": config,
    }
