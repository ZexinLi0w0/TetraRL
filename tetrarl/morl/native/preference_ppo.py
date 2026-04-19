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
from tetrarl.morl.native.masking import ActionMask, apply_logit_mask
from tetrarl.morl.native.override import (
    HardwareTelemetry,
    OverrideLayer,
)
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
    """Actor-critic conditioned on (observation, preference vector).

    Optionally accepts an action mask (applied to discrete logits) and a
    GNN feature extractor (replaces the flat observation with a graph-level
    embedding before concatenating with the preference vector).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        pref_dim: int,
        hidden_dim: int = 64,
        continuous: bool = False,
        *,
        action_mask: ActionMask | None = None,
        gnn_extractor: nn.Module | None = None,
    ):
        super().__init__()
        self.continuous = continuous
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pref_dim = pref_dim
        self.action_mask = action_mask
        self.gnn_extractor = gnn_extractor

        # When a GNN extractor is supplied, the MLP heads consume
        # (gnn_out_dim + pref_dim,). Otherwise they consume (obs_dim + pref_dim,).
        if gnn_extractor is not None:
            gnn_out_dim = int(getattr(gnn_extractor, "out_dim"))
            feat_dim = gnn_out_dim
            self.expects_graph = True
        else:
            feat_dim = obs_dim
            self.expects_graph = False
        input_dim = feat_dim + pref_dim
        self._feat_dim = feat_dim

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

    def _build_graph_feat(
        self,
        graph_obs: dict,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        """Run the GNN extractor and concat preference vector.

        Returns a (B, gnn_out + pref_dim) feature tensor that plays the
        same role as ``obs_aug`` in the flat-observation code path.
        """
        if not self.expects_graph or self.gnn_extractor is None:
            raise RuntimeError(
                "graph_obs supplied but PreferenceNetwork has no GNN extractor"
            )
        if omega is None:
            raise ValueError("omega must be provided alongside graph_obs")
        graph_emb = self.gnn_extractor(
            graph_obs["node_features"],
            graph_obs["edge_index"],
            graph_obs.get("batch"),
        )
        if omega.dim() == 1:
            omega = omega.unsqueeze(0)
        if omega.shape[0] != graph_emb.shape[0]:
            raise ValueError(
                f"omega batch dim {omega.shape[0]} does not match graph batch "
                f"dim {graph_emb.shape[0]}"
            )
        return torch.cat([graph_emb, omega], dim=-1)

    def get_value(
        self,
        obs_aug: torch.Tensor | None = None,
        *,
        graph_obs: dict | None = None,
        omega: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if graph_obs is not None:
            feat = self._build_graph_feat(graph_obs, omega)
            return self.critic(feat)
        return self.critic(obs_aug)

    def _actor_critic(
        self,
        feat: torch.Tensor,
        action: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.continuous:
            # Continuous branch: mask is not applicable; ignored silently.
            action_mean = self.actor_mean(feat)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return (
                action,
                probs.log_prob(action).sum(-1),
                probs.entropy().sum(-1),
                self.critic(feat),
            )
        else:
            logits = self.actor_logits(feat)
            if mask is not None:
                logits = apply_logit_mask(logits, mask)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return (
                action,
                probs.log_prob(action),
                probs.entropy(),
                self.critic(feat),
            )

    def get_action_and_value(
        self,
        obs_aug: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        graph_obs: dict | None = None,
        omega: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if graph_obs is not None:
            feat = self._build_graph_feat(graph_obs, omega)
        else:
            feat = obs_aug
        return self._actor_critic(feat, action, mask)

    def get_deterministic_action(
        self,
        obs_aug: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        graph_obs: dict | None = None,
        omega: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if graph_obs is not None:
            feat = self._build_graph_feat(graph_obs, omega)
        else:
            feat = obs_aug
        if self.continuous:
            # Continuous branch: mask is not applicable; ignored silently.
            return self.actor_mean(feat)
        else:
            logits = self.actor_logits(feat)
            if mask is not None:
                logits = apply_logit_mask(logits, mask)
            return logits.argmax(-1)


def evaluate_policy(
    network: PreferenceNetwork,
    env: gym.Env,
    omega: np.ndarray,
    n_episodes: int = 3,
    device: str | torch.device = "cpu",
    deterministic: bool = True,
    mask_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Evaluate current policy at a specific preference vector.

    Optionally accepts a `mask_fn(obs) -> bool ndarray` (applied to the raw
    obs, not obs_aug). Continuous action spaces ignore the mask.

    Returns the mean multi-objective return across episodes.
    """
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    act_dim = env.action_space.n if discrete else None
    is_graph = isinstance(env.observation_space, gym.spaces.Dict)
    all_returns: list[np.ndarray] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_rewards: list[np.ndarray] = []
        done = False
        while not done:
            mask_t: torch.Tensor | None = None
            if mask_fn is not None and discrete:
                m = np.asarray(mask_fn(obs), dtype=bool)
                mask_t = (
                    torch.as_tensor(m, dtype=torch.bool, device=device)
                    .unsqueeze(0)
                )
            if is_graph:
                nf = torch.from_numpy(
                    np.asarray(obs["node_features"], dtype=np.float32)
                ).to(device)
                ne = int(obs["num_edges"])
                ei_full = np.asarray(obs["edge_index"], dtype=np.int64)
                if ne > 0:
                    ei = torch.from_numpy(ei_full[:, :ne]).long().to(device)
                else:
                    ei = torch.zeros((2, 0), dtype=torch.long, device=device)
                om = (
                    torch.from_numpy(np.asarray(omega, dtype=np.float32))
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                graph_obs = {
                    "node_features": nf,
                    "edge_index": ei,
                    "batch": None,
                }
                with torch.no_grad():
                    if deterministic:
                        action = network.get_deterministic_action(
                            graph_obs=graph_obs, omega=om, mask=mask_t
                        )
                    else:
                        action, _, _, _ = network.get_action_and_value(
                            graph_obs=graph_obs, omega=om, mask=mask_t
                        )
            else:
                obs_aug = np.concatenate([obs, omega]).astype(np.float32)
                obs_t = torch.from_numpy(obs_aug).unsqueeze(0).to(device)
                with torch.no_grad():
                    if deterministic:
                        action = network.get_deterministic_action(
                            obs_t, mask=mask_t
                        )
                    else:
                        action, _, _, _ = network.get_action_and_value(
                            obs_t, mask=mask_t
                        )
            if discrete:
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
    *,
    mask: ActionMask | None = None,
    override: OverrideLayer | None = None,
    telemetry_fn: Callable[[], HardwareTelemetry] | None = None,
    gnn_extractor: nn.Module | None = None,
) -> dict[str, Any]:
    """Run preference-conditioned PPO training.

    Optional extensions (default-OFF for backward compat):
      * mask: ActionMask -- per-step discrete-action mask applied to logits.
      * override: OverrideLayer + telemetry_fn -- swap env-step action with
        a safe fallback when telemetry violates thresholds. Policy gradient
        is unaffected (the policy's proposed action and logprob are still
        what gets stored and trained on).
      * gnn_extractor: nn.Module -- forwarded to PreferenceNetwork.

    Returns dict with keys: network, pareto_front, hv_history,
    best_hv, global_step, config, override_fire_count.
    """
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = env_fn()
    is_graph = isinstance(env.observation_space, gym.spaces.Dict)

    # Consistency check: graph extractor <-> Dict obs space.
    if (gnn_extractor is not None) != is_graph:
        env.close()
        raise ValueError(
            "Inconsistent configuration: graph envs (Dict obs space) require a "
            "gnn_extractor; flat envs must not be paired with one. "
            f"is_graph={is_graph}, gnn_extractor={'set' if gnn_extractor is not None else 'None'}."
        )

    continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = (
        env.action_space.shape[0] if continuous else env.action_space.n
    )
    n_obj = config.n_objectives

    if is_graph:
        obs_space = env.observation_space
        n_nodes = int(obs_space["node_features"].shape[0])
        feat_dim = int(obs_space["node_features"].shape[1])
        e_max = int(obs_space["edge_index"].shape[1])
        # PreferenceNetwork's flat-MLP heads ignore obs_dim when an extractor
        # is supplied, but we still pass the per-node feature dim so the
        # constructor signature stays well-formed.
        obs_dim = feat_dim
    else:
        obs_dim = int(np.prod(env.observation_space.shape))
        n_nodes = feat_dim = e_max = 0

    network = PreferenceNetwork(
        obs_dim,
        act_dim,
        n_obj,
        config.hidden_dim,
        continuous,
        gnn_extractor=gnn_extractor,
    ).to(device)
    optimizer = optim.Adam(network.parameters(), lr=config.lr, eps=1e-5)

    # Rollout buffers
    if is_graph:
        nf_buf = torch.zeros(
            (config.num_steps, n_nodes, feat_dim), device=device
        )
        ei_buf = torch.zeros(
            (config.num_steps, 2, max(e_max, 1)),
            dtype=torch.long,
            device=device,
        )
        ne_buf = torch.zeros(
            config.num_steps, dtype=torch.long, device=device
        )
        om_buf = torch.zeros((config.num_steps, n_obj), device=device)
        obs_buf = None
    else:
        aug_dim = obs_dim + n_obj
        obs_buf = torch.zeros((config.num_steps, aug_dim), device=device)
        nf_buf = ei_buf = ne_buf = om_buf = None

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

    # Mask buffer is allocated only when masking is active for a discrete env.
    use_mask = (mask is not None) and (not continuous)
    mask_buf: torch.Tensor | None = None
    if use_mask:
        mask_buf = torch.zeros(
            (config.num_steps, act_dim), dtype=torch.bool, device=device
        )

    use_override = override is not None and telemetry_fn is not None
    override_fire_count = 0

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

            done_buf[step] = next_done

            if is_graph:
                nf_np = np.asarray(obs["node_features"], dtype=np.float32)
                nf_buf[step] = torch.from_numpy(nf_np).to(device)
                ne = int(obs["num_edges"])
                ne_buf[step] = ne
                if ne > 0:
                    ei_np = np.asarray(obs["edge_index"], dtype=np.int64)[:, :ne]
                    ei_buf[step, :, :ne] = (
                        torch.from_numpy(ei_np).long().to(device)
                    )
                om_buf[step] = torch.from_numpy(
                    np.asarray(omega, dtype=np.float32)
                ).to(device)
            else:
                obs_aug = np.concatenate([obs, omega]).astype(np.float32)
                obs_buf[step] = torch.from_numpy(obs_aug).to(device)

            mask_t: torch.Tensor | None = None
            if use_mask:
                m = mask.as_tensor(obs, act_dim, device=device)
                mask_buf[step] = m
                mask_t = m.unsqueeze(0)

            with torch.no_grad():
                if is_graph:
                    nf_t = nf_buf[step]
                    ne_step = int(ne_buf[step].item())
                    ei_t = ei_buf[step, :, :ne_step]
                    om_t = om_buf[step].unsqueeze(0)
                    action, logprob, _, value = (
                        network.get_action_and_value(
                            graph_obs={
                                "node_features": nf_t,
                                "edge_index": ei_t,
                                "batch": None,
                            },
                            omega=om_t,
                            mask=mask_t,
                        )
                    )
                else:
                    obs_t = obs_buf[step].unsqueeze(0)
                    action, logprob, _, value = (
                        network.get_action_and_value(obs_t, mask=mask_t)
                    )
                value_buf[step] = value.flatten()[0]

            act_buf[step] = action.squeeze(0)
            logprob_buf[step] = logprob

            if continuous:
                action_np = action.squeeze(0).cpu().numpy()
            else:
                action_np = int(action.item())

            # Hardware override: substitute the env-step action with a
            # safe fallback while still training on the policy's choice.
            step_action = action_np
            if use_override:
                tele = telemetry_fn()
                override_fired, fallback = override.step(tele)
                if override_fired:
                    override_fire_count += 1
                    step_action = fallback

            next_obs, reward_vec, terminated, truncated, _ = env.step(
                step_action
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
            if is_graph:
                nf_next = torch.from_numpy(
                    np.asarray(obs["node_features"], dtype=np.float32)
                ).to(device)
                ne_next = int(obs["num_edges"])
                if ne_next > 0:
                    ei_next = (
                        torch.from_numpy(
                            np.asarray(
                                obs["edge_index"], dtype=np.int64
                            )[:, :ne_next]
                        )
                        .long()
                        .to(device)
                    )
                else:
                    ei_next = torch.zeros(
                        (2, 0), dtype=torch.long, device=device
                    )
                om_next = (
                    torch.from_numpy(np.asarray(omega, dtype=np.float32))
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                next_value = network.get_value(
                    graph_obs={
                        "node_features": nf_next,
                        "edge_index": ei_next,
                        "batch": None,
                    },
                    omega=om_next,
                ).flatten()[0]
            else:
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

                mb_mask = (
                    mask_buf[mb_inds] if mask_buf is not None else None
                )
                if is_graph:
                    # Build a merged batched graph: stack each rollout step's
                    # (n_nodes, feat_dim) node tensor and offset its edges by
                    # k * n_nodes to keep the per-graph adjacencies disjoint.
                    nfs: list[torch.Tensor] = []
                    eis: list[torch.Tensor] = []
                    batches: list[torch.Tensor] = []
                    for k, i in enumerate(mb_inds):
                        nfs.append(nf_buf[i])
                        ne_i = int(ne_buf[i].item())
                        if ne_i > 0:
                            ei_k = ei_buf[i, :, :ne_i] + (k * n_nodes)
                            eis.append(ei_k)
                        batches.append(
                            torch.full(
                                (n_nodes,),
                                k,
                                dtype=torch.long,
                                device=device,
                            )
                        )
                    merged_nf = torch.cat(nfs, dim=0)
                    if eis:
                        merged_ei = torch.cat(eis, dim=1)
                    else:
                        merged_ei = torch.zeros(
                            (2, 0), dtype=torch.long, device=device
                        )
                    merged_batch = torch.cat(batches)
                    omega_mb = om_buf[mb_inds]
                    _, newlogprob, entropy, newvalue = (
                        network.get_action_and_value(
                            graph_obs={
                                "node_features": merged_nf,
                                "edge_index": merged_ei,
                                "batch": merged_batch,
                            },
                            omega=omega_mb,
                            action=act_buf[mb_inds],
                            mask=mb_mask,
                        )
                    )
                else:
                    _, newlogprob, entropy, newvalue = (
                        network.get_action_and_value(
                            obs_buf[mb_inds],
                            act_buf[mb_inds],
                            mask=mb_mask,
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
                    f"SPS={sps} "
                    f"override_fires={override_fire_count}"
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
        "override_fire_count": override_fire_count,
    }
