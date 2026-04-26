"""Minimal but real DRL algorithms for the P15 matrix (DQN/DDQN/C51/A2C/PPO).

CartPole-shape MLP backbone (hidden=64) by default; switches to a Mnih-2015
NatureCNN encoder + uint8 replay when the observation looks like an Atari
frame-stack ((C, 84, 84)). Uniform interface across algos: ``act / observe /
update`` plus mutable ``batch_size`` / ``replay_capacity`` knobs for system
wrappers.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# --- shared utilities ------------------------------------------------------


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def _is_atari_obs(obs_shape: tuple[int, ...]) -> bool:
    """True when obs looks like an Atari frame-stack: (C, 84, 84)."""
    return len(obs_shape) == 3 and tuple(obs_shape[1:]) == (84, 84)


class NatureCNN(nn.Module):
    """Mnih et al. 2015 Nature-DQN encoder. Input: (B, in_channels, 84, 84) uint8 or float."""

    def __init__(self, in_channels: int = 4, out_dim: int = 512) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return self.fc(self.conv(x))  # type: ignore[no-any-return]


class _CNNQNet(nn.Module):
    """NatureCNN trunk + linear head producing ``out_dim`` logits."""

    def __init__(self, in_channels: int, out_dim: int, feature_dim: int = 512) -> None:
        super().__init__()
        self.trunk = NatureCNN(in_channels=in_channels, out_dim=feature_dim)
        self.head = nn.Linear(feature_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))  # type: ignore[no-any-return]


def _build_qnet(obs_shape: tuple[int, ...], out_dim: int) -> nn.Module:
    """CNN if Atari, MLP otherwise."""
    if _is_atari_obs(obs_shape):
        return _CNNQNet(in_channels=int(obs_shape[0]), out_dim=out_dim)
    in_dim = int(np.prod(obs_shape))
    return _mlp(in_dim, out_dim)


class ReplayBuffer:
    """Fixed-capacity ring buffer of (s, a, r, s', done) numpy arrays."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        obs_dtype: Any = np.float32,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.s = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.a = np.zeros((self.capacity,), dtype=np.int64)
        self.r = np.zeros((self.capacity,), dtype=np.float32)
        self.sn = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.d = np.zeros((self.capacity,), dtype=np.float32)
        self._idx = 0
        self._size = 0

    def push(self, s, a, r, sn, d) -> None:
        i = self._idx
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.sn[i] = sn
        self.d[i] = float(d)
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, n: int) -> tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self._size, size=n)
        return self.s[idx], self.a[idx], self.r[idx], self.sn[idx], self.d[idx]


def _obs_to_tensor(
    obs_batch: np.ndarray,
    obs_shape: tuple[int, ...],
    is_cnn: bool,
    device: torch.device,
) -> torch.Tensor:
    """Reshape numpy obs into a torch tensor matching the backbone's expectations."""
    if is_cnn:
        x = obs_batch.reshape(obs_batch.shape[0], *obs_shape)
        # NatureCNN.forward handles uint8 -> float / 255.
        return torch.as_tensor(x, device=device)
    flat = obs_batch.reshape(obs_batch.shape[0], -1)
    return torch.as_tensor(flat, dtype=torch.float32, device=device)


# --- DQN -------------------------------------------------------------------


class DQNAlgo:
    """Vanilla DQN with target net and epsilon-greedy exploration."""

    paradigm = "off_policy"

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        *,
        seed: int = 0,
        device: str = "cpu",
        replay_capacity: int = 10_000,
        batch_size: int = 64,
        train_after: int = 200,
        target_update_every: int = 100,
        gamma: float = 0.99,
        lr: float = 1e-3,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 5_000,
        **kwargs: Any,
    ) -> None:
        _seed_all(seed)
        self.obs_shape = obs_shape
        self.n_actions = int(n_actions)
        self.device = torch.device(device)
        self.replay_capacity = int(replay_capacity)
        self.batch_size = int(batch_size)
        self.train_after = int(train_after)
        self.target_update_every = int(target_update_every)
        self.gamma = float(gamma)
        self.eps_start, self.eps_end = float(eps_start), float(eps_end)
        self.eps_decay_steps = int(eps_decay_steps)
        self._is_cnn = _is_atari_obs(obs_shape)
        self.q = _build_qnet(obs_shape, self.n_actions).to(self.device)
        self.q_target = _build_qnet(obs_shape, self.n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = Adam(self.q.parameters(), lr=lr)
        obs_dtype = np.uint8 if self._is_cnn else np.float32
        self.buffer = ReplayBuffer(self.replay_capacity, obs_shape, obs_dtype=obs_dtype)
        self._step = 0

    def _eps(self) -> float:
        frac = min(1.0, self._step / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if (not deterministic) and random.random() < self._eps():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = _obs_to_tensor(obs[None, ...], self.obs_shape, self._is_cnn, self.device)
            q = self.q(x)
            return int(torch.argmax(q, dim=-1).item())

    def observe(self, s, a, r, s_next, done) -> None:
        self.buffer.push(s, a, r, s_next, done)
        self._step += 1

    def _td_loss(self, s, a, r, sn, d) -> torch.Tensor:
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(sn).max(dim=1).values
            target = r + self.gamma * (1.0 - d) * q_next
        return F.smooth_l1_loss(q_sa, target)

    def update(self) -> dict[str, float]:
        if len(self.buffer) < self.train_after:
            return {}
        s_np, a_np, r_np, sn_np, d_np = self.buffer.sample(self.batch_size)
        s = _obs_to_tensor(s_np, self.obs_shape, self._is_cnn, self.device)
        sn = _obs_to_tensor(sn_np, self.obs_shape, self._is_cnn, self.device)
        a = torch.as_tensor(a_np, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r_np, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d_np, dtype=torch.float32, device=self.device)
        loss = self._td_loss(s, a, r, sn, d)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self._step % self.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return {"loss": float(loss.item()), "eps": float(self._eps())}


# --- DDQN ------------------------------------------------------------------


class DDQNAlgo(DQNAlgo):
    """Double-DQN: action selection by online net, value by target net."""

    def _td_loss(self, s, a, r, sn, d) -> torch.Tensor:
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_next = self.q(sn).argmax(dim=1, keepdim=True)
            q_next = self.q_target(sn).gather(1, a_next).squeeze(1)
            target = r + self.gamma * (1.0 - d) * q_next
        return F.smooth_l1_loss(q_sa, target)


# --- C51 -------------------------------------------------------------------


class C51Algo:
    """Categorical DQN (C51): 51 atoms over [-10, 10], expected-value action."""

    paradigm = "off_policy"

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        *,
        seed: int = 0,
        device: str = "cpu",
        replay_capacity: int = 10_000,
        batch_size: int = 64,
        train_after: int = 200,
        target_update_every: int = 100,
        gamma: float = 0.99,
        lr: float = 1e-3,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 5_000,
        **kwargs: Any,
    ) -> None:
        _seed_all(seed)
        self.obs_shape = obs_shape
        self.n_actions = int(n_actions)
        self.device = torch.device(device)
        self.replay_capacity = int(replay_capacity)
        self.batch_size = int(batch_size)
        self.train_after = int(train_after)
        self.target_update_every = int(target_update_every)
        self.gamma = float(gamma)
        self.n_atoms = int(n_atoms)
        self.v_min, self.v_max = float(v_min), float(v_max)
        self.eps_start, self.eps_end = float(eps_start), float(eps_end)
        self.eps_decay_steps = int(eps_decay_steps)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms, device=self.device)
        self._is_cnn = _is_atari_obs(obs_shape)
        out_dim = self.n_actions * self.n_atoms
        self.q = _build_qnet(obs_shape, out_dim).to(self.device)
        self.q_target = _build_qnet(obs_shape, out_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = Adam(self.q.parameters(), lr=lr)
        obs_dtype = np.uint8 if self._is_cnn else np.float32
        self.buffer = ReplayBuffer(self.replay_capacity, obs_shape, obs_dtype=obs_dtype)
        self._step = 0

    def _eps(self) -> float:
        frac = min(1.0, self._step / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _dist(self, x: torch.Tensor) -> torch.Tensor:
        # Returns (B, A, n_atoms) probability distribution.
        logits = self.q(x).view(-1, self.n_actions, self.n_atoms)
        return F.softmax(logits, dim=-1)

    def _dist_target(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.q_target(x).view(-1, self.n_actions, self.n_atoms)
        return F.softmax(logits, dim=-1)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if (not deterministic) and random.random() < self._eps():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = _obs_to_tensor(obs[None, ...], self.obs_shape, self._is_cnn, self.device)
            dist = self._dist(x)  # (1, A, atoms)
            q = (dist * self.support).sum(dim=-1)  # (1, A)
            return int(torch.argmax(q, dim=-1).item())

    def observe(self, s, a, r, s_next, done) -> None:
        self.buffer.push(s, a, r, s_next, done)
        self._step += 1

    def update(self) -> dict[str, float]:
        if len(self.buffer) < self.train_after:
            return {}
        s_np, a_np, r_np, sn_np, d_np = self.buffer.sample(self.batch_size)
        B = s_np.shape[0]
        s = _obs_to_tensor(s_np, self.obs_shape, self._is_cnn, self.device)
        sn = _obs_to_tensor(sn_np, self.obs_shape, self._is_cnn, self.device)
        a = torch.as_tensor(a_np, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r_np, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            dist_next = self._dist_target(sn)  # (B, A, atoms)
            q_next = (dist_next * self.support).sum(dim=-1)  # (B, A)
            a_star = q_next.argmax(dim=-1)  # (B,)
            p_next = dist_next[torch.arange(B), a_star]  # (B, atoms)
            # Project Tz onto support.
            Tz = r.unsqueeze(1) + self.gamma * (1.0 - d.unsqueeze(1)) * self.support.unsqueeze(0)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            lo = b.floor().long()
            u = b.ceil().long()
            lo = lo.clamp(0, self.n_atoms - 1)
            u = u.clamp(0, self.n_atoms - 1)
            m = torch.zeros_like(p_next)
            # Handle lo == u edge case by adding to both sides as (u - b) and (b - lo).
            offset = torch.arange(B, device=self.device).unsqueeze(1) * self.n_atoms
            m.view(-1).index_add_(0, (lo + offset).view(-1), (p_next * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (p_next * (b - lo.float())).view(-1))
            # Equal indices: ensure mass is preserved (when b is integer, both u-b and b-lo = 0).
            eq = (lo == u)
            if eq.any():
                m.view(-1).index_add_(0, (lo + offset).view(-1), (p_next * eq.float()).view(-1))

        dist = self._dist(s)  # (B, A, atoms)
        log_p = torch.log(dist[torch.arange(B), a].clamp(min=1e-8))
        loss = -(m * log_p).sum(dim=-1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self._step % self.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return {"loss": float(loss.item()), "eps": float(self._eps())}


# --- Actor-Critic core (shared by A2C / PPO) -------------------------------


class _ActorCritic(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.pi(h), self.v(h).squeeze(-1)


class _CNNActorCritic(nn.Module):
    """NatureCNN trunk + linear policy/value heads."""

    def __init__(self, in_channels: int, n_actions: int, feature_dim: int = 512) -> None:
        super().__init__()
        self.shared = NatureCNN(in_channels=in_channels, out_dim=feature_dim)
        self.pi = nn.Linear(feature_dim, n_actions)
        self.v = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.pi(h), self.v(h).squeeze(-1)


def _build_actor_critic(obs_shape: tuple[int, ...], n_actions: int) -> nn.Module:
    if _is_atari_obs(obs_shape):
        return _CNNActorCritic(in_channels=int(obs_shape[0]), n_actions=n_actions)
    in_dim = int(np.prod(obs_shape))
    return _ActorCritic(in_dim, n_actions)


class _RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        obs_shape: tuple[int, ...],
        obs_dtype: Any = np.float32,
    ) -> None:
        self.cap = int(rollout_steps)
        self.obs_dtype = obs_dtype
        self.s = np.zeros((self.cap, *obs_shape), dtype=obs_dtype)
        self.a = np.zeros((self.cap,), dtype=np.int64)
        self.r = np.zeros((self.cap,), dtype=np.float32)
        self.sn = np.zeros((self.cap, *obs_shape), dtype=obs_dtype)
        self.d = np.zeros((self.cap,), dtype=np.float32)
        self.logp = np.zeros((self.cap,), dtype=np.float32)
        self.v = np.zeros((self.cap,), dtype=np.float32)
        self.size = 0

    def push(self, s, a, r, sn, d, logp, v) -> None:
        i = self.size
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.sn[i] = sn
        self.d[i] = float(d)
        self.logp[i] = logp
        self.v[i] = v
        self.size += 1

    def reset(self) -> None:
        self.size = 0


class _OnPolicyBase:
    paradigm = "on_policy"

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        *,
        seed: int = 0,
        device: str = "cpu",
        rollout_steps: int = 32,
        n_epochs: int = 4,
        mini_batch_size: int = 16,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 1e-3,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        **kwargs: Any,
    ) -> None:
        _seed_all(seed)
        self.obs_shape = obs_shape
        self.n_actions = int(n_actions)
        self.device = torch.device(device)
        self.rollout_steps = int(rollout_steps)
        self.n_epochs = int(n_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        # batch_size mirror so wrappers can read/write a uniform attribute.
        self.batch_size = int(mini_batch_size)
        self._is_cnn = _is_atari_obs(obs_shape)
        self.net = _build_actor_critic(obs_shape, self.n_actions).to(self.device)
        self.opt = Adam(self.net.parameters(), lr=lr)
        obs_dtype = np.uint8 if self._is_cnn else np.float32
        self.buffer = _RolloutBuffer(self.rollout_steps, obs_shape, obs_dtype=obs_dtype)
        self._last_logp: float = 0.0
        self._last_v: float = 0.0

    def _policy(self, x: torch.Tensor) -> torch.distributions.Categorical:
        logits, _ = self.net(x)
        return torch.distributions.Categorical(logits=logits)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        with torch.no_grad():
            x = _obs_to_tensor(obs[None, ...], self.obs_shape, self._is_cnn, self.device)
            logits, v = self.net(x)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                a = int(torch.argmax(logits, dim=-1).item())
            else:
                a = int(dist.sample().item())
            self._last_logp = float(dist.log_prob(torch.as_tensor(a, device=self.device)).item())
            self._last_v = float(v.item())
        return a

    def observe(self, s, a, r, s_next, done) -> None:
        self.buffer.push(s, a, r, s_next, done, self._last_logp, self._last_v)

    def _gae(self, last_v: float) -> tuple[np.ndarray, np.ndarray]:
        n = self.buffer.size
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_v = last_v if t == n - 1 else self.buffer.v[t + 1]
            nonterm = 1.0 - self.buffer.d[t]
            delta = self.buffer.r[t] + self.gamma * next_v * nonterm - self.buffer.v[t]
            gae = delta + self.gamma * self.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + self.buffer.v[:n]
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def _last_value(self) -> float:
        # Bootstrap value of the terminal next-state.
        if self.buffer.size == 0:
            return 0.0
        last_sn = self.buffer.sn[self.buffer.size - 1]
        with torch.no_grad():
            x = _obs_to_tensor(last_sn[None, ...], self.obs_shape, self._is_cnn, self.device)
            _, v = self.net(x)
            return float(v.item())

    def update(self) -> dict[str, float]:
        raise NotImplementedError


class A2CAlgo(_OnPolicyBase):
    """A2C: synchronous advantage actor-critic with GAE returns."""

    def update(self) -> dict[str, float]:
        if self.buffer.size < self.rollout_steps:
            return {}
        last_v = self._last_value()
        adv, ret = self._gae(last_v)
        n = self.buffer.size
        s = _obs_to_tensor(self.buffer.s[:n], self.obs_shape, self._is_cnn, self.device)
        a = torch.as_tensor(self.buffer.a[:n], dtype=torch.long, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        logits, v = self.net(s)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(a)
        ent = dist.entropy().mean()
        pi_loss = -(logp * adv_t).mean()
        v_loss = F.mse_loss(v, ret_t)
        loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.buffer.reset()
        return {"loss": float(loss.item()), "pi_loss": float(pi_loss.item()), "v_loss": float(v_loss.item())}


class PPOAlgo(_OnPolicyBase):
    """PPO with clipped surrogate over multiple epochs of mini-batches."""

    def __init__(self, *args: Any, clip_eps: float = 0.2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.clip_eps = float(clip_eps)

    def update(self) -> dict[str, float]:
        if self.buffer.size < self.rollout_steps:
            return {}
        last_v = self._last_value()
        adv, ret = self._gae(last_v)
        n = self.buffer.size
        s = _obs_to_tensor(self.buffer.s[:n], self.obs_shape, self._is_cnn, self.device)
        a = torch.as_tensor(self.buffer.a[:n], dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor(self.buffer.logp[:n], dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        last_loss = 0.0
        last_pi, last_v_loss = 0.0, 0.0
        idx = np.arange(n)
        mb = max(1, min(self.mini_batch_size, n))
        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb):
                b = idx[start:start + mb]
                bt = torch.as_tensor(b, dtype=torch.long, device=self.device)
                logits, v = self.net(s[bt])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a[bt])
                ent = dist.entropy().mean()
                ratio = torch.exp(logp - old_logp[bt])
                surr1 = ratio * adv_t[bt]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[bt]
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(v, ret_t[bt])
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                last_loss = float(loss.item())
                last_pi = float(pi_loss.item())
                last_v_loss = float(v_loss.item())
        self.buffer.reset()
        return {"loss": last_loss, "pi_loss": last_pi, "v_loss": last_v_loss}
