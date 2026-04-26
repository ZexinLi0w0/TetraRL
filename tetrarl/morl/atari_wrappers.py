"""ALE preprocessing wrappers + ``make_atari_env`` factory.

Mirrors the cleanrl ``dqn_atari.py`` ``make_env`` pipeline using gymnasium APIs:

  NoopResetEnv → MaxAndSkipEnv → EpisodicLifeEnv → FireResetEnv → WarpFrame
   → ClipRewardEnv → FrameStack(k=4)

Outputs uint8 (4, 84, 84) frame stacks suitable for the NatureCNN backbone.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium import spaces

# --- optional resize backends ---------------------------------------------

try:  # pragma: no cover - depends on local install
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover - depends on local install
    _HAS_CV2 = False

try:  # pragma: no cover - depends on local install
    from PIL import Image  # type: ignore

    _HAS_PIL = True
except Exception:  # pragma: no cover - depends on local install
    _HAS_PIL = False


def _to_grayscale_84(frame: np.ndarray) -> np.ndarray:
    """RGB uint8 frame → 84x84 uint8 grayscale, cv2 if present else PIL."""
    if frame.ndim == 3 and frame.shape[-1] == 3:
        if _HAS_CV2:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            # ITU-R BT.601 luma weights to match cv2's default.
            gray = (
                0.299 * frame[..., 0].astype(np.float32)
                + 0.587 * frame[..., 1].astype(np.float32)
                + 0.114 * frame[..., 2].astype(np.float32)
            ).astype(np.uint8)
    else:
        gray = frame.astype(np.uint8)
        if gray.ndim == 3 and gray.shape[-1] == 1:
            gray = gray[..., 0]
    if _HAS_CV2:
        out = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    elif _HAS_PIL:
        img = Image.fromarray(gray)
        out = np.asarray(img.resize((84, 84), Image.BILINEAR), dtype=np.uint8)
    else:
        # Last-resort nearest-neighbour resize via numpy indexing.
        h, w = gray.shape[:2]
        ys = (np.linspace(0, h - 1, 84)).astype(np.int64)
        xs = (np.linspace(0, w - 1, 84)).astype(np.int64)
        out = gray[ys[:, None], xs[None, :]].astype(np.uint8)
    return np.ascontiguousarray(out, dtype=np.uint8)


# --- NoopResetEnv ----------------------------------------------------------


class NoopResetEnv(gym.Wrapper):
    """Reset with a random 1..noop_max NoOp warm-up."""

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = int(noop_max)
        self.noop_action = 0
        meanings = getattr(env.unwrapped, "get_action_meanings", lambda: [])()
        if meanings:
            assert meanings[0] == "NOOP", f"expected NOOP at action 0, got {meanings[:1]}"

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        rng = self.unwrapped.np_random if hasattr(self.unwrapped, "np_random") else np.random
        n = int(rng.integers(1, self.noop_max + 1)) if self.noop_max >= 1 else 0
        for _ in range(n):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset()
        return obs, info


# --- MaxAndSkipEnv --------------------------------------------------------


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat action ``skip`` times; obs = max-pool of last 2 raw frames."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self.skip = int(skip)
        shape = env.observation_space.shape
        dtype = env.observation_space.dtype or np.uint8
        self._obs_buffer = np.zeros((2, *shape), dtype=dtype)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self.skip - 2:
                self._obs_buffer[0] = obs
            elif i == self.skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


# --- EpisodicLifeEnv ------------------------------------------------------


class EpisodicLifeEnv(gym.Wrapper):
    """Treat life loss as terminal for the agent (no real reset until lives==0)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = bool(terminated)
        ale = getattr(self.env.unwrapped, "ale", None)
        if ale is not None:
            lives = int(ale.lives())
            if 0 < lives < self.lives:
                terminated = True
            self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.was_real_done:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            # Step with NoOp to advance past the lost-life screen without
            # truly resetting the underlying ALE state.
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)
        ale = getattr(self.env.unwrapped, "ale", None)
        if ale is not None:
            self.lives = int(ale.lives())
        return obs, info


# --- FireResetEnv ---------------------------------------------------------


class FireResetEnv(gym.Wrapper):
    """Take FIRE on reset (Breakout/Atari games that need a launch action)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        assert "FIRE" in meanings, "FireResetEnv requires FIRE in action_meanings"
        assert len(meanings) >= 3

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.env.reset(seed=seed, options=options)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(seed=seed, options=options)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(seed=seed, options=options)
        return obs, info


# --- WarpFrame ------------------------------------------------------------


class WarpFrame(gym.ObservationWrapper):
    """Resize to 84x84 grayscale uint8."""

    def __init__(
        self,
        env: gym.Env,
        width: int = 84,
        height: int = 84,
        grayscale: bool = True,
    ) -> None:
        super().__init__(env)
        assert width == 84 and height == 84 and grayscale, (
            "WarpFrame here is fixed at 84x84 grayscale (cleanrl convention)."
        )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        return _to_grayscale_84(np.asarray(obs))


# --- ClipRewardEnv --------------------------------------------------------


class ClipRewardEnv(gym.RewardWrapper):
    """Clip reward via ``np.sign`` → {-1, 0, +1}."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward):
        return float(np.sign(float(reward)))


# --- FrameStack -----------------------------------------------------------


class FrameStack(gym.Wrapper):
    """Stack last k single-channel frames into shape (k, 84, 84) uint8."""

    def __init__(self, env: gym.Env, k: int = 4) -> None:
        super().__init__(env)
        self.k = int(k)
        self._frames: deque[np.ndarray] = deque(maxlen=self.k)
        inner_shape = env.observation_space.shape
        assert len(inner_shape) == 2, (
            f"FrameStack expects single-channel (H, W) obs; got {inner_shape}"
        )
        h, w = inner_shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.k, h, w), dtype=np.uint8
        )

    def _stacked(self) -> np.ndarray:
        return np.ascontiguousarray(np.stack(list(self._frames), axis=0), dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        frame = np.asarray(obs, dtype=np.uint8)
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(frame)
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(np.asarray(obs, dtype=np.uint8))
        return self._stacked(), reward, terminated, truncated, info


# --- factory --------------------------------------------------------------


def make_atari_env(
    env_id: str,
    *,
    episodic_life: bool = True,
    clip_reward: bool = True,
    frame_stack: int = 4,
    frame_skip: int = 4,
    noop_max: int = 30,
    fire_reset: bool = True,
) -> gym.Env:
    """Build a cleanrl-style ALE preprocessing pipeline."""
    try:
        import ale_py  # type: ignore

        try:
            gym.register_envs(ale_py)
        except Exception:
            pass
    except Exception:
        pass

    env = gym.make(env_id, frameskip=1)
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if fire_reset and "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_reward:
        env = ClipRewardEnv(env)
    env = FrameStack(env, k=frame_stack)
    return env


__all__ = [
    "ClipRewardEnv",
    "EpisodicLifeEnv",
    "FireResetEnv",
    "FrameStack",
    "MaxAndSkipEnv",
    "NoopResetEnv",
    "WarpFrame",
    "make_atari_env",
]
