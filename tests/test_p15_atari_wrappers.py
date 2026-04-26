"""Tests for the P15 ALE preprocessing wrappers + ``make_atari_env``.

Covers:
- ``make_atari_env("ALE/Breakout-v5")`` shape/dtype contract (uint8, (4,84,84), n=4).
- reset/step output contract.
- ``MaxAndSkipEnv`` calls inner ``env.step`` exactly ``skip`` times per outer step.
- ``ClipRewardEnv`` clips via ``np.sign``.
- ``WarpFrame`` produces single-channel uint8 (84, 84).
- ``FrameStack(k=4)`` produces (4, 84, 84) uint8 with the post-reset stack
  filled by the initial frame.
- ``EpisodicLifeEnv`` reports ``terminated=True`` on life loss without resetting
  the underlying env.
- ``NoopResetEnv`` performs >=1 NoOp on reset.
- ``FireResetEnv`` triggers FIRE on reset.
- end-to-end ALE smoke (skipped when ALE isn't installed locally).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import spaces

from tetrarl.morl.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    FrameStack,
    MaxAndSkipEnv,
    NoopResetEnv,
    WarpFrame,
    make_atari_env,
)


try:
    import ale_py  # noqa: F401

    _ALE_AVAILABLE = True
except Exception:
    _ALE_AVAILABLE = False


# --- mock envs --------------------------------------------------------------


class _MockAtariRGB(gym.Env):
    """Tiny env emitting RGB-shaped uint8 frames; lets us exercise WarpFrame
    and FrameStack without ALE."""

    metadata: dict[str, Any] = {}

    def __init__(self, h: int = 100, w: int = 120) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)
        self._h, self._w = h, w
        self._t = 0

    def _frame(self) -> np.ndarray:
        return np.full((self._h, self._w, 3), self._t % 255, dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._t = 0
        return self._frame(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= 100
        return self._frame(), 0.0, terminated, False, {}


class _StepCounterEnv(gym.Env):
    """Counts how many times .step is invoked."""

    metadata: dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(2, 2, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.n_steps = 0
        self.n_resets = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.n_resets += 1
        return np.zeros((2, 2, 3), dtype=np.uint8), {}

    def step(self, action):
        self.n_steps += 1
        # Reward grows each call so we can verify summation.
        return (
            np.full((2, 2, 3), self.n_steps, dtype=np.uint8),
            float(self.n_steps),
            False,
            False,
            {},
        )


class _RewardInjectEnv(gym.Env):
    """Returns whatever reward we set externally."""

    metadata: dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.next_reward = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((1,), dtype=np.float32), float(self.next_reward), False, False, {}


class _FakeALE:
    def __init__(self, lives: int = 5) -> None:
        self._lives = lives

    def lives(self) -> int:
        return self._lives


class _LivesEnv(gym.Env):
    """Mock with controllable .unwrapped.ale.lives()."""

    metadata: dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(2, 2, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.ale = _FakeALE(lives=5)
        self._real_terminated = False
        self.n_real_resets = 0

    def get_action_meanings(self) -> list[str]:
        return ["NOOP", "FIRE", "RIGHT", "LEFT"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.n_real_resets += 1
        self.ale = _FakeALE(lives=5)
        self._real_terminated = False
        return np.zeros((2, 2, 3), dtype=np.uint8), {}

    def step(self, action):
        return (
            np.zeros((2, 2, 3), dtype=np.uint8),
            0.0,
            self._real_terminated,
            False,
            {},
        )


class _NoopCounterEnv(gym.Env):
    """Counts NoOp (action==0) invocations on reset, with action_meanings."""

    metadata: dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(2, 2, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.noop_count = 0
        self.fire_count = 0
        self.ale = _FakeALE(lives=1)

    def get_action_meanings(self) -> list[str]:
        return ["NOOP", "FIRE", "RIGHT", "LEFT"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.noop_count = 0
        self.fire_count = 0
        return np.zeros((2, 2, 3), dtype=np.uint8), {}

    def step(self, action):
        if action == 0:
            self.noop_count += 1
        elif action == 1:
            self.fire_count += 1
        return np.zeros((2, 2, 3), dtype=np.uint8), 0.0, False, False, {}


# --- MaxAndSkipEnv ---------------------------------------------------------


def test_max_and_skip_calls_inner_step_exactly_skip_times() -> None:
    inner = _StepCounterEnv()
    env = MaxAndSkipEnv(inner, skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert inner.n_steps == 4
    # Reward should be summed 1+2+3+4 = 10
    assert reward == pytest.approx(10.0)


def test_max_and_skip_returns_max_pool_of_last_two_frames() -> None:
    inner = _StepCounterEnv()
    env = MaxAndSkipEnv(inner, skip=4)
    env.reset()
    obs, _, _, _, _ = env.step(0)
    # Last two raw frames carry values 3 and 4 → max == 4 everywhere.
    assert obs.dtype == np.uint8
    assert int(obs.max()) == 4
    assert int(obs.min()) == 4


# --- ClipRewardEnv ---------------------------------------------------------


def test_clip_reward_env_clips_via_sign() -> None:
    inner = _RewardInjectEnv()
    env = ClipRewardEnv(inner)
    env.reset()
    inner.next_reward = 5.0
    _, r1, _, _, _ = env.step(0)
    assert r1 == 1.0
    inner.next_reward = -3.0
    _, r2, _, _, _ = env.step(0)
    assert r2 == -1.0
    inner.next_reward = 0.0
    _, r3, _, _, _ = env.step(0)
    assert r3 == 0.0


# --- WarpFrame -------------------------------------------------------------


def test_warp_frame_produces_single_channel_84x84_uint8() -> None:
    inner = _MockAtariRGB(h=210, w=160)
    env = WarpFrame(inner)
    obs, _ = env.reset()
    assert obs.shape == (84, 84)
    assert obs.dtype == np.uint8
    obs2, _, _, _, _ = env.step(0)
    assert obs2.shape == (84, 84)
    assert obs2.dtype == np.uint8
    # observation_space matches.
    assert env.observation_space.shape == (84, 84)
    assert env.observation_space.dtype == np.uint8


# --- FrameStack -------------------------------------------------------------


def test_frame_stack_post_reset_repeats_initial_frame() -> None:
    inner = _MockAtariRGB(h=210, w=160)
    env = FrameStack(WarpFrame(inner), k=4)
    obs, _ = env.reset()
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8
    # All 4 channels equal initial frame after reset.
    assert np.array_equal(obs[0], obs[1])
    assert np.array_equal(obs[1], obs[2])
    assert np.array_equal(obs[2], obs[3])


def test_frame_stack_returns_contiguous_numpy_array() -> None:
    inner = _MockAtariRGB(h=210, w=160)
    env = FrameStack(WarpFrame(inner), k=4)
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.flags["C_CONTIGUOUS"]
    obs2, _, _, _, _ = env.step(0)
    assert isinstance(obs2, np.ndarray)
    assert obs2.flags["C_CONTIGUOUS"]
    assert obs2.shape == (4, 84, 84)


# --- EpisodicLifeEnv -------------------------------------------------------


def test_episodic_life_env_terminates_on_life_loss_without_real_reset() -> None:
    inner = _LivesEnv()
    env = EpisodicLifeEnv(inner)
    env.reset()
    pre_resets = inner.n_real_resets
    # Drop a life mid-step.
    inner.ale = _FakeALE(lives=4)
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is True
    # The underlying env should NOT have been hard-reset.
    assert inner.n_real_resets == pre_resets


def test_episodic_life_env_user_reset_skips_real_reset_when_lives_remain() -> None:
    inner = _LivesEnv()
    env = EpisodicLifeEnv(inner)
    env.reset()
    initial_resets = inner.n_real_resets
    # Trigger life-loss.
    inner.ale = _FakeALE(lives=3)
    env.step(0)
    # User-facing reset should NOT call inner.reset() again, since lives>0.
    env.reset()
    assert inner.n_real_resets == initial_resets


# --- NoopResetEnv ----------------------------------------------------------


def test_noop_reset_env_runs_at_least_one_noop_on_reset() -> None:
    inner = _NoopCounterEnv()
    env = NoopResetEnv(inner, noop_max=5)
    env.reset(seed=0)
    assert inner.noop_count >= 1
    assert inner.noop_count <= 5


# --- FireResetEnv ----------------------------------------------------------


def test_fire_reset_env_invokes_fire_on_reset() -> None:
    inner = _NoopCounterEnv()
    env = FireResetEnv(inner)
    env.reset()
    assert inner.fire_count >= 1


# --- end-to-end ALE smoke -------------------------------------------------


@pytest.mark.skipif(not _ALE_AVAILABLE, reason="ALE not installed")
def test_make_atari_env_breakout_observation_space_shape_dtype() -> None:
    env = make_atari_env("ALE/Breakout-v5")
    assert env.observation_space.shape == (4, 84, 84)
    assert env.observation_space.dtype == np.uint8
    assert env.action_space.n == 4
    obs, _ = env.reset()
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8


@pytest.mark.skipif(not _ALE_AVAILABLE, reason="ALE not installed")
def test_make_atari_env_step_contract() -> None:
    env = make_atari_env("ALE/Breakout-v5")
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8
    assert isinstance(terminated, bool) or isinstance(terminated, np.bool_)
    assert isinstance(truncated, bool) or isinstance(truncated, np.bool_)
    assert np.isfinite(float(reward))


@pytest.mark.skipif(not _ALE_AVAILABLE, reason="ALE not installed")
def test_make_atari_env_50_step_smoke() -> None:
    env = make_atari_env("ALE/Breakout-v5")
    env.reset(seed=0)
    n_outer = 0
    for _ in range(50):
        action = int(env.action_space.sample())
        obs, _, terminated, truncated, _ = env.step(action)
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == np.uint8
        n_outer += 1
        if terminated or truncated:
            env.reset()
    assert n_outer == 50
