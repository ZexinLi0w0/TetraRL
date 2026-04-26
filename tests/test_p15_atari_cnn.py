"""Unit tests for the NatureCNN backbone + uint8 replay buffer + Atari obs wiring."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from tetrarl.morl.algos import (
    A2CAlgo,
    C51Algo,
    DDQNAlgo,
    DQNAlgo,
    NatureCNN,
    PPOAlgo,
    ReplayBuffer,
)

# --- NatureCNN encoder ----------------------------------------------------


def test_nature_cnn_forward_shape_uint8() -> None:
    """(B, 4, 84, 84) uint8 input -> (B, 512) float32 output."""
    cnn = NatureCNN(in_channels=4, out_dim=512)
    x = torch.zeros(3, 4, 84, 84, dtype=torch.uint8)
    y = cnn(x)
    assert y.shape == (3, 512)
    assert y.dtype == torch.float32


def test_nature_cnn_forward_shape_float() -> None:
    """Float input also works (already-normalized path)."""
    cnn = NatureCNN(in_channels=4, out_dim=512)
    x = torch.zeros(2, 4, 84, 84, dtype=torch.float32)
    y = cnn(x)
    assert y.shape == (2, 512)
    assert y.dtype == torch.float32


def test_nature_cnn_uint8_normalization_matches_float() -> None:
    """A constant 255 uint8 image == constant 1.0 float image after / 255."""
    cnn = NatureCNN(in_channels=4, out_dim=512)
    cnn.eval()
    x_u8 = torch.full((1, 4, 84, 84), 255, dtype=torch.uint8)
    x_f = torch.ones(1, 4, 84, 84, dtype=torch.float32) * 255.0
    with torch.no_grad():
        y_u8 = cnn(x_u8)
        y_f = cnn(x_f)
    assert torch.allclose(y_u8, y_f, atol=1e-5)


def test_nature_cnn_in_channels_one() -> None:
    """Single-channel grayscale input also works."""
    cnn = NatureCNN(in_channels=1, out_dim=512)
    x = torch.zeros(2, 1, 84, 84, dtype=torch.uint8)
    y = cnn(x)
    assert y.shape == (2, 512)


def test_nature_cnn_has_three_conv_layers() -> None:
    """Mnih 2015 architecture has exactly three Conv2d layers."""
    cnn = NatureCNN(in_channels=4, out_dim=512)
    convs = [m for m in cnn.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 3


# --- ReplayBuffer dtype --------------------------------------------------


def test_replay_buffer_default_dtype_float32() -> None:
    """Default ReplayBuffer obs dtype stays float32 (CartPole backward compat)."""
    buf = ReplayBuffer(capacity=8, obs_shape=(4,))
    assert buf.s.dtype == np.float32
    assert buf.sn.dtype == np.float32


def test_replay_buffer_uint8_dtype_storage() -> None:
    """obs_dtype=np.uint8 stores frames as uint8 and uses ~1/4 the bytes."""
    cap = 100
    buf_u8 = ReplayBuffer(capacity=cap, obs_shape=(4, 84, 84), obs_dtype=np.uint8)
    buf_f32 = ReplayBuffer(capacity=cap, obs_shape=(4, 84, 84), obs_dtype=np.float32)
    assert buf_u8.s.dtype == np.uint8
    assert buf_u8.sn.dtype == np.uint8
    # uint8 = 1 byte/element, float32 = 4 bytes/element.
    assert buf_u8.s.nbytes == buf_f32.s.nbytes // 4


def test_replay_buffer_uint8_push_and_sample_preserves_values() -> None:
    """A pushed uint8 frame round-trips through sample() with bytes preserved."""
    buf = ReplayBuffer(capacity=4, obs_shape=(4, 84, 84), obs_dtype=np.uint8)
    s = (np.random.randint(0, 256, size=(4, 84, 84))).astype(np.uint8)
    sn = (np.random.randint(0, 256, size=(4, 84, 84))).astype(np.uint8)
    buf.push(s, 1, 1.0, sn, 0.0)
    s_b, a_b, r_b, sn_b, d_b = buf.sample(1)
    assert s_b.dtype == np.uint8
    assert sn_b.dtype == np.uint8
    assert np.array_equal(s_b[0], s)
    assert np.array_equal(sn_b[0], sn)


# --- DQN/DDQN/C51 with Atari obs -----------------------------------------


@pytest.mark.parametrize("AlgoCls", [DQNAlgo, DDQNAlgo, C51Algo])
def test_off_policy_atari_builds_cnn_backbone(AlgoCls) -> None:
    """Atari obs_shape (4,84,84) builds a CNN-based Q-net."""
    algo = AlgoCls(obs_shape=(4, 84, 84), n_actions=4, seed=0)
    has_conv = any(isinstance(m, nn.Conv2d) for m in algo.q.modules())
    assert has_conv, f"{AlgoCls.__name__} did not build a Conv2d backbone for Atari"


@pytest.mark.parametrize("AlgoCls", [DQNAlgo, DDQNAlgo, C51Algo])
def test_off_policy_atari_replay_uses_uint8(AlgoCls) -> None:
    """Replay buffer for Atari off-policy algos stores uint8 obs."""
    algo = AlgoCls(obs_shape=(4, 84, 84), n_actions=4, seed=0)
    assert algo.buffer.s.dtype == np.uint8
    assert algo.buffer.sn.dtype == np.uint8


@pytest.mark.parametrize("AlgoCls", [DQNAlgo, DDQNAlgo, C51Algo, A2CAlgo, PPOAlgo])
def test_act_returns_valid_action_for_atari_uint8(AlgoCls) -> None:
    """All algos return a valid action index from a uint8 (4,84,84) frame."""
    algo = AlgoCls(obs_shape=(4, 84, 84), n_actions=4, seed=0)
    obs = np.zeros((4, 84, 84), dtype=np.uint8)
    a = algo.act(obs)
    assert isinstance(a, int)
    assert 0 <= a < 4


def test_c51_atari_distribution_shape_and_softmax() -> None:
    """C51 Atari distribution is (B, A, n_atoms) and rows sum to 1."""
    algo = C51Algo(obs_shape=(4, 84, 84), n_actions=4, seed=0)
    x = torch.zeros(2, 4, 84, 84, dtype=torch.uint8, device=algo.device)
    dist = algo._dist(x)
    assert dist.shape == (2, 4, algo.n_atoms)
    sums = dist.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# --- Update cycles produce finite losses ----------------------------------


def test_dqn_atari_observe_update_cycle_finite_loss() -> None:
    """DQN trains for a few uint8 Atari steps and produces a finite loss."""
    algo = DQNAlgo(
        obs_shape=(4, 84, 84),
        n_actions=4,
        seed=0,
        replay_capacity=64,
        batch_size=4,
        train_after=4,
    )
    rng = np.random.default_rng(0)
    last_metrics: dict = {}
    for _ in range(8):
        s = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
        sn = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
        a = algo.act(s)
        algo.observe(s, a, 1.0, sn, False)
        m = algo.update()
        if m:
            last_metrics = m
    assert "loss" in last_metrics
    assert np.isfinite(last_metrics["loss"])


def test_a2c_atari_rollout_update_cycle_finite_loss() -> None:
    """A2C rolls out for `rollout_steps` Atari frames and produces a finite loss."""
    algo = A2CAlgo(
        obs_shape=(4, 84, 84),
        n_actions=4,
        seed=0,
        rollout_steps=4,
        mini_batch_size=2,
    )
    rng = np.random.default_rng(0)
    s = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
    metrics: dict = {}
    for _ in range(4):
        a = algo.act(s)
        sn = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
        algo.observe(s, a, 1.0, sn, False)
        s = sn
    metrics = algo.update()
    assert "loss" in metrics
    assert np.isfinite(metrics["loss"])
