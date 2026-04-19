"""Tests for the pre-allocated soft-truncation replay buffer.

The buffer must:
  * Pre-allocate `capacity` slots at construction (no runtime malloc on add).
  * Use a boolean index_mask for "soft truncation" (logical removal of the
    oldest items) rather than physical re-allocation.
  * Behave as a ring on overflow (oldest physical slot overwritten).
  * Sample uniformly from valid (mask=True) slots only.

These properties matter on Jetson Unified Memory where PyTorch
re-allocation causes fragmentation (R^3, Li RTSS'23 Section 5).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tetrarl.sys.buffer import ReplayBuffer


def _make(capacity: int = 8, obs_dim: int = 4):
    return ReplayBuffer(
        capacity=capacity,
        obs_shape=(obs_dim,),
        act_shape=(),
        obs_dtype=torch.float32,
        act_dtype=torch.long,
        device="cpu",
    )


def test_init_preallocates_storage_tensors():
    buf = _make(capacity=8, obs_dim=4)
    assert buf.obs.shape == (8, 4)
    assert buf.next_obs.shape == (8, 4)
    assert buf.actions.shape == (8,)
    assert buf.rewards.shape == (8,)
    assert buf.dones.shape == (8,)
    assert buf.valid_mask.shape == (8,)
    assert len(buf) == 0
    assert not buf.valid_mask.any().item()


def test_init_memory_is_fixed():
    buf = _make(capacity=16, obs_dim=4)
    initial_bytes = buf.memory_bytes()
    for i in range(5):
        buf.add(
            obs=np.zeros(4, dtype=np.float32),
            action=i,
            reward=float(i),
            next_obs=np.ones(4, dtype=np.float32),
            done=False,
        )
    assert buf.memory_bytes() == initial_bytes


def test_add_writes_values_and_advances_size():
    buf = _make(capacity=4, obs_dim=2)
    buf.add(
        obs=np.array([1.0, 2.0], dtype=np.float32),
        action=3,
        reward=0.5,
        next_obs=np.array([4.0, 5.0], dtype=np.float32),
        done=True,
    )
    assert len(buf) == 1
    assert buf.valid_mask[0].item() is True
    assert torch.allclose(buf.obs[0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(buf.next_obs[0], torch.tensor([4.0, 5.0]))
    assert buf.actions[0].item() == 3
    assert buf.rewards[0].item() == pytest.approx(0.5)
    assert buf.dones[0].item() is True


def test_add_beyond_capacity_overwrites_oldest():
    buf = _make(capacity=3, obs_dim=1)
    for r in range(3):
        buf.add(
            obs=np.array([float(r)], dtype=np.float32),
            action=r,
            reward=float(r),
            next_obs=np.array([float(r)], dtype=np.float32),
            done=False,
        )
    assert len(buf) == 3
    buf.add(
        obs=np.array([99.0], dtype=np.float32),
        action=99,
        reward=99.0,
        next_obs=np.array([99.0], dtype=np.float32),
        done=False,
    )
    assert len(buf) == 3
    assert buf.actions[0].item() == 99
    assert buf.rewards[0].item() == pytest.approx(99.0)


def test_soft_truncate_removes_oldest_via_mask_only():
    buf = _make(capacity=8, obs_dim=2)
    obs_storage_id = id(buf.obs)
    rewards_storage_id = id(buf.rewards)

    for i in range(5):
        buf.add(
            obs=np.array([float(i), 0.0], dtype=np.float32),
            action=i,
            reward=float(i),
            next_obs=np.array([float(i), 1.0], dtype=np.float32),
            done=False,
        )
    assert len(buf) == 5

    n_truncated = buf.soft_truncate(2)
    assert n_truncated == 2
    assert len(buf) == 3

    assert id(buf.obs) == obs_storage_id
    assert id(buf.rewards) == rewards_storage_id

    assert buf.valid_mask[0].item() is False
    assert buf.valid_mask[1].item() is False
    assert buf.valid_mask[2].item() is True
    assert buf.valid_mask[3].item() is True
    assert buf.valid_mask[4].item() is True


def test_soft_truncate_more_than_size_clamps():
    buf = _make(capacity=4, obs_dim=1)
    for i in range(2):
        buf.add(
            obs=np.array([float(i)], dtype=np.float32),
            action=i,
            reward=float(i),
            next_obs=np.array([float(i)], dtype=np.float32),
            done=False,
        )
    n = buf.soft_truncate(99)
    assert n == 2
    assert len(buf) == 0
    assert not buf.valid_mask.any().item()


def test_sample_returns_only_valid_items():
    buf = _make(capacity=8, obs_dim=1)
    for i in range(5):
        buf.add(
            obs=np.array([float(i)], dtype=np.float32),
            action=i,
            reward=float(i),
            next_obs=np.array([float(i)], dtype=np.float32),
            done=False,
        )
    buf.soft_truncate(2)
    rng = torch.Generator()
    rng.manual_seed(0)
    batch = buf.sample(64, generator=rng)
    assert batch["rewards"].shape == (64,)
    seen_rewards = set(batch["rewards"].tolist())
    assert seen_rewards.issubset({2.0, 3.0, 4.0})


def test_sample_shapes():
    buf = _make(capacity=8, obs_dim=4)
    for i in range(8):
        buf.add(
            obs=np.zeros(4, dtype=np.float32),
            action=i,
            reward=float(i),
            next_obs=np.zeros(4, dtype=np.float32),
            done=False,
        )
    batch = buf.sample(3)
    assert batch["obs"].shape == (3, 4)
    assert batch["next_obs"].shape == (3, 4)
    assert batch["actions"].shape == (3,)
    assert batch["rewards"].shape == (3,)
    assert batch["dones"].shape == (3,)


def test_sample_empty_raises():
    buf = _make(capacity=4, obs_dim=2)
    with pytest.raises(ValueError):
        buf.sample(1)


def test_clear_resets_state_without_realloc():
    buf = _make(capacity=4, obs_dim=2)
    obs_id = id(buf.obs)
    for i in range(3):
        buf.add(
            obs=np.zeros(2, dtype=np.float32),
            action=i,
            reward=float(i),
            next_obs=np.zeros(2, dtype=np.float32),
            done=False,
        )
    assert len(buf) == 3
    buf.clear()
    assert len(buf) == 0
    assert not buf.valid_mask.any().item()
    assert id(buf.obs) == obs_id


def test_ring_then_truncate_consistency():
    """After overflow, soft_truncate still removes the *logically* oldest."""
    buf = _make(capacity=4, obs_dim=1)
    for i in range(4):
        buf.add(
            obs=np.array([float(i)], dtype=np.float32),
            action=i,
            reward=float(i),
            next_obs=np.array([float(i)], dtype=np.float32),
            done=False,
        )
    for r in (10.0, 11.0):
        buf.add(
            obs=np.array([r], dtype=np.float32),
            action=int(r),
            reward=r,
            next_obs=np.array([r], dtype=np.float32),
            done=False,
        )
    assert len(buf) == 4
    buf.soft_truncate(1)
    assert len(buf) == 3
    assert buf.valid_mask[2].item() is False
    assert buf.valid_mask[3].item() is True
    assert buf.valid_mask[0].item() is True
    assert buf.valid_mask[1].item() is True
