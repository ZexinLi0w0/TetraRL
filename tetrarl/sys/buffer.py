"""Pre-allocated, soft-truncation replay buffer.

All storage tensors are allocated once at construction and never resized.
Inserts wrap around as a ring; logical removal of the oldest items is done
by flipping bits in `valid_mask` (soft truncation), so the underlying
buffers are never reallocated.

Rationale: on Jetson Unified Memory (CPU + iGPU share the same pool),
PyTorch's caching allocator does not return freed blocks to the OS, and
re-allocating large replay buffers fragments the unified pool and can OOM
the iGPU under load. R^3 (Li, RTSS'23 Section 5) shows that pre-allocating
training-time tensors and using an index mask for "shrink" operations
avoids this fragmentation. This buffer follows that discipline: a fixed
storage budget set at construction, and `soft_truncate` that touches only
a boolean mask.
"""
from __future__ import annotations

from typing import Sequence

import torch


class ReplayBuffer:
    """Ring buffer with pre-allocated storage and soft-truncation via a mask."""

    def __init__(
        self,
        capacity: int,
        obs_shape: Sequence[int],
        act_shape: Sequence[int] = (),
        obs_dtype: torch.dtype = torch.float32,
        act_dtype: torch.dtype = torch.long,
        device: str | torch.device = "cpu",
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.act_shape = tuple(act_shape)
        self.obs_dtype = obs_dtype
        self.act_dtype = act_dtype
        self.device = torch.device(device)

        self.obs = torch.zeros((self.capacity, *self.obs_shape), dtype=obs_dtype, device=self.device)
        self.next_obs = torch.zeros((self.capacity, *self.obs_shape), dtype=obs_dtype, device=self.device)
        self.actions = torch.zeros((self.capacity, *self.act_shape), dtype=act_dtype, device=self.device)
        self.rewards = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.capacity,), dtype=torch.bool, device=self.device)
        self.valid_mask = torch.zeros((self.capacity,), dtype=torch.bool, device=self.device)

        self._head = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def memory_bytes(self) -> int:
        tensors = (self.obs, self.next_obs, self.actions, self.rewards, self.dones, self.valid_mask)
        return sum(t.element_size() * t.numel() for t in tensors)

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        idx = self._head
        self.obs[idx] = torch.as_tensor(obs, dtype=self.obs_dtype, device=self.device)
        self.next_obs[idx] = torch.as_tensor(next_obs, dtype=self.obs_dtype, device=self.device)
        self.actions[idx] = torch.as_tensor(action, dtype=self.act_dtype, device=self.device)
        self.rewards[idx] = float(reward)
        self.dones[idx] = bool(done)
        self.valid_mask[idx] = True

        self._head = (self._head + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def soft_truncate(self, n: int) -> int:
        n = int(n)
        if n <= 0 or self._size == 0:
            return 0
        n = min(n, self._size)
        tail = (self._head - self._size) % self.capacity
        for k in range(n):
            self.valid_mask[(tail + k) % self.capacity] = False
        self._size -= n
        return n

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        valid_idx = torch.nonzero(self.valid_mask, as_tuple=False).flatten()
        choice = torch.randint(
            low=0,
            high=valid_idx.numel(),
            size=(int(batch_size),),
            generator=generator,
            device=self.device,
        )
        idx = valid_idx[choice]
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }

    def clear(self) -> None:
        self._head = 0
        self._size = 0
        self.valid_mask.zero_()
