"""Tests for tetrarl.morl.native.masking."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tetrarl.morl.native.masking import (
    ActionMask,
    DeadlineMask,
    NoOpMask,
    apply_logit_mask,
)


class TestNoOpMask:
    def test_all_allowed(self):
        mask = NoOpMask()
        out = mask.compute(state=np.zeros(3), act_dim=5)
        assert out.shape == (5,)
        assert out.dtype == bool
        assert out.all()

    def test_as_tensor_returns_bool(self):
        mask = NoOpMask()
        t = mask.as_tensor(state=np.zeros(3), act_dim=4, device="cpu")
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.bool
        assert t.shape == (4,)
        assert bool(t.all())


class TestDeadlineMask:
    def test_rejects_invalid_freq_scale(self):
        with pytest.raises(ValueError):
            DeadlineMask(freq_scale=[1.0, 0.0, 2.0], deadline_ms=5.0)
        with pytest.raises(ValueError):
            DeadlineMask(freq_scale=[1.0, -1.0], deadline_ms=5.0)
        with pytest.raises(ValueError):
            DeadlineMask(freq_scale=[], deadline_ms=5.0)

    def test_dim_mismatch_raises(self):
        m = DeadlineMask(freq_scale=[1.0, 2.0, 4.0], deadline_ms=100.0)
        with pytest.raises(ValueError):
            m.compute(state=np.zeros(2), act_dim=4)

    def test_all_allowed_when_under_deadline(self):
        m = DeadlineMask(
            freq_scale=[1.0, 2.0, 4.0],
            deadline_ms=1000.0,
            initial_latency_ms=1.0,
        )
        out = m.compute(state=np.zeros(2), act_dim=3)
        assert out.dtype == bool
        assert out.tolist() == [True, True, True]

    def test_masks_slow_actions_when_tight(self):
        m = DeadlineMask(
            freq_scale=[1.0, 2.0, 4.0],
            deadline_ms=5.0,
            initial_latency_ms=10.0,
        )
        out = m.compute(state=np.zeros(2), act_dim=3)
        assert out.tolist() == [False, True, True]

    def test_keeps_fastest_when_all_miss(self):
        m = DeadlineMask(
            freq_scale=[1.0, 2.0, 4.0],
            deadline_ms=0.1,
            initial_latency_ms=10.0,
        )
        out = m.compute(state=np.zeros(2), act_dim=3)
        assert out.any()
        # fastest = highest freq_scale = index 2
        assert out[int(np.argmax(m.freq_scale))]

    def test_ema_update(self):
        m = DeadlineMask(
            freq_scale=[1.0, 2.0],
            deadline_ms=5.0,
            ema_alpha=0.5,
            initial_latency_ms=10.0,
        )
        m.update_latency(20.0)
        assert m.latency_ms == pytest.approx(15.0)
        m.update_latency(20.0)
        assert m.latency_ms == pytest.approx(17.5)


class TestApplyLogitMask:
    def test_masked_logits_become_very_negative(self):
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        out = apply_logit_mask(logits, mask)
        assert out[0].item() == pytest.approx(1.0)
        assert out[2].item() == pytest.approx(3.0)
        assert out[1].item() < -1e8
        assert out[3].item() < -1e8

    def test_softmax_after_mask_zeros_disallowed(self):
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        out = apply_logit_mask(logits, mask)
        probs = torch.softmax(out, dim=-1)
        assert probs[1].item() == pytest.approx(0.0, abs=1e-6)
        assert probs[3].item() == pytest.approx(0.0, abs=1e-6)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_batched_mask(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[True, False, True], [False, True, True]])
        out = apply_logit_mask(logits, mask)
        assert out.shape == (2, 3)
        assert out[0, 0].item() == pytest.approx(1.0)
        assert out[0, 1].item() < -1e8
        assert out[1, 0].item() < -1e8
        assert out[1, 2].item() == pytest.approx(6.0)

    def test_accepts_int_mask_via_cast(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([1, 0, 1], dtype=torch.uint8)
        out = apply_logit_mask(logits, mask)
        assert out[0].item() == pytest.approx(1.0)
        assert out[1].item() < -1e8
        assert out[2].item() == pytest.approx(3.0)


def test_action_mask_is_abstract():
    with pytest.raises(TypeError):
        ActionMask()
