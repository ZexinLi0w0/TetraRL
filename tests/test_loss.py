"""Unit tests for multi-objective loss functions."""

import pytest
import torch

from tetrarl.morl.loss import (
    cosine_similarity_envelope_loss,
    directional_regularization,
)


class TestCosineEnvelopeLoss:

    def test_zero_when_q_equals_target(self):
        omega = torch.tensor([[0.6, 0.4]])
        q = torch.tensor([[3.0, 2.0]])
        loss = cosine_similarity_envelope_loss(omega, q, q.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_q_differs(self):
        omega = torch.tensor([[0.5, 0.5]])
        q = torch.tensor([[1.0, 1.0]])
        target_q = torch.tensor([[5.0, 5.0]])
        loss = cosine_similarity_envelope_loss(omega, q, target_q)
        assert loss.item() > 0.0

    def test_gradient_flows_through_q(self):
        omega = torch.tensor([[0.5, 0.5]])
        q = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target_q = torch.tensor([[3.0, 4.0]])
        loss = cosine_similarity_envelope_loss(omega, q, target_q)
        loss.backward()
        assert q.grad is not None
        assert not torch.all(q.grad == 0)

    def test_no_gradient_through_target(self):
        omega = torch.tensor([[0.5, 0.5]])
        q = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target_q = torch.tensor([[3.0, 4.0]], requires_grad=True)
        loss = cosine_similarity_envelope_loss(omega, q, target_q)
        loss.backward()
        assert target_q.grad is None or torch.all(target_q.grad == 0)

    def test_batched(self):
        omega = torch.rand(16, 3)
        omega = omega / omega.sum(dim=-1, keepdim=True)
        q = torch.randn(16, 3)
        target_q = torch.randn(16, 3)
        loss = cosine_similarity_envelope_loss(omega, q, target_q)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_scalar_output(self):
        omega = torch.rand(8, 2)
        q = torch.randn(8, 2)
        target_q = torch.randn(8, 2)
        loss = cosine_similarity_envelope_loss(omega, q, target_q)
        assert loss.dim() == 0


class TestDirectionalRegularization:

    def test_zero_when_aligned(self):
        omega = torch.tensor([[1.0, 2.0]])
        q = torch.tensor([[2.0, 4.0]])  # same direction
        reg = directional_regularization(omega, q)
        assert reg.item() == pytest.approx(0.0, abs=1e-4)

    def test_positive_when_misaligned(self):
        omega = torch.tensor([[1.0, 0.0]])
        q = torch.tensor([[0.0, 1.0]])  # orthogonal
        reg = directional_regularization(omega, q)
        assert reg.item() > 0.0

    def test_max_at_anti_aligned(self):
        omega = torch.tensor([[1.0, 0.0]])
        q_ortho = torch.tensor([[0.0, 1.0]])
        q_anti = torch.tensor([[-1.0, 0.0]])
        reg_ortho = directional_regularization(omega, q_ortho)
        reg_anti = directional_regularization(omega, q_anti)
        assert reg_anti.item() > reg_ortho.item()

    def test_gradient_flows(self):
        omega = torch.tensor([[0.5, 0.5]])
        q = torch.tensor([[1.0, -1.0]], requires_grad=True)
        reg = directional_regularization(omega, q)
        reg.backward()
        assert q.grad is not None

    def test_batched(self):
        omega = torch.rand(32, 4)
        q = torch.randn(32, 4)
        reg = directional_regularization(omega, q)
        assert reg.shape == ()
        assert reg.item() >= 0.0

    def test_range(self):
        omega = torch.rand(100, 3)
        q = torch.randn(100, 3)
        reg = directional_regularization(omega, q)
        # 1 - cos_sim output is in [0, 2], mean should be in [0, 2]
        assert 0.0 <= reg.item() <= 2.0 + 1e-6
