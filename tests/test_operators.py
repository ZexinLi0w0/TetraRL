"""Unit tests for the cosine-similarity envelope operator."""

import torch

from tetrarl.morl.operators import (
    action_selection,
    cosine_similarity_scalarization,
    envelope_operator,
)


class TestCosineSimilarityScalarization:

    def test_range(self):
        omega = torch.rand(32, 3)
        q = torch.randn(32, 3)
        sc = cosine_similarity_scalarization(omega, q)
        assert (sc >= -1.0 - 1e-6).all()
        assert (sc <= 1.0 + 1e-6).all()

    def test_identity(self):
        omega = torch.tensor([[1.0, 2.0, 3.0]])
        sc = cosine_similarity_scalarization(omega, omega)
        assert torch.allclose(sc, torch.ones(1), atol=1e-5)

    def test_anti_correlation(self):
        omega = torch.tensor([[1.0, 2.0, 3.0]])
        sc = cosine_similarity_scalarization(omega, -omega)
        assert torch.allclose(sc, -torch.ones(1), atol=1e-5)

    def test_orthogonal(self):
        omega = torch.tensor([[1.0, 0.0]])
        q = torch.tensor([[0.0, 1.0]])
        sc = cosine_similarity_scalarization(omega, q)
        assert torch.allclose(sc, torch.zeros(1), atol=1e-5)

    def test_batched(self):
        omega = torch.rand(16, 4)
        q = torch.randn(16, 4)
        sc = cosine_similarity_scalarization(omega, q)
        assert sc.shape == (16,)


class TestEnvelopeOperator:

    def test_positive_aligned(self):
        omega = torch.tensor([[0.5, 0.5]])
        q = torch.tensor([[2.0, 4.0]])
        result = envelope_operator(omega, q)
        sc = cosine_similarity_scalarization(omega, q)
        linear = (omega * q).sum(dim=-1)
        assert torch.allclose(result, sc * linear, atol=1e-5)

    def test_zero_q(self):
        omega = torch.tensor([[0.5, 0.5]])
        q = torch.tensor([[0.0, 0.0]])
        result = envelope_operator(omega, q)
        assert torch.allclose(result, torch.zeros(1), atol=1e-5)

    def test_batched(self):
        omega = torch.rand(8, 2)
        q = torch.randn(8, 2)
        result = envelope_operator(omega, q)
        assert result.shape == (8,)


class TestActionSelection:

    def test_selects_best_action(self):
        omega = torch.tensor([[0.8, 0.2]])
        q_all = torch.tensor([[[10.0, 1.0], [1.0, 10.0], [5.0, 5.0]]])
        selected = action_selection(omega, q_all)
        assert selected.item() == 0

    def test_batched(self):
        omega = torch.rand(4, 2)
        q_all = torch.randn(4, 5, 2)
        selected = action_selection(omega, q_all)
        assert selected.shape == (4,)
        assert (selected >= 0).all()
        assert (selected < 5).all()
