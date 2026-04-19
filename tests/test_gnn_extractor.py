"""Tests for the pure-PyTorch GCN feature extractor."""
from __future__ import annotations

import pytest
import torch

from tetrarl.morl.native.gnn_extractor import (
    GCNFeatureExtractor,
    GCNLayer,
    make_single_graph_batch,
    normalized_adjacency,
)


def test_normalized_adjacency_self_loops():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    A = normalized_adjacency(edge_index, num_nodes=3)
    assert A.shape == (3, 3)
    # self-loops -> diagonal must be non-zero
    assert torch.all(torch.diagonal(A) > 0)
    # symmetric
    assert torch.allclose(A, A.t(), atol=1e-6)


def test_normalized_adjacency_isolated_node():
    # 2 edges among 3 nodes; node 2 is isolated
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    A = normalized_adjacency(edge_index, num_nodes=3)
    assert A.shape == (3, 3)
    assert torch.isfinite(A).all()
    # isolated node still has its self-loop entry contributing
    assert A[2, 2] > 0
    # row 2 / col 2 should otherwise be zero (no neighbors)
    off_diag_row = A[2, [0, 1]]
    assert torch.allclose(off_diag_row, torch.zeros(2), atol=1e-6)


def test_gcn_layer_forward_shape():
    layer = GCNLayer(4, 8)
    h = torch.randn(3, 4)
    norm_adj = torch.eye(3)
    out = layer(h, norm_adj)
    assert out.shape == (3, 8)


def test_extractor_single_graph_shape():
    torch.manual_seed(0)
    ext = GCNFeatureExtractor(in_dim=4, hidden_dim=8, out_dim=16)
    h = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out_none = ext(h, edge_index, batch=None)
    assert out_none.shape == (1, 16)
    batch = torch.zeros(5, dtype=torch.long)
    out_zero = ext(h, edge_index, batch=batch)
    assert out_zero.shape == (1, 16)
    assert torch.allclose(out_none, out_zero, atol=1e-6)


def test_extractor_batched_shape():
    torch.manual_seed(0)
    ext = GCNFeatureExtractor(in_dim=4, hidden_dim=8, out_dim=16)
    # graph A: 3 nodes (ids 0,1,2), edges 0-1, 1-2
    # graph B: 4 nodes (ids 3,4,5,6), edges 3-4, 4-5, 5-6
    h = torch.randn(7, 4)
    edge_index = torch.tensor(
        [[0, 1, 3, 4, 5], [1, 2, 4, 5, 6]],
        dtype=torch.long,
    )
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    out = ext(h, edge_index, batch=batch)
    assert out.shape == (2, 16)


def test_extractor_invalid_pooling_raises():
    with pytest.raises(ValueError):
        GCNFeatureExtractor(in_dim=4, hidden_dim=8, out_dim=16, pooling="bogus")


def test_extractor_supports_pooling_modes():
    torch.manual_seed(0)
    h = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    for pool in ("mean", "sum", "max"):
        ext = GCNFeatureExtractor(in_dim=4, hidden_dim=8, out_dim=16, pooling=pool)
        out = ext(h, edge_index)
        assert out.shape == (1, 16), f"pooling={pool} shape mismatch"


def test_extractor_grad_flow():
    torch.manual_seed(0)
    ext = GCNFeatureExtractor(in_dim=4, hidden_dim=8, out_dim=16)
    h = torch.randn(5, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out = ext(h, edge_index)
    out.sum().backward()
    assert ext.gcn1.lin.weight.grad is not None
    assert torch.isfinite(ext.gcn1.lin.weight.grad).all()


def test_extractor_deterministic():
    torch.manual_seed(0)
    ext = GCNFeatureExtractor(in_dim=4, hidden_dim=8, out_dim=16)
    h = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out1 = ext(h, edge_index)
    out2 = ext(h, edge_index)
    assert torch.allclose(out1, out2, atol=1e-7)


def test_make_single_graph_batch():
    b = make_single_graph_batch(7)
    assert b.shape == (7,)
    assert b.dtype == torch.long
    assert torch.all(b == 0)
