"""Pure-PyTorch GCN feature extractor for DAG observations.

Implements the symmetric-normalized GCN from Kipf & Welling (2017):
  H' = sigma(D^-1/2 (A + I) D^-1/2 H W)
two layers + mean (or sum) pooling -> graph-level embedding.

No torch_geometric dependency. Edge index is a (2, E) LongTensor in
the standard COO format. Batches are encoded by a `batch` LongTensor of
shape (N,) marking which graph each node belongs to (mirrors the
torch_geometric convention so we can drop in PyG later if we want).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Build a dense D^-1/2 (A + I) D^-1/2 matrix of shape (N, N).

    edge_index: (2, E) LongTensor, treated as undirected (we add both
    directions). Self-loops are added.
    """
    device = edge_index.device if edge_index.numel() > 0 else torch.device("cpu")
    A = torch.zeros(num_nodes, num_nodes, device=device)
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        A[src, dst] = 1.0
        A[dst, src] = 1.0  # symmetrize
    A = A + torch.eye(num_nodes, device=A.device)  # self-loops
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.where(
        deg > 0,
        deg.pow(-0.5),
        torch.zeros_like(deg),
    )
    return deg_inv_sqrt.unsqueeze(1) * A * deg_inv_sqrt.unsqueeze(0)


class GCNLayer(nn.Module):
    """Single GCN layer: H' = norm_adj @ H @ W."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        return norm_adj @ self.lin(x)


class GCNFeatureExtractor(nn.Module):
    """Two-layer GCN -> graph-level embedding.

    Input:
        node_features: (N, F_in)
        edge_index:    (2, E) long
        batch:         (N,) long, batch index for each node. If None, treats
                       all nodes as a single graph (batch idx 0).
    Output:
        graph_emb:     (B, out_dim) where B = number of graphs in the batch
    Pooling: mean by default; pass pooling="sum" or "max" for variants.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, pooling: str = "mean"):
        super().__init__()
        if pooling not in ("mean", "sum", "max"):
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, node_features, edge_index, batch=None):
        n = node_features.shape[0]
        if batch is None:
            batch = torch.zeros(n, dtype=torch.long, device=node_features.device)
        # 1. block-diagonal normalized adjacency for the merged batch graph
        norm_adj = normalized_adjacency(edge_index, n)
        # 2. two GCN layers with ReLU between
        h = F.relu(self.gcn1(node_features, norm_adj))
        h = self.gcn2(h, norm_adj)
        # 3. pool per-graph
        num_graphs = int(batch.max().item()) + 1 if n > 0 else 1
        out_dim = h.shape[1]
        if self.pooling == "sum":
            out = torch.zeros(num_graphs, out_dim, device=h.device, dtype=h.dtype)
            out.index_add_(0, batch, h)
            return out
        if self.pooling == "mean":
            out = torch.zeros(num_graphs, out_dim, device=h.device, dtype=h.dtype)
            out.index_add_(0, batch, h)
            counts = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
            counts.index_add_(0, batch, torch.ones(n, device=h.device, dtype=h.dtype))
            counts = counts.clamp(min=1.0).unsqueeze(1)
            return out / counts
        # max pooling: small B, loop is fine
        out = torch.full(
            (num_graphs, out_dim),
            float("-inf"),
            device=h.device,
            dtype=h.dtype,
        )
        for g in range(num_graphs):
            mask = batch == g
            if mask.any():
                out[g] = h[mask].max(dim=0).values
            else:
                out[g] = torch.zeros(out_dim, device=h.device, dtype=h.dtype)
        return out

    @property
    def out_dim(self) -> int:
        return self.gcn2.lin.out_features


def make_single_graph_batch(num_nodes: int, device="cpu") -> torch.Tensor:
    """Helper: returns a batch tensor of zeros for a single-graph case."""
    return torch.zeros(num_nodes, dtype=torch.long, device=device)
