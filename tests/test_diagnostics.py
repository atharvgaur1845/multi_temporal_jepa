"""Collapse diagnostics must actually distinguish collapsed vs healthy representations.

Two DISTINCT failure modes, two metrics:
  - magnitude collapse (near-constant embeddings)  -> caught by per_dim_std
  - dimensional collapse (embeddings in a low-rank subspace) -> caught by effective_rank
A fixture must model the failure mode the metric is meant to catch.
"""
import torch

from engine.diagnostics import effective_rank, per_dim_std


def test_std_detects_collapse(collapsed_embeddings, healthy_embeddings):
    assert per_dim_std(collapsed_embeddings) < per_dim_std(healthy_embeddings)
    assert per_dim_std(collapsed_embeddings) < 0.1  # rule-of-thumb collapse floor


def test_effective_rank_detects_collapse(healthy_embeddings):
    # dimensional collapse: a rank-1 embedding (every row is a scaled copy of one direction).
    g = torch.Generator().manual_seed(0)
    low_rank = torch.randn(64, 1, generator=g) @ torch.randn(1, 32, generator=g)
    assert effective_rank(low_rank) < effective_rank(healthy_embeddings)
    assert effective_rank(low_rank) < 2.0  # rank-1 -> effective rank ~1
