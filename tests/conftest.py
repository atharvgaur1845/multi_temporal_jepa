"""Shared fixtures. Synthetic tensors only — no PASTIS download needed for the logic tests."""
import pytest
import torch


@pytest.fixture
def grid_hw():
    return (8, 8)  # 128/16 token grid


@pytest.fixture
def collapsed_embeddings():
    """Magnitude collapse: a (near-)constant batch of embeddings."""
    g = torch.Generator().manual_seed(0)
    return torch.zeros(64, 32) + 1e-6 * torch.randn(64, 32, generator=g)


@pytest.fixture
def healthy_embeddings():
    """Full-rank, high-variance embeddings == healthy representation."""
    g = torch.Generator().manual_seed(1)
    return torch.randn(64, 32, generator=g)
