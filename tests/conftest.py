"""Shared fixtures. Synthetic tensors only — no PASTIS download needed for the logic tests."""
import pytest
import torch


@pytest.fixture
def grid_hw():
    return (8, 8)  # 128/16 token grid


@pytest.fixture
def collapsed_embeddings():
    """A (near-)constant batch of embeddings == representation collapse."""
    z = torch.zeros(64, 32)
    return z + 1e-6 * torch.randn_like(z)


@pytest.fixture
def healthy_embeddings():
    """Full-rank, high-variance embeddings == healthy representation."""
    return torch.randn(64, 32)
