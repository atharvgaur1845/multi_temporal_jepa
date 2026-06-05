"""Collapse diagnostics must actually distinguish collapsed vs healthy representations."""
from engine.diagnostics import effective_rank, per_dim_std


def test_std_detects_collapse(collapsed_embeddings, healthy_embeddings):
    assert per_dim_std(collapsed_embeddings) < per_dim_std(healthy_embeddings)
    assert per_dim_std(collapsed_embeddings) < 0.1  # rule-of-thumb collapse floor


def test_effective_rank_detects_collapse(collapsed_embeddings, healthy_embeddings):
    assert effective_rank(collapsed_embeddings) < effective_rank(healthy_embeddings)
