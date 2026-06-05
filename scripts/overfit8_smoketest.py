"""M1 HARD GATE: overfit 8 samples and prove the model is learning, not collapsing.

Because we skipped the CIFAR warm-up, this is your fast feedback loop. It must pass before you
train on full PASTIS. Two conditions, BOTH required:

    (1) LEARNING:    loss drops to near-0 on 8 fixed samples within a few hundred steps.
    (2) NO COLLAPSE: per-dim std stays bounded away from 0, effective rank stays high,
                     predictor/target variance ratio stays O(1).

A model can satisfy (1) by collapsing (predicting a constant that the EMA target also drifts
toward) — that is why (2) is non-negotiable. If loss->0 but std->0, you have collapse, not success.

TODO
    - take 8 fixed samples (one batch), no shuffling.
    - train the JEPA for K steps; every few steps print loss + engine.diagnostics.collapse_metrics.
    - assert final loss < tol AND per_dim_std > std_floor AND effective_rank > rank_floor.
    - exit non-zero on failure so it can gate CI.
"""
from __future__ import annotations


def main():
    raise NotImplementedError("M1 gate — implement after the JEPA core is in place")


if __name__ == "__main__":
    main()
