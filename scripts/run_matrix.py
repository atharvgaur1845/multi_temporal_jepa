"""Drive the full experiment matrix (M5) and log cost honestly.

Matrix (full, as specified):
    objectives : {temporal_jepa, spatial_jepa, mae, byol, simclr}
    horizon    : Δ ∈ {1, 2, 4, 8}                 (temporal_jepa only)
    ablations  : predictor depth {1,2,4,6}, embed dim {128,256,512,768}

Each cell = (pretrain -> freeze -> {linear_probe mIoU, k-NN, few-shot}) with a fixed seed and a
GpuHourMeter. THIS IS GPU-WEEKS on one card — order cells to get signal early, and:

    IMPORTANT: if you cap/skip cells for budget, LOG exactly which cells were skipped
    (Common Mistake / plan §5 M5). A matrix that silently drops cells looks complete but isn't.

TODO
    - enumerate the matrix from configs/exp/*.yaml.
    - for each cell: set seed, pretrain, run eval suite, record metrics + gpu_hours to a CSV.
    - print a summary table; explicitly list skipped cells.
"""
from __future__ import annotations


def main():
    raise NotImplementedError("M5")


if __name__ == "__main__":
    main()
