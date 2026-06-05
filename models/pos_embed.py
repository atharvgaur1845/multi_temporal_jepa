"""Positional encodings: 2D spatial (for token grid) + 1D temporal from day-of-year (DOY).

The temporal encoding is the part that is *non-standard* and matters for this project.
Because PASTIS acquisitions are irregularly spaced, we encode the actual DOY of each frame
(not its integer position), so that a horizon of "Δ acquisitions" carries a real notion of
elapsed time and the model can generalize across different cadences/seasons.
"""
from __future__ import annotations


def build_2d_sincos_pos_embed(grid_hw, dim):
    """Fixed 2D sine-cosine positional embedding for an (H',W') token grid.

    Math: split `dim` in half, encode the row index and column index each with a standard
    1D sin/cos table, concatenate. Returns (N, dim) with N = H'*W'.

    TODO: implement (this is standard; do it once, correctly, and unit-test the shape).
    """
    raise NotImplementedError("M1")


def doy_sincos_pos_embed(dates, dim, period=366):
    """Date-based (day-of-year) sine-cosine temporal embedding.

    Math/spec
        For each frame with day-of-year d, build a sin/cos table over frequencies, but use
        d / period as the phase (so the embedding is periodic over a year and reflects real
        elapsed time between irregular acquisitions).
        Input dates: (B, T) long. Output: (B, T, dim).

    Why DOY and not frame index: Common Mistake #4 — index-based positions make Δ meaningless
    on irregular cadence.

    TODO: implement; make sure padded frames (per pad_mask) don't get a spurious encoding.
    """
    raise NotImplementedError("M1/M2")
