"""Positional encodings: 2D spatial (for token grid) + 1D temporal from day-of-year (DOY).

The temporal encoding is the part that is *non-standard* and matters for this project.
Because PASTIS acquisitions are irregularly spaced, we encode the actual DOY of each frame
(not its integer position), so that a horizon of "Δ acquisitions" carries a real notion of
elapsed time and the model can generalize across different cadences/seasons.
"""
from __future__ import annotations

import torch


def _sincos_1d(positions, dim):
    """Standard 1D sin/cos table. positions: (...,) float; returns (..., dim).
    Even indices = sin, odd = cos, with geometric frequencies (Vaswani et al.)."""
    assert dim % 2 == 0, "dim must be even for sin/cos"
    device = positions.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0)) / half)
    )  # (half,)
    args = positions.float().unsqueeze(-1) * freqs  # (..., half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (..., dim)


def build_2d_sincos_pos_embed(grid_hw, dim):
    """Fixed 2D sine-cosine positional embedding for an (H',W') token grid.

    Split `dim` in half: encode the row index and column index each with a 1D sin/cos table,
    concatenate. Returns (N, dim) with N = H'*W', flattened row-major to match PatchEmbed.
    """
    H, W = grid_hw
    assert dim % 2 == 0, "dim must be even"
    rows = torch.arange(H)
    cols = torch.arange(W)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    emb_r = _sincos_1d(rr.flatten(), dim // 2)  # (N, dim/2)
    emb_c = _sincos_1d(cc.flatten(), dim // 2)  # (N, dim/2)
    return torch.cat([emb_r, emb_c], dim=-1)    # (N, dim)


def doy_sincos_pos_embed(dates, dim, period=366, pad_mask=None):
    """Date-based (day-of-year) sine-cosine temporal embedding.

    For each frame with day-of-year d, build a sin/cos table using the phase d/period so the
    embedding is periodic over a year and reflects real elapsed time between irregular
    acquisitions (Common Mistake #4: index-based positions make Δ meaningless).

    Args
        dates : (B, T) long DOY in [1, 366]   (padded frames carry 0)
        dim   : embedding width (even)
        pad_mask : (B, T) bool; if given, padded frames get a zero embedding so they carry no
                   spurious positional signal.
    Returns: (B, T, dim).
    """
    phase = dates.float() / period * (2 * torch.pi)  # scale into radians over the year
    emb = _sincos_1d(phase, dim)                      # (B, T, dim)
    if pad_mask is not None:
        emb = emb * pad_mask.unsqueeze(-1).float()
    return emb
