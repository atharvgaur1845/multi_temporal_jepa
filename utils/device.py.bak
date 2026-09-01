"""Single source of truth for the compute device.

Set the GPU in ONE place — `device:` in configs/model/tjepa.yaml (e.g. `cuda`, `cuda:1`,
`cuda:2`, or `cpu`) — or override per-run with `--device`. Every entry point resolves through
`resolve_device` so you never hard-code a GPU index anywhere else.
"""
from __future__ import annotations

import torch


def resolve_device(name=None):
    """Return a torch.device for `name` ('cuda', 'cuda:1', 'cpu', ...).

    Falls back to CPU (with a warning) if CUDA is requested but unavailable, so the same config
    runs on a CPU-only box. Default (name=None) is 'cuda' when available else 'cpu'.
    """
    if name is None:
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] {name} requested but CUDA unavailable — falling back to cpu.")
        return torch.device("cpu")
    return torch.device(name)
