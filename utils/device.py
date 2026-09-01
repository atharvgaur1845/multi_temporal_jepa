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
    dev = torch.device(name)
    # ALWAYS return an INDEXED cuda device. A bare torch.device("cuda") has
    # index=None, which torch.cuda.set_device() rejects outright:
    #     ValueError: Expected a torch.device with a specified index ... but got: cuda
    # and which makes per-device queries (max_memory_allocated) silently read GPU 0.
    # Under SLURM, --gres=gpu:1 sets CUDA_VISIBLE_DEVICES so the allocated card is
    # always index 0 — resolving "cuda" -> "cuda:0" is therefore correct there too.
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev
