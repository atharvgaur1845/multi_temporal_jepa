"""Checkpoint save/load.

Save enough to RESUME and to EVAL later: model (incl. target encoder), optimizer, scaler,
scheduler, global step, RNG states, and the config. Reproducibility depends on this.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, scaler, scheduler, step, config, extra=None):
    """torch.save a dict with all state needed to resume + reproduce."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "config": config,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "extra": extra or {},
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, scheduler=None, map_location=None):
    """Restore state; return the global step. Handles eval-only loads (optimizer=None)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    rng = ckpt.get("rng")
    if rng is not None:
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            pass  # RNG restore is best-effort across machines
    return ckpt.get("step", 0)
