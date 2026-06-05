"""Checkpoint save/load (thin skeleton — implement when you set up the train loop).

Save enough to RESUME and to EVAL later: model (incl. target encoder), optimizer, scaler,
scheduler, global step, RNG states, and the config. Reproducibility depends on this.
"""
from __future__ import annotations


def save_checkpoint(path, model, optimizer, scaler, scheduler, step, config, extra=None):
    """TODO: torch.save a dict with all state needed to resume + reproduce."""
    raise NotImplementedError("M1")


def load_checkpoint(path, model, optimizer=None, scaler=None, scheduler=None, map_location=None):
    """TODO: restore state; return the global step. Handle eval-only loads (optimizer=None)."""
    raise NotImplementedError("M1")
