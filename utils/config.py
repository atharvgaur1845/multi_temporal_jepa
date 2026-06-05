"""YAML config loading (infra — implemented for you)."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path) -> dict:
    """Load a single YAML file into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(*paths) -> dict:
    """Shallow-merge several YAML files (later paths override earlier keys at the top level).

    For deep merging of nested keys, extend this — but keep configs flat-ish to avoid surprises.
    """
    cfg: dict = {}
    for p in paths:
        cfg.update(load_yaml(Path(p)))
    return cfg
