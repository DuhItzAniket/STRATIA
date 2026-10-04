"""Experiment configuration: YAML files composed with OmegaConf, plus a stable config hash.

A config file may list `defaults: [other.yaml, ...]` (paths relative to the file); those are
merged first, then the file itself, then command-line overrides (`key.sub=value`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    path = Path(path)
    raw = OmegaConf.load(path)
    defaults = raw.pop("defaults", []) if isinstance(raw, DictConfig) else []
    merged = OmegaConf.create()
    for d in defaults:
        merged = OmegaConf.merge(merged, load_config(path.parent / str(d)))
    merged = OmegaConf.merge(merged, raw)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(merged)
    return merged


def config_hash(cfg: DictConfig, length: int = 10) -> str:
    """SHA-256 of the resolved config with sorted keys; identical configs give identical hashes."""
    text = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
