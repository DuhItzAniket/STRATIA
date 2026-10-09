"""The augmentation policy (P048): configs/augment.yaml applied to one sample.

``AugmentationPolicy(config).__call__(image, camera_type, rng, state=None, label=None, sun_px=None, valid=None)``
returns an ``Augmented`` record: the geometric part (per camera-type group, through the P046 transforms so the ray
map follows) and the photometric part (``stratia.augment.photometric``), plus the obstruction mask for
``geometry_for``. ``strength`` scales ranges; every augmentation has an ``enabled`` switch for the ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..geometry import transforms as T
from . import photometric as P

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "augment.yaml"


@dataclass
class Augmented:
    image: np.ndarray                     # (H, W, 3) uint8 at the policy's input size
    state: T.GeoState                     # geometry after the geometric transforms
    label: np.ndarray | None              # dense label map transformed alongside, None when there was none
    obstruction: np.ndarray | None        # (H, W) bool, True where a synthetic obstruction was pasted
    applied: list[str]                    # names of the augmentations that fired, in order


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _scaled_range(low: float, high: float, centre: float, strength: float) -> tuple[float, float]:
    """Scale a range about `centre` by `strength` (log scale for multiplicative ranges around 1)."""
    if centre == 1.0:
        return float(np.exp(np.log(low) * strength)), float(np.exp(np.log(high) * strength))
    return centre + (low - centre) * strength, centre + (high - centre) * strength


class AugmentationPolicy:
    def __init__(self, config: dict | str | Path | None = None, strength: float | None = None):
        self.cfg = config if isinstance(config, dict) else load_config(config)
        self.strength = float(self.cfg.get("strength", 1.0) if strength is None else strength)
        self.size = int(self.cfg.get("input_size", 512))
        self.group_of = {ct: g for g, types in self.cfg["groups"].items() for ct in types}

    # -- geometric ---------------------------------------------------------------------------------------------
    def group(self, camera_type: str) -> str:
        return self.group_of.get(camera_type, "consumer")

    def geometric_chain(self, camera_type: str) -> T.Compose:
        g = self.cfg["geometric"][self.group(camera_type)]
        s = self.strength
        steps = []
        crop = g.get("crop", {})
        if crop.get("enabled", False) and s > 0:
            lo, hi = crop["scale"]
            lo = 1.0 - (1.0 - lo) * min(s, 1.0 / max(1.0 - lo, 1e-6))          # never below 0 area share
            r_lo, r_hi = crop["ratio"]
            steps.append(T.RandomResizedCrop(self.size, scale=(max(lo, 0.05), hi), ratio=(r_lo ** s, r_hi ** s)))
        else:
            steps.append(T.Resize(self.size, self.size))
        rot = g.get("rotate", {})
        if rot.get("enabled", False) and s > 0:
            lo, hi = rot["degrees"]
            steps.append(T.Rotate((max(lo * s, -180.0), min(hi * s, 180.0))))
        flip = g.get("flip", {})
        if flip.get("enabled", False) and flip.get("p", 0) > 0:
            steps.append(T.HorizontalFlip(p=float(flip["p"])))
        return T.Compose(steps)

    # -- photometric ---------------------------------------------------------------------------------------------
    def photometric(self, image: np.ndarray, rng: np.random.Generator, sun_px=None, valid=None,
                    only: set[str] | None = None) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
        """Apply the photometric part; `only` restricts to a subset (galleries, ablations)."""
        ph = self.cfg["photometric"]
        s = self.strength
        obstruction = None
        applied = []
        order = ["exposure", "white_balance", "sun_glare", "dirt_and_drops", "obstruction", "sensor_noise", "jpeg"]
        for name in order:
            c = ph.get(name, {})
            if not c.get("enabled", False) or (only is not None and name not in only):
                continue
            if only is None and rng.uniform() >= float(c.get("p", 1.0)):
                continue
            if s <= 0:
                continue
            if name == "exposure":
                lo, hi = _scaled_range(c["low"], c["high"], 1.0, s)
                image = P.exposure(image, rng, lo, hi)
            elif name == "white_balance":
                image = P.white_balance(image, rng, float(c["max_gain"]) * s)
            elif name == "sun_glare":
                image = P.sun_glare(image, rng, sun_px=sun_px, valid=valid, strength=float(c.get("strength", 1.0)) * s)
            elif name == "dirt_and_drops":
                image = P.dirt_and_drops(image, rng, valid=valid)
            elif name == "obstruction":
                image, obstruction = P.obstruction_cutout(image, rng, valid=valid)
            elif name == "sensor_noise":
                image = P.sensor_noise(image, rng, float(c["read_sigma"]) * s, float(c["shot_scale"]) * s)
            elif name == "jpeg":
                image = P.jpeg(image, rng, int(c["q_low"]), int(c["q_high"]))
            applied.append(name)
        return image, obstruction, applied

    # -- whole policy --------------------------------------------------------------------------------------------
    def __call__(self, image: np.ndarray, camera_type: str, rng: np.random.Generator, state: T.GeoState | None = None,
                 label: np.ndarray | None = None, sun_px_source: tuple[float, float] | None = None,
                 valid_source: np.ndarray | None = None) -> Augmented:
        """`sun_px_source` and `valid_source` are in the coordinates of `image` (before the geometric chain)."""
        h, w = image.shape[:2]
        state = T.GeoState.identity(w, h) if state is None else state
        chain = self.geometric_chain(camera_type)
        if label is None:
            image, state = chain(image, state, rng)
        else:
            image, state, label = chain(image, state, rng, label=label)
        # Sun pixel and validity in the new coordinates (through the composed map)
        sun_px = None
        if sun_px_source is not None and np.all(np.isfinite(sun_px_source)):
            # the chain started from `state` whose affine maps to the *source*; the given pixel is in the pre-chain image,
            # so map source -> pre-chain is identity only when state was the identity. Callers pass source pixels.
            ou, ov = state.from_source(sun_px_source[0], sun_px_source[1])
            if 0 <= ou < state.width and 0 <= ov < state.height:
                sun_px = (float(ou), float(ov))
        valid = state.coverage.copy()
        if valid_source is not None:
            valid &= T.resized_mask(valid_source, (state.height, state.width), state.affine)
        image, obstruction, applied = self.photometric(image, rng, sun_px=sun_px, valid=valid)
        if obstruction is not None and label is not None:
            label = label.copy()
            label[obstruction] = 0                    # sky-parsing class 0 = invalid / obstruction (contract)
        return Augmented(image=image, state=state, label=label, obstruction=obstruction, applied=applied)
