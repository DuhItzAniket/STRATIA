"""Geometry dropout (P050): train-time masking of the ray map and the Sun metadata.

The contract's "unknown geometry" state is an all-zero ray map and an all-zero meta vector. During training the
same state is produced on purpose, with probability ``p_ray`` for the ray map and ``p_meta`` for the metadata
(independently unless ``tie`` is set), so that the model (1) still works on uncalibrated cameras and (2) cannot use
the mere presence of geometry as a dataset identity (P029: dataset is readable from pixels already; zeros must not
add a second, trivial cue). The null state is turned into learned null tokens by ``stratia.model.skyray``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometryDropout:
    p_ray: float = 0.3
    p_meta: float = 0.3
    tie: bool = False            # True: one draw decides both

    def __call__(self, ray_map: np.ndarray, meta: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
        drop_ray = rng.uniform() < self.p_ray
        drop_meta = drop_ray if self.tie else rng.uniform() < self.p_meta
        rm = np.zeros_like(ray_map) if drop_ray else ray_map
        mt = np.zeros_like(meta) if drop_meta else meta
        return rm, mt, {"ray_dropped": bool(drop_ray), "meta_dropped": bool(drop_meta)}

    def batch(self, ray_maps: np.ndarray, metas: np.ndarray,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-sample dropout over a batch: (ray_maps, metas, ray_dropped, meta_dropped)."""
        n = ray_maps.shape[0]
        drop_ray = rng.uniform(size=n) < self.p_ray
        drop_meta = drop_ray.copy() if self.tie else rng.uniform(size=n) < self.p_meta
        rm = ray_maps.copy()
        rm[drop_ray] = 0
        mt = metas.copy()
        mt[drop_meta] = 0
        return rm, mt, drop_ray, drop_meta


def is_null_ray_map(ray_map: np.ndarray) -> bool:
    return not np.any(ray_map[3] > 0)


def is_null_meta(meta: np.ndarray) -> bool:
    return float(meta[2]) <= 0.0
