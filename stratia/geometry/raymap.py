"""Ray maps (P045): the contract's per-patch geometry input, computed from a camera model, a pose and the Sun.

Contract (docs/contract.md, v1): ``ray_map`` is float32 ``[4, H/p, W/p]`` at the patch centres of the model input.
Channels 0-2 are the unit viewing direction in the Sun-aligned frame (z up, x toward the Sun's azimuth,
y = z x x); channel 3 is 1 where the direction is valid (calibrated camera, inside the camera mask, above the
horizon) and 0 elsewhere, where channels 0-2 are 0 too. An uncalibrated camera, a missing pose or an unknown Sun
give an all-zero tensor. ``meta`` is ``[cos(Sun zenith), sin(Sun zenith), valid]``.

Geometry goes through one affine map from model-input pixels to source-image pixels (3x3, pixel-centre
convention), so the same code serves the plain full-frame resize of the contract and every crop/resize/flip/
rotation of P046: the ray of an output patch is the ray of the source pixel its centre came from.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..contract import DEFAULT_INPUT_SIZE, PATCH_SIZE
from .cameras import CameraModel
from .pose import Pose
from .sun import sun_aligned_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
NEAR_HORIZON_FILE = REPO_ROOT / "configs" / "near_horizon.csv"
MASK_FRACTION = 0.5        # a patch is inside the mask when at least this share of its footprint is


# ----------------------------------------------------------------------------------------------- affine helpers
def full_frame_affine(src_w: int, src_h: int, out_w: int = DEFAULT_INPUT_SIZE, out_h: int = DEFAULT_INPUT_SIZE) -> np.ndarray:
    """Output pixel -> source pixel for the contract's resize of the whole frame (no crop), pixel-centre convention."""
    sx, sy = src_w / out_w, src_h / out_h
    return np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]])


def apply_affine(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return a[0, 0] * x + a[0, 1] * y + a[0, 2], a[1, 0] * x + a[1, 1] * y + a[1, 2]


def patch_centres(out_h: int, out_w: int, patch: int = PATCH_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) grids of shape (out_h/patch, out_w/patch): the centre of every patch in output pixel coordinates."""
    if out_h % patch or out_w % patch:
        raise ValueError(f"output {out_w}x{out_h} is not a multiple of the patch size {patch}")
    cu = np.arange(out_w // patch) * patch + (patch - 1) / 2.0
    cv = np.arange(out_h // patch) * patch + (patch - 1) / 2.0
    return np.meshgrid(cu, cv)


# ----------------------------------------------------------------------------------------------- camera rays
@dataclass(frozen=True)
class PatchRays:
    rays_cam: np.ndarray      # (H', W', 3) unit rays in the camera frame (NaN where undefined)
    inside: np.ndarray        # (H', W') patch centre maps inside the source image
    mask_fraction: np.ndarray  # (H', W') share of the patch footprint inside the camera mask (1 when no mask)

    @property
    def shape(self) -> tuple[int, int]:
        return self.inside.shape


def patch_rays(camera: CameraModel, affine: np.ndarray | None = None, out_size: int | tuple[int, int] = DEFAULT_INPUT_SIZE,
               patch: int = PATCH_SIZE, mask: np.ndarray | None = None, subgrid: int = 4) -> PatchRays:
    """Camera-frame rays at the patch centres of an output of `out_size` whose pixels map to the source through
    `affine` (default: the contract's full-frame resize). `mask` (H, W, True = sky) is sampled on a subgrid x subgrid
    lattice inside every patch footprint to give the mask fraction."""
    out_h, out_w = (out_size, out_size) if isinstance(out_size, int) else out_size
    a = full_frame_affine(camera.width, camera.height, out_w, out_h) if affine is None else np.asarray(affine, dtype=float)
    cu, cv = patch_centres(out_h, out_w, patch)
    xs, ys = apply_affine(a, cu, cv)
    inside = (np.isfinite(xs) & np.isfinite(ys) & (xs >= -0.5) & (xs <= camera.width - 0.5)
              & (ys >= -0.5) & (ys <= camera.height - 0.5))
    rays = camera.pixel_to_ray(xs, ys) if camera.calibrated else np.full(xs.shape + (3,), np.nan)
    if mask is None:
        frac = np.ones(inside.shape)
    else:
        offs = (np.arange(subgrid) + 0.5) / subgrid * patch - patch / 2.0       # sub-sample offsets inside a patch
        du, dv = np.meshgrid(offs, offs)
        su = cu[..., None] + du.ravel()                                         # (H', W', s*s)
        sv = cv[..., None] + dv.ravel()
        mx, my = apply_affine(a, su, sv)
        ix, iy = np.rint(mx).astype(int), np.rint(my).astype(int)
        ok = (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
        hit = np.zeros(ix.shape, dtype=bool)
        hit[ok] = mask[iy[ok], ix[ok]]
        frac = hit.mean(axis=-1)
    return PatchRays(rays_cam=rays, inside=inside, mask_fraction=frac)


# ----------------------------------------------------------------------------------------------- ray map
def ray_map(camera: CameraModel | None, pose: Pose | None, sun_azimuth_deg: float | None, sun_zenith_deg: float | None,
            mask: np.ndarray | None = None, out_size: int | tuple[int, int] = DEFAULT_INPUT_SIZE, patch: int = PATCH_SIZE,
            affine: np.ndarray | None = None, min_elevation_deg: float = 0.0, rays: PatchRays | None = None) -> np.ndarray:
    """The contract's ray map, float32 (4, H/p, W/p). All zeros when the geometry is unknown."""
    out_h, out_w = (out_size, out_size) if isinstance(out_size, int) else out_size
    shape = (4, out_h // patch, out_w // patch)
    sun_known = (sun_azimuth_deg is not None and sun_zenith_deg is not None
                 and np.isfinite(sun_azimuth_deg) and np.isfinite(sun_zenith_deg))
    if camera is None or not camera.calibrated or pose is None or not sun_known:
        return np.zeros(shape, dtype=np.float32)
    pr = rays if rays is not None else patch_rays(camera, affine, (out_h, out_w), patch, mask)
    enu = pose.cam_to_enu(pr.rays_cam)
    aligned = enu @ sun_aligned_frame(sun_azimuth_deg).T
    elevation = np.degrees(np.arcsin(np.clip(aligned[..., 2], -1.0, 1.0)))
    valid = pr.inside & (pr.mask_fraction >= MASK_FRACTION) & np.isfinite(aligned).all(axis=-1) & (elevation >= min_elevation_deg)
    out = np.zeros(shape, dtype=np.float32)
    out[:3][:, valid] = aligned[valid].T.astype(np.float32)
    out[3] = valid
    return out


def meta_vector(sun_zenith_deg: float | None, valid: bool = True) -> np.ndarray:
    """[cos(Sun zenith), sin(Sun zenith), valid]; zeros when the Sun is unknown."""
    if sun_zenith_deg is None or not np.isfinite(sun_zenith_deg) or not valid:
        return np.zeros(3, dtype=np.float32)
    z = np.radians(float(sun_zenith_deg))
    return np.array([np.cos(z), np.sin(z), 1.0], dtype=np.float32)


def resized_mask(mask: np.ndarray, out_size: int | tuple[int, int] = DEFAULT_INPUT_SIZE,
                 affine: np.ndarray | None = None) -> np.ndarray:
    """The camera mask in output pixels (nearest source pixel through `affine`): for zeroing masked pixels."""
    out_h, out_w = (out_size, out_size) if isinstance(out_size, int) else out_size
    a = full_frame_affine(mask.shape[1], mask.shape[0], out_w, out_h) if affine is None else np.asarray(affine, dtype=float)
    u, v = np.meshgrid(np.arange(out_w, dtype=float), np.arange(out_h, dtype=float))
    xs, ys = apply_affine(a, u, v)
    ix, iy = np.rint(xs).astype(int), np.rint(ys).astype(int)
    ok = (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
    out = np.zeros((out_h, out_w), dtype=bool)
    out[ok] = mask[iy[ok], ix[ok]]
    return out


# ----------------------------------------------------------------------------------------------- derived views
def zenith_angle_deg(rm: np.ndarray) -> np.ndarray:
    """Zenith angle per patch from a ray map (NaN where invalid)."""
    z = np.where(rm[3] > 0, rm[2], np.nan)
    return np.degrees(np.arccos(np.clip(z, -1.0, 1.0)))


def azimuth_from_sun_deg(rm: np.ndarray) -> np.ndarray:
    """Azimuth relative to the Sun per patch, -180..180 (NaN where invalid)."""
    return np.where(rm[3] > 0, np.degrees(np.arctan2(rm[1], rm[0])), np.nan)


def sun_distance_deg(rm: np.ndarray, sun_zenith_deg: float) -> np.ndarray:
    """Angular distance to the Sun per patch (the Sun lies in the x-z plane at the given zenith angle)."""
    s = np.array([np.sin(np.radians(sun_zenith_deg)), 0.0, np.cos(np.radians(sun_zenith_deg))])
    dot = rm[0] * s[0] + rm[1] * s[1] + rm[2] * s[2]
    return np.where(rm[3] > 0, np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))), np.nan)


# ----------------------------------------------------------------------------------------------- near horizon
@lru_cache(maxsize=4)
def near_horizon_table(path: str | None = None) -> dict[str, float]:
    """configs/near_horizon.csv -> {camera_id or prefix pattern: minimum elevation in degrees}."""
    p = Path(path) if path else NEAR_HORIZON_FILE
    if not p.exists():
        return {}
    with p.open(encoding="utf-8", newline="") as f:
        return {row["camera_id"]: float(row["min_elevation_deg"]) for row in csv.DictReader(f)}


def min_elevation_for(camera_id: str, table: dict[str, float] | None = None) -> float:
    table = near_horizon_table() if table is None else table
    if camera_id in table:
        return table[camera_id]
    for key, value in table.items():
        if key.endswith("*") and camera_id.startswith(key[:-1]):
            return value
    return 0.0
