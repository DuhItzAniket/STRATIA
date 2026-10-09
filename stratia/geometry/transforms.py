"""Geometry-consistent image transforms (P046, ADR-004).

Every geometric augmentation is a pixel remapping. A ``GeoState`` carries the affine map from the *current* image's
pixels to the *source* image's pixels and a coverage image (which current pixels hold real content rather than
padding). Transforms warp the image and update the state; the ray map is then computed from the camera model
through the composed map (``geometry_for``), so the ray of an output patch is always the ray of the source pixel
its centre came from. Ray maps are never resampled or rotated as images.

Transforms: ``Resize``, ``Crop``, ``RandomResizedCrop``, ``HorizontalFlip``, ``Rotate``, ``Compose``. Each is
callable as ``transform(image, state, rng) -> (image, state)``; images are HxWxC uint8 or float32 arrays. With
``label=`` (a HxW integer label map) the same remapping is applied to it with nearest-neighbour sampling and
``LABEL_IGNORE`` outside the source, and the call returns ``(image, state, label)``.
Pixel-centre convention throughout (CloudScope ADR-012), matching cv2.warpAffine and cv2.resize.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from ..contract import PATCH_SIZE
from .cameras import CameraModel
from .pose import Pose
from .raymap import MASK_FRACTION, full_frame_affine, patch_rays, ray_map, resized_mask

LABEL_IGNORE = 255      # value of label pixels that come from outside the source (stratia.data.segmentation.IGNORE)


# ----------------------------------------------------------------------------------------------- state
@dataclass(frozen=True)
class GeoState:
    width: int
    height: int
    affine: np.ndarray            # (3, 3): current pixel -> source pixel
    src_width: int
    src_height: int
    coverage: np.ndarray          # (height, width) bool: True where the current image holds source content

    @classmethod
    def identity(cls, width: int, height: int) -> GeoState:
        return cls(width, height, np.eye(3), width, height, np.ones((height, width), dtype=bool))

    @classmethod
    def cached(cls, cached_width: int, cached_height: int, src_width: int, src_height: int) -> GeoState:
        """A cached image that is a plain resize of the source (P022 cache: longest side 768)."""
        a = full_frame_affine(src_width, src_height, cached_width, cached_height)
        return cls(cached_width, cached_height, a, src_width, src_height, np.ones((cached_height, cached_width), dtype=bool))

    def to_source(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        a = self.affine
        return a[0, 0] * x + a[0, 1] * y + a[0, 2], a[1, 0] * x + a[1, 1] * y + a[1, 2]

    def from_source(self, xs, ys) -> tuple[np.ndarray, np.ndarray]:
        inv = np.linalg.inv(self.affine)
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        return inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2], inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]


def _warp_label(label: np.ndarray | None, m: np.ndarray, new_w: int, new_h: int) -> np.ndarray | None:
    if label is None:
        return None
    return cv2.warpAffine(label, m, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=LABEL_IGNORE)


def _result(image, state, label, had_label: bool):
    return (image, state, label) if had_label else (image, state)


def _warp(image: np.ndarray, state: GeoState, m_cur_to_new: np.ndarray, new_w: int, new_h: int,
          interpolation: int = cv2.INTER_LINEAR, label: np.ndarray | None = None):
    """Apply a 2x3 current->new affine (cv2 convention) to the image, the coverage and the label; compose the
    state. Returns (image, state) or (image, state, label) when a label was given."""
    m = np.asarray(m_cur_to_new, dtype=float)
    out = cv2.warpAffine(image, m, (new_w, new_h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cov = cv2.warpAffine(state.coverage.astype(np.uint8), m, (new_w, new_h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    m3 = np.vstack([m, [0.0, 0.0, 1.0]])
    new_to_cur = np.linalg.inv(m3)
    new_state = replace(state, width=new_w, height=new_h, affine=state.affine @ new_to_cur, coverage=cov)
    return _result(out, new_state, _warp_label(label, m, new_w, new_h), label is not None)


# ----------------------------------------------------------------------------------------------- transforms
class Resize:
    def __init__(self, width: int, height: int | None = None):
        self.width, self.height = width, height or width

    def __call__(self, image: np.ndarray, state: GeoState, rng=None, label: np.ndarray | None = None):
        shrinking = self.width < state.width or self.height < state.height
        inter = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        out = cv2.resize(image, (self.width, self.height), interpolation=inter)
        cov = cv2.resize(state.coverage.astype(np.uint8), (self.width, self.height), interpolation=cv2.INTER_NEAREST) > 0
        step = full_frame_affine(state.width, state.height, self.width, self.height)    # new pixel -> current pixel
        new_state = replace(state, width=self.width, height=self.height, affine=state.affine @ step, coverage=cov)
        new_label = None if label is None else cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return _result(out, new_state, new_label, label is not None)


class Crop:
    """Crop the box (x0, y0, width, height) of the current image; parts outside it are zero padding."""

    def __init__(self, x0: int, y0: int, width: int, height: int):
        self.x0, self.y0, self.width, self.height = int(x0), int(y0), int(width), int(height)

    def __call__(self, image: np.ndarray, state: GeoState, rng=None, label: np.ndarray | None = None):
        m = np.array([[1.0, 0.0, -self.x0], [0.0, 1.0, -self.y0]])
        return _warp(image, state, m, self.width, self.height, cv2.INTER_NEAREST, label)


class RandomResizedCrop:
    """torchvision-style: a box with area share in `scale` and aspect ratio in `ratio`, resized to `size`."""

    def __init__(self, size: int, scale=(0.5, 1.0), ratio=(0.8, 1.25)):
        self.size, self.scale, self.ratio = size, scale, ratio

    def sample_box(self, width: int, height: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
        area = width * height
        for _ in range(10):
            target = area * rng.uniform(*self.scale)
            log_ratio = np.log(self.ratio)
            aspect = float(np.exp(rng.uniform(*log_ratio)))
            w = int(round(np.sqrt(target * aspect)))
            h = int(round(np.sqrt(target / aspect)))
            if 0 < w <= width and 0 < h <= height:
                x0 = int(rng.integers(0, width - w + 1))
                y0 = int(rng.integers(0, height - h + 1))
                return x0, y0, w, h
        side = min(width, height)
        return (width - side) // 2, (height - side) // 2, side, side

    def __call__(self, image: np.ndarray, state: GeoState, rng=None, label: np.ndarray | None = None):
        rng = np.random.default_rng() if rng is None else rng
        x0, y0, w, h = self.sample_box(state.width, state.height, rng)
        if label is None:
            image, state = Crop(x0, y0, w, h)(image, state)
            return Resize(self.size, self.size)(image, state)
        image, state, label = Crop(x0, y0, w, h)(image, state, label=label)
        return Resize(self.size, self.size)(image, state, label=label)


class HorizontalFlip:
    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, image: np.ndarray, state: GeoState, rng=None, label: np.ndarray | None = None):
        rng = np.random.default_rng() if rng is None else rng
        if self.p < 1.0 and rng.uniform() >= self.p:
            return _result(image, state, label, label is not None)
        m = np.array([[-1.0, 0.0, state.width - 1.0], [0.0, 1.0, 0.0]])
        return _warp(image, state, m, state.width, state.height, cv2.INTER_NEAREST, label)


class Rotate:
    """Rotate about the image centre by `degrees` (a number, or a (low, high) range sampled per call), same size.
    Positive angles turn the content counter-clockwise on screen (cv2.getRotationMatrix2D convention)."""

    def __init__(self, degrees: float | tuple[float, float]):
        self.degrees = degrees

    def angle(self, rng) -> float:
        if isinstance(self.degrees, tuple | list):
            rng = np.random.default_rng() if rng is None else rng
            return float(rng.uniform(*self.degrees))
        return float(self.degrees)

    def __call__(self, image: np.ndarray, state: GeoState, rng=None, label: np.ndarray | None = None):
        angle = self.angle(rng)
        centre = ((state.width - 1) / 2.0, (state.height - 1) / 2.0)
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        return _warp(image, state, m, state.width, state.height, cv2.INTER_LINEAR, label)


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image: np.ndarray, state: GeoState, rng=None, label: np.ndarray | None = None):
        if label is None:
            for t in self.transforms:
                image, state = t(image, state, rng)
            return image, state
        for t in self.transforms:
            image, state, label = t(image, state, rng, label=label)
        return image, state, label


# ----------------------------------------------------------------------------------------------- geometry outputs
def patch_fraction(flags: np.ndarray, patch: int = PATCH_SIZE) -> np.ndarray:
    """Share of True pixels in every patch of a (H, W) boolean image (H, W multiples of `patch`)."""
    h, w = flags.shape
    return flags.reshape(h // patch, patch, w // patch, patch).mean(axis=(1, 3))


def geometry_for(state: GeoState, camera: CameraModel | None, pose: Pose | None, sun_azimuth_deg: float | None,
                 sun_zenith_deg: float | None, mask: np.ndarray | None = None, patch: int = PATCH_SIZE,
                 min_elevation_deg: float = 0.0, obstruction: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(ray_map, mask_out) for the current image of `state`: the contract's ray map computed through the composed
    pixel map, with padding patches (no source content) invalid; `mask_out` is the camera mask in current pixels,
    False where the image is padding. `obstruction` (current pixels, True = synthetic obstruction, P048) is treated
    like padding: those pixels leave the mask and patches mostly covered by it become invalid."""
    if obstruction is not None:
        state = replace(state, coverage=state.coverage & ~obstruction.astype(bool))
    out_size = (state.height, state.width)
    rays = None
    if camera is not None and camera.calibrated:
        rays = patch_rays(camera, state.affine, out_size, patch, mask)
        covered = patch_fraction(state.coverage, patch) >= MASK_FRACTION
        rays = replace(rays, mask_fraction=np.where(covered, rays.mask_fraction, 0.0))
    rm = ray_map(camera, pose, sun_azimuth_deg, sun_zenith_deg, mask, out_size, patch, state.affine, min_elevation_deg, rays)
    if mask is None:
        mask_out = state.coverage.copy()
    else:
        mask_out = resized_mask(mask, out_size, state.affine) & state.coverage
    return rm, mask_out
