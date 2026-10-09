"""Photometric augmentations (P048): what a sky camera can actually do to a frame.

Every function takes an RGB uint8 image (H, W, 3) and a NumPy Generator and returns a new uint8 image. Ranges are
stated in physical terms and kept inside what the cameras in the manifest produce:

* exposure: a factor on linear intensity (a longer or shorter exposure), applied after decoding the sRGB-like
  gamma so that highlights clip the way a sensor clips;
* white balance: red and blue gains within +-12 %, moving along the colour-temperature axis only. No hue rotation
  anywhere: a hue shift turns a blue sky into something no sky is, and sky colour carries cloud information;
* JPEG recompression at a random quality;
* sensor noise: Gaussian in linear space with a shot-noise term (stronger in the bright parts, as in a sensor);
* Sun glare: a bloom at the Sun's own pixel when the geometry is known (else a random place inside the valid
  mask), with a faint ghost reflection mirrored through the image centre, as the Q25 domes show (P044);
* dirt and raindrops on the dome: dark soft blobs, and small lens-like discs that blur and brighten what is under
  them;
* obstruction cut-outs: dark silhouettes (a pole, a branch-like blob) pasted into the frame; the returned mask
  says where, so the dense label becomes "invalid" there and the ray-map patches lose their valid flag
  (``stratia.geometry.transforms.geometry_for(..., obstruction=...)``).
"""

from __future__ import annotations

import cv2
import numpy as np

GAMMA = 2.2


def _to_linear(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0) ** GAMMA


def _to_uint8(lin: np.ndarray) -> np.ndarray:
    return np.clip(np.clip(lin, 0.0, 1.0) ** (1.0 / GAMMA) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def exposure(img: np.ndarray, rng: np.random.Generator, low: float = 0.6, high: float = 1.5) -> np.ndarray:
    """Linear-intensity factor in [low, high] (log-uniform); highlights clip."""
    f = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return _to_uint8(_to_linear(img) * f)


def white_balance(img: np.ndarray, rng: np.random.Generator, max_gain: float = 0.12) -> np.ndarray:
    """Red and blue gains within +-max_gain along the colour-temperature axis (one moves up, the other down)."""
    t = float(rng.uniform(-max_gain, max_gain))
    gains = np.array([1.0 + t, 1.0, 1.0 - t], dtype=np.float32)
    return _to_uint8(_to_linear(img) * gains)


def jpeg(img: np.ndarray, rng: np.random.Generator, q_low: int = 40, q_high: int = 95) -> np.ndarray:
    q = int(rng.integers(q_low, q_high + 1))
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def sensor_noise(img: np.ndarray, rng: np.random.Generator, read_sigma: float = 0.01, shot_scale: float = 0.03) -> np.ndarray:
    """Gaussian noise in linear space: sigma = read_sigma + shot_scale * sqrt(intensity) (both sampled up to the given maxima)."""
    lin = _to_linear(img)
    read = float(rng.uniform(0.0, read_sigma))
    shot = float(rng.uniform(0.0, shot_scale))
    sigma = read + shot * np.sqrt(lin)
    return _to_uint8(lin + rng.normal(size=lin.shape).astype(np.float32) * sigma)


def sun_glare(img: np.ndarray, rng: np.random.Generator, sun_px: tuple[float, float] | None = None,
              valid: np.ndarray | None = None, strength: float = 1.0) -> np.ndarray:
    """Bloom at the Sun (or a random valid pixel) plus a faint ghost mirrored through the image centre."""
    h, w = img.shape[:2]
    if sun_px is None or not np.all(np.isfinite(sun_px)):
        if valid is not None and valid.any():
            ys, xs = np.nonzero(valid)
            k = int(rng.integers(len(xs)))
            sun_px = (float(xs[k]), float(ys[k]))
        else:
            sun_px = (float(rng.uniform(0.2 * w, 0.8 * w)), float(rng.uniform(0.2 * h, 0.8 * h)))
    su, sv = sun_px
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    radius = float(rng.uniform(0.04, 0.12)) * min(h, w) * strength
    amp = float(rng.uniform(0.3, 1.0)) * strength
    d2 = (xx - su) ** 2 + (yy - sv) ** 2
    bloom = amp * np.exp(-d2 / (2 * radius**2))
    # warm tint and a wide halo
    halo = 0.25 * amp * np.exp(-d2 / (2 * (3 * radius) ** 2))
    gu, gv = (w - 1) - su, (h - 1) - sv                      # ghost through the centre
    ghost_r = float(rng.uniform(0.5, 1.2)) * radius
    ghost = 0.15 * amp * np.exp(-((xx - gu) ** 2 + (yy - gv) ** 2) / (2 * ghost_r**2))
    lin = _to_linear(img)
    tint = np.array([1.0, 0.95, 0.8], dtype=np.float32)
    lin = lin + (bloom + halo)[..., None] * tint + ghost[..., None]
    out = _to_uint8(lin)
    if valid is not None:
        out[~valid] = img[~valid]
    return out


def dirt_and_drops(img: np.ndarray, rng: np.random.Generator, n_dirt: tuple[int, int] = (2, 8),
                   n_drops: tuple[int, int] = (0, 6), valid: np.ndarray | None = None) -> np.ndarray:
    """Soft dark dirt blobs and small lens-like water drops on the dome."""
    h, w = img.shape[:2]
    lin = _to_linear(img)
    shade = np.zeros((h, w), dtype=np.float32)
    for _ in range(int(rng.integers(n_dirt[0], n_dirt[1] + 1))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.01, 0.05) * min(h, w)
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        ax, ay = r * rng.uniform(0.6, 1.4), r * rng.uniform(0.6, 1.4)
        blob = np.exp(-(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2) / 2)
        shade += float(rng.uniform(0.15, 0.5)) * blob
    lin = lin * (1.0 - np.clip(shade, 0, 0.7))[..., None]
    out = _to_uint8(lin)
    blurred = cv2.GaussianBlur(out, (0, 0), 3.0)
    for _ in range(int(rng.integers(n_drops[0], n_drops[1] + 1))):
        cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        r = int(rng.uniform(0.008, 0.03) * min(h, w)) + 2
        yy, xx = np.mgrid[max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        rim = ((xx - cx) ** 2 + (yy - cy) ** 2 > (0.8 * r) ** 2) & inside
        patch = out[yy, xx].astype(np.float32)
        patch[inside] = blurred[yy, xx][inside].astype(np.float32) * 1.08
        patch[rim] *= 0.85
        out[yy, xx] = np.clip(patch, 0, 255).astype(np.uint8)
    if valid is not None:
        out[~valid] = img[~valid]
    return out


def obstruction_cutout(img: np.ndarray, rng: np.random.Generator, n: tuple[int, int] = (1, 2),
                       valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Dark silhouettes pasted into the frame: a pole from an edge, or a branch-like blob. Returns (image, mask)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(int(rng.integers(n[0], n[1] + 1))):
        if rng.uniform() < 0.5:
            # pole: a thick line from a random border point toward the interior
            side = int(rng.integers(4))
            if side == 0:
                p0, p1 = (int(rng.uniform(0, w)), 0), (int(rng.uniform(0, w)), int(rng.uniform(0.2, 0.7) * h))
            elif side == 1:
                p0, p1 = (int(rng.uniform(0, w)), h - 1), (int(rng.uniform(0, w)), int(rng.uniform(0.3, 0.8) * h))
            elif side == 2:
                p0, p1 = (0, int(rng.uniform(0, h))), (int(rng.uniform(0.2, 0.7) * w), int(rng.uniform(0, h)))
            else:
                p0, p1 = (w - 1, int(rng.uniform(0, h))), (int(rng.uniform(0.3, 0.8) * w), int(rng.uniform(0, h)))
            cv2.line(mask, p0, p1, 255, thickness=int(rng.uniform(0.01, 0.03) * min(h, w)) + 1)
        else:
            # branch-like blob: a random polygon around a centre
            cx, cy = rng.uniform(0.1 * w, 0.9 * w), rng.uniform(0.1 * h, 0.9 * h)
            k = int(rng.integers(6, 12))
            ang = np.sort(rng.uniform(0, 2 * np.pi, k))
            rad = rng.uniform(0.03, 0.12, k) * min(h, w)
            pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], axis=1).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
    m = mask > 0
    if valid is not None:
        m &= valid
    out = img.copy()
    dark = (img.astype(np.float32) * float(rng.uniform(0.02, 0.15))).astype(np.uint8)
    out[m] = dark[m]
    return out, m
