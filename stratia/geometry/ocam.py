"""Scaramuzza omnidirectional camera model (OCamCalib), as used by the Eye2Sky calibration files.

Conventions
-----------
* OCamCalib's centre (xc, yc) is (row, column). Verified on the Eye2Sky files in P013: for 34 of 38
  calibrations the centre is closer to the camera-mask centroid when xc is read as the row.
* The OCamCalib camera frame has x along rows, y along columns and z = F(rho) (negative toward the scene).
  Public functions use pixel coordinates (u = column, v = row) and return rays in the STRATIA/CloudScope
  camera frame (ADR-012): +x toward +u, +y toward +v, +z along the optical axis (into the scene).
  The mapping (x, y, z)_std = (y, x, -z)_ocam is a proper rotation (determinant +1).
* world2cam has no closed form; it inverts theta(rho) = atan(F(rho) / rho) with a dense monotonic table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class OcamModel:
    ss: tuple[float, ...]       # F(rho) = ss[0] + ss[1] rho + ss[2] rho^2 + ...
    xc: float                   # centre row (pixels, 0-based)
    yc: float                   # centre column
    c: float = 1.0              # affine (stretch) parameters
    d: float = 0.0
    e: float = 0.0
    width: int = 0
    height: int = 0
    _table: tuple[np.ndarray, np.ndarray] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        r_max = float(np.hypot(self.width, self.height)) if self.width and self.height else 3000.0
        rho = np.linspace(1e-6, r_max, 40001)
        theta = np.arctan(self._poly(rho) / rho)
        if np.any(np.diff(theta) <= 0):
            # Keep only the monotonic part (the model is not invertible beyond it).
            stop = int(np.argmax(np.diff(theta) <= 0))
            rho, theta = rho[: stop + 1], theta[: stop + 1]
        object.__setattr__(self, "_table", (theta, rho))

    def _poly(self, rho: np.ndarray) -> np.ndarray:
        return np.polynomial.polynomial.polyval(rho, np.asarray(self.ss, dtype=float))

    # -- pixel -> ray ---------------------------------------------------------
    def pixel_to_ray(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Unit rays (..., 3) in the standard camera frame for pixel columns u and rows v."""
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        inv_det = 1.0 / (self.c - self.d * self.e)
        dr, dc = v - self.xc, u - self.yc
        xp = inv_det * (dr - self.d * dc)
        yp = inv_det * (-self.e * dr + self.c * dc)
        zp = self._poly(np.hypot(xp, yp))
        ocam = np.stack([xp, yp, zp], axis=-1)
        ocam /= np.linalg.norm(ocam, axis=-1, keepdims=True)
        return np.stack([ocam[..., 1], ocam[..., 0], -ocam[..., 2]], axis=-1)

    # -- ray -> pixel ---------------------------------------------------------
    def ray_to_pixel(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel (u, v) for rays (..., 3) in the standard camera frame; NaN where outside the model's range."""
        rays = np.asarray(rays, dtype=float)
        x, y, z = rays[..., 1], rays[..., 0], -rays[..., 2]          # back to the OCamCalib frame
        norm = np.hypot(x, y)
        theta = np.arctan2(z, norm)
        th_tab, rho_tab = self._table
        rho = np.interp(theta, th_tab, rho_tab, left=np.nan, right=np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            xs = np.where(norm > 0, x / norm * rho, 0.0)
            ys = np.where(norm > 0, y / norm * rho, 0.0)
        row = xs * self.c + ys * self.d + self.xc
        col = xs * self.e + ys + self.yc
        return col, row

    @property
    def max_theta_deg(self) -> float:
        """Largest representable angle above the OCamCalib image plane (0 deg = 90 deg off the optical axis)."""
        return float(np.degrees(self._table[0][-1]))
