"""Sun position and the East-North-Up frame (P044).

* ``sun_position``: NREL SPA through pvlib. ``zenith``/``elevation`` are geometric (what the manifest stores and the
  contract's ``meta`` uses); ``apparent_*`` include refraction and are what a camera sees near the horizon.
* ENU vectors: x east, y north, z up; azimuth from north clockwise, elevation from the horizon (CloudScope ADR-012).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sun_position(times, latitude: float, longitude: float, altitude: float = 0.0) -> pd.DataFrame:
    """SPA solar position for UTC `times` (anything pandas turns into a DatetimeIndex; naive times are taken as UTC).

    Columns: zenith, elevation, apparent_zenith, apparent_elevation, azimuth (degrees).
    """
    import pvlib

    idx = pd.DatetimeIndex(pd.to_datetime(times))
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    sp = pvlib.solarposition.get_solarposition(idx, latitude, longitude, altitude=altitude)
    out = pd.DataFrame({"zenith": sp.zenith.to_numpy(), "elevation": sp.elevation.to_numpy(),
                        "apparent_zenith": sp.apparent_zenith.to_numpy(),
                        "apparent_elevation": sp.apparent_elevation.to_numpy(),
                        "azimuth": sp.azimuth.to_numpy()}, index=idx)
    return out


def azel_to_enu(azimuth_deg, elevation_deg) -> np.ndarray:
    """Unit vectors (..., 3) in East-North-Up for azimuth (from north, clockwise) and elevation in degrees."""
    az, el = np.radians(np.asarray(azimuth_deg, dtype=float)), np.radians(np.asarray(elevation_deg, dtype=float))
    return np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], axis=-1)


def enu_to_azel(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth (0..360, from north clockwise) and elevation in degrees for ENU vectors (..., 3)."""
    v = np.asarray(vectors, dtype=float)
    e, n, u = v[..., 0], v[..., 1], v[..., 2]
    az = np.degrees(np.arctan2(e, n)) % 360.0
    el = np.degrees(np.arctan2(u, np.hypot(e, n)))
    return az, el


def sun_aligned_frame(sun_azimuth_deg: float) -> np.ndarray:
    """Rotation (3x3) from ENU to the contract's Sun-aligned frame: z up, x toward the Sun's azimuth, y = z x x.

    Rows are the new axes expressed in ENU, so ``R @ v_enu`` gives the Sun-aligned coordinates.
    """
    az = np.radians(float(sun_azimuth_deg))
    x = np.array([np.sin(az), np.cos(az), 0.0])        # horizontal direction toward the Sun's azimuth
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z])
