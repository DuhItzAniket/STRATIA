"""Data-free tests of the B0268 Sun-track fit: the plateau detector and the rotation recovered from a synthetic track."""

import importlib.util
from pathlib import Path

import numpy as np

from stratia.geometry.pose import euler_to_rotation, fit_rotation, rotation_to_euler_zyx
from stratia.geometry.sun import azel_to_enu

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("b0268_sun_track", ROOT / "scripts" / "b0268_sun_track.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_find_sun_takes_the_saturated_plateau_on_the_ring_and_nothing_elsewhere():
    mod = _load()
    h, w = 200, 300
    yy, xx = np.mgrid[0:h, 0:w]
    zenith = np.hypot(xx - 150, yy - 100) * (90.0 / 100.0)  # an equidistant hemisphere, 90 degrees at r = 100 px
    lum = np.full((h, w), 0.4, np.float32)
    lum[(xx - 190) ** 2 + (yy - 100) ** 2 <= 12**2] = 1.0  # the Sun: 36 degrees from the zenith, along +x
    lum[(xx - 150) ** 2 + (yy - 100) ** 2 <= 12**2] = 1.0  # a saturated cumulus at the zenith, off both rings below
    found = mod.find_sun(lum, zenith, sun_zenith_deg=36.0, blur=15)
    assert found is not None
    col, row, value = found
    assert abs(col - 190) <= 3 and abs(row - 100) <= 3 and value > 0.9
    assert mod.find_sun(np.full((h, w), 0.4, np.float32), zenith, 36.0, blur=15) is None  # nothing saturated
    assert mod.find_sun(lum, zenith, sun_zenith_deg=85.0, blur=15) is None  # the plateau is off that ring


def test_rotation_is_recovered_from_a_synthetic_sun_track():
    rng = np.random.default_rng(0)
    truth = euler_to_rotation(np.radians(1.0), np.radians(-2.0), np.radians(37.0))  # camera -> ENU
    az = np.linspace(80.0, 260.0, 25)
    el = 60.0 - np.abs(np.linspace(-1.0, 1.0, 25)) * 40.0
    enu = azel_to_enu(az, el)
    cam = enu @ truth  # enu_to_cam
    cam = cam + rng.normal(0.0, 0.002, cam.shape)
    cam /= np.linalg.norm(cam, axis=1, keepdims=True)
    r, inlier, err = fit_rotation(cam, enu)
    roll, pitch, yaw = (np.degrees(a) for a in rotation_to_euler_zyx(r))
    assert inlier.sum() >= 20
    assert abs(yaw - 37.0) < 0.3 and abs(pitch + 2.0) < 0.3 and abs(roll - 1.0) < 0.3
    assert np.median(err[inlier]) < 0.3
