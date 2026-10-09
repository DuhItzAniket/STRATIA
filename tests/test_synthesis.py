"""P049: consumer-view synthesis keeps directions exact and labels honest."""

import numpy as np
import pandas as pd

from stratia.geometry import synthesis as S
from stratia.geometry.cameras import nominal_fisheye
from stratia.geometry.ocam import OcamModel
from stratia.geometry.pose import Pose, euler_to_rotation
from stratia.geometry.raymap import zenith_angle_deg
from stratia.geometry.sun import azel_to_enu, enu_to_azel

OLDLR = OcamModel(ss=(-642.6690459406254, 0.0, 0.0002441927468927244, 4.838462720720562e-07),
                  xc=1028.60, yc=1072.59, width=2112, height=2048)
UP = Pose(R=euler_to_rotation(0.01, -0.02, 1.4, "zyx"), source="test")
B0268 = S.scaled_camera(nominal_fisheye(4656, 3496, 105.0, "b0268", "nominal"), 512)


def test_pointing_pose_axes():
    pose = S.pointing_pose(S.Pointing(azimuth_deg=0.0, elevation_deg=30.0))
    axis = pose.cam_to_enu(np.array([0.0, 0.0, 1.0]))
    az, el = enu_to_azel(axis)
    assert abs(az) < 1e-9 and abs(el - 30.0) < 1e-9
    right = pose.cam_to_enu(np.array([1.0, 0.0, 0.0]))
    down = pose.cam_to_enu(np.array([0.0, 1.0, 0.0]))
    assert np.allclose(right, [1.0, 0.0, 0.0])                      # facing north, image right = east
    assert down[2] < 0 and abs(down[0]) < 1e-9                       # image down points toward the ground
    assert np.isclose(np.linalg.det(pose.R), 1.0)
    rolled = S.pointing_pose(S.Pointing(0.0, 30.0, roll_deg=10.0))
    assert np.allclose(rolled.cam_to_enu(np.array([0.0, 0.0, 1.0])), axis)   # roll keeps the axis


def test_scaled_camera_keeps_the_field_of_view():
    full = nominal_fisheye(4656, 3496, 105.0, "b0268", "nominal")
    small = S.scaled_camera(full, 1024)
    assert (small.width, small.height) == (1024, 769) or (small.width, small.height) == (1024, 768)
    left_full = full.pixel_to_ray(0.0, full.cy)
    left_small = small.pixel_to_ray(0.0, small.cy)
    assert abs(np.degrees(np.arccos(left_full[2])) - np.degrees(np.arccos(left_small[2]))) < 0.1


def _synthetic_source():
    """A fisheye frame whose colour encodes the ENU direction: R = east, G = north, B = up (0..255)."""
    u, v = np.meshgrid(np.arange(OLDLR.width, dtype=float), np.arange(OLDLR.height, dtype=float))
    enu = UP.cam_to_enu(OLDLR.pixel_to_ray(u, v))
    img = np.clip((enu + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    mask = (u - OLDLR.yc) ** 2 + (v - OLDLR.xc) ** 2 < 960**2
    img[~mask] = 0
    return img, mask


def test_render_view_samples_the_right_directions():
    src, mask = _synthetic_source()
    rng = np.random.default_rng(0)
    for _ in range(3):
        p = S.random_pointing(rng, elevation=(25.0, 60.0))
        pose = S.pointing_pose(p)
        view = S.render_view(src, OLDLR, UP, B0268, pose, mask)
        assert view.image.shape == (B0268.height, B0268.width, 3) and view.valid.mean() > 0.3
        # decode the colour at valid pixels back to a direction and compare with the virtual camera's own ray
        u, v = np.meshgrid(np.arange(B0268.width, dtype=float), np.arange(B0268.height, dtype=float))
        expected = pose.cam_to_enu(B0268.pixel_to_ray(u, v))
        decoded = view.image.astype(float) / 255.0 * 2.0 - 1.0
        sel = view.valid & (expected[..., 2] > 0.05)
        err = np.degrees(np.arccos(np.clip(np.sum(decoded[sel] * expected[sel], axis=-1)
                                           / np.linalg.norm(decoded[sel], axis=-1), -1, 1)))
        assert np.median(err) < 1.5, np.median(err)                     # 8-bit colour quantisation ~ 1 deg


def test_view_ray_map_and_zenith_direction():
    pose = S.pointing_pose(S.Pointing(180.0, 40.0))
    valid = np.ones((B0268.height, B0268.width), dtype=bool)
    rm = S.view_ray_map(B0268, pose, 180.0, 50.0, valid, out_size=512)
    assert rm.shape == (4, 32, 32)
    zen = zenith_angle_deg(rm)
    assert abs(np.nanmin(zen) - (90 - 40 - 52.5 * 3496 / 4656)) < 8.0   # top edge of the view
    assert rm[3].mean() > 0.5
    # the Sun (azimuth 180, zenith 50 = elevation 40) is on the axis: the centre patch looks at it
    sun = pose.enu_to_cam(azel_to_enu(180.0, 40.0))
    assert np.allclose(sun, [0, 0, 1], atol=1e-9)
    assert abs(rm[0, 16, 16] - np.sin(np.radians(50))) < 0.05 and abs(rm[1, 16, 16]) < 0.05


def test_source_footprint_is_inside_the_source_when_the_view_is_high():
    pose = S.pointing_pose(S.Pointing(90.0, 60.0))
    us, vs = S.source_footprint(B0268, pose, OLDLR, UP)
    assert np.isfinite(us).mean() > 0.9
    assert np.nanmin(us) > 0 and np.nanmax(us) < OLDLR.width


def test_inherit_label_nearest_grid_point():
    times = pd.date_range("2022-04-01 10:00:00", periods=5, freq="30s", tz="UTC")
    t = pd.DataFrame({"ceilometer": "CDLRA", "time": times, "label": ["low", "none", "mid", "mid", "high"],
                      "etage_wmo": ["low", "none", "mid", "mid", "high"], "cbh_m": [900.0, np.nan, 3000.0, 3100.0, 7000.0],
                      "cbh2_m": [np.nan] * 5, "layers": [1.0, 0.0, 1.0, 1.0, 1.0], "confidence": [1.0, 1.0, 0.8, 0.9, 1.0],
                      "valid": [True, True, False, True, True], "mixed": [False] * 5})
    lab = S.inherit_label(t, "2022-04-01T10:00:10Z")
    assert lab["label"] == "low" and lab["cbh_m"] == 900.0 and lab["gap_s"] == 10 and lab["assumption"] == "flat layer"
    assert S.inherit_label(t, "2022-04-01T10:01:05Z") is None             # nearest grid point is invalid
    assert S.inherit_label(t, "2022-04-01T10:03:00Z") is None             # 60 s away: no grid point within 15 s
    assert S.inherit_label(t.iloc[0:0], "2022-04-01T10:00:00Z") is None
