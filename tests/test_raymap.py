"""P045: ray maps follow the contract and the geometry they are built from."""

import numpy as np
import pytest

from stratia.contract import DEFAULT_INPUT_SIZE, PATCH_SIZE, RAY_MAP_CHANNELS
from stratia.geometry import raymap as RM
from stratia.geometry.cameras import OpenCVCamera, UnknownCamera, nominal_fisheye
from stratia.geometry.ocam import OcamModel
from stratia.geometry.pose import Pose, euler_to_rotation
from stratia.geometry.sun import azel_to_enu

OLDLR = OcamModel(ss=(-642.6690459406254, 0.0, 0.0002441927468927244, 4.838462720720562e-07),
                  xc=1028.60, yc=1072.59, width=2112, height=2048)
UP = Pose(R=euler_to_rotation(0.0, 0.0, 0.7, "zyx"), source="test")       # zenith camera, rotated 40 deg about it


def _disc_mask(cam, radius=950):
    yy, xx = np.mgrid[: cam.height, : cam.width]
    return (xx - cam.yc) ** 2 + (yy - cam.xc) ** 2 < radius**2


def test_full_frame_affine_maps_pixel_centres():
    a = RM.full_frame_affine(2112, 2048, 512, 512)
    xs, ys = RM.apply_affine(a, np.array([0.0, 511.0]), np.array([0.0, 511.0]))
    # output pixel 0 covers source pixels 0..4.125: its centre maps to the centre of that span
    assert np.isclose(xs[0], 2112 / 512 / 2 - 0.5) and np.isclose(ys[0], 2048 / 512 / 2 - 0.5)
    assert np.isclose(xs[1], 2112 - 2112 / 512 / 2 - 0.5) and np.isclose(ys[1], 2048 - 2048 / 512 / 2 - 0.5)


def test_patch_centres_shape_and_positions():
    cu, cv = RM.patch_centres(512, 512, 16)
    assert cu.shape == (32, 32) and cu[0, 0] == 7.5 and cv[0, 0] == 7.5 and cu[0, 1] == 23.5 and cv[1, 0] == 23.5
    with pytest.raises(ValueError):
        RM.patch_centres(500, 512, 16)


def test_ray_map_contract_shape_and_values():
    mask = _disc_mask(OLDLR)
    rm = RM.ray_map(OLDLR, UP, sun_azimuth_deg=200.0, sun_zenith_deg=50.0, mask=mask)
    assert rm.shape == (RAY_MAP_CHANNELS, DEFAULT_INPUT_SIZE // PATCH_SIZE, DEFAULT_INPUT_SIZE // PATCH_SIZE)
    assert rm.dtype == np.float32
    valid = rm[3] > 0
    assert set(np.unique(rm[3])) <= {0.0, 1.0} and 0.3 < valid.mean() < 0.8
    norms = np.linalg.norm(rm[:3], axis=0)
    assert np.allclose(norms[valid], 1.0, atol=1e-6) and np.all(norms[~valid] == 0)
    assert np.all(rm[2][valid] > 0)                                   # above the horizon
    # The optical axis is the zenith: the patch holding the image centre has z ~ 1.
    a = RM.full_frame_affine(OLDLR.width, OLDLR.height)
    cu, cv = RM.patch_centres(512, 512)
    xs, ys = RM.apply_affine(a, cu, cv)
    i, j = np.unravel_index(np.argmin((xs - OLDLR.yc) ** 2 + (ys - OLDLR.xc) ** 2), xs.shape)
    assert rm[2, i, j] > 0.995                                       # within ~6 deg: a patch is ~6 deg wide here


def test_sun_patch_points_at_the_sun():
    """The patch where the Sun projects must have y ~ 0 and x, z of the Sun's own direction."""
    az, zen = 215.0, 42.0
    sun_cam = UP.enu_to_cam(azel_to_enu(az, 90 - zen))
    su, sv = OLDLR.ray_to_pixel(sun_cam)
    a = RM.full_frame_affine(OLDLR.width, OLDLR.height)
    cu, cv = RM.patch_centres(512, 512)
    xs, ys = RM.apply_affine(a, cu, cv)
    i, j = np.unravel_index(np.argmin((xs - su) ** 2 + (ys - sv) ** 2), xs.shape)
    rm = RM.ray_map(OLDLR, UP, az, zen)
    assert rm[3, i, j] == 1
    # a patch is 66 source pixels wide here (~6 deg at the centre), so the centre is within a few degrees of the Sun
    assert RM.sun_distance_deg(rm, zen)[i, j] < 4.0
    assert abs(RM.azimuth_from_sun_deg(rm)[i, j]) < 10.0
    assert abs(RM.zenith_angle_deg(rm)[i, j] - zen) < 4.0


def test_invalid_cases_give_zero_tensors():
    zeros = np.zeros((4, 32, 32), dtype=np.float32)
    assert np.array_equal(RM.ray_map(UnknownCamera(640, 480), UP, 100.0, 40.0), zeros)
    assert np.array_equal(RM.ray_map(OLDLR, None, 100.0, 40.0), zeros)
    assert np.array_equal(RM.ray_map(OLDLR, UP, None, 40.0), zeros)
    assert np.array_equal(RM.ray_map(OLDLR, UP, 100.0, float("nan")), zeros)
    assert np.array_equal(RM.meta_vector(None), np.zeros(3, dtype=np.float32))
    m = RM.meta_vector(60.0)
    assert np.allclose(m, [0.5, np.sqrt(3) / 2, 1.0]) and m.dtype == np.float32


def test_mask_and_horizon_remove_patches():
    full = RM.ray_map(OLDLR, UP, 180.0, 40.0)
    masked = RM.ray_map(OLDLR, UP, 180.0, 40.0, mask=_disc_mask(OLDLR, radius=600))
    assert masked[3].sum() < full[3].sum()
    assert np.all(masked[3] <= full[3])
    raised = RM.ray_map(OLDLR, UP, 180.0, 40.0, min_elevation_deg=30.0)
    assert np.all(RM.zenith_angle_deg(raised)[raised[3] > 0] <= 60.0 + 1e-6)
    assert raised[3].sum() < full[3].sum()


def test_patch_rays_mask_fraction_is_a_share_of_the_footprint():
    mask = np.zeros((2048, 2112), dtype=bool)
    mask[:, : 2112 // 2] = True                                       # left half only
    pr = RM.patch_rays(OLDLR, mask=mask)
    assert pr.shape == (32, 32)
    assert np.all(pr.mask_fraction[:, :15] == 1.0) and np.all(pr.mask_fraction[:, 17:] == 0.0)
    assert 0.0 < pr.mask_fraction[:, 15:17].mean() < 1.0


def test_resized_mask_matches_nearest_source_pixel():
    mask = _disc_mask(OLDLR)
    small = RM.resized_mask(mask, 512)
    assert small.shape == (512, 512) and 0.4 < small.mean() < 0.8
    # the resized disc is centred where the source disc is
    yy, xx = np.nonzero(small)
    assert abs(xx.mean() - (OLDLR.yc + 0.5) * 512 / 2112 + 0.5) < 1.5
    assert abs(yy.mean() - (OLDLR.xc + 0.5) * 512 / 2048 + 0.5) < 1.5


def test_consumer_camera_pointed_at_the_horizon():
    """A pointed camera (B0268-like): optical axis at azimuth 180, elevation 30; image up = toward the zenith."""
    cam = nominal_fisheye(4656, 3496, 105.0, "b0268", "nominal")
    z = azel_to_enu(180.0, 30.0)                                      # camera +z (optical axis) in ENU
    x = np.array([-1.0, 0.0, 0.0])                                    # camera +x (image right) = west when facing south
    y = np.cross(z, x)                                                # camera +y (image down) = toward the ground
    pose = Pose(R=np.stack([x, y, z], axis=1), source="test")
    assert np.allclose(pose.R @ pose.R.T, np.eye(3)) and np.isclose(np.linalg.det(pose.R), 1.0)
    rm = RM.ray_map(cam, pose, 180.0, 60.0, out_size=(512, 512))
    zen = RM.zenith_angle_deg(rm)
    assert abs(zen[16, 16] - 60.0) < 3.0                              # the centre patch looks along the axis
    assert np.nanmean(zen[0]) < np.nanmean(zen[16]) < 90.0            # top rows look higher up
    assert rm[3, 0].all() and not rm[3, 31].all()                     # bottom rows reach below the horizon


def test_opencv_camera_ray_map_runs():
    cam = OpenCVCamera("opencv_fisheye", 640, 480, 300.0, 302.0, 325.0, 236.0, (-0.05, 0.01, -0.002, 0.0005))
    rm = RM.ray_map(cam, UP, 90.0, 30.0, out_size=(480, 640), patch=16)
    assert rm.shape == (4, 30, 40) and rm[3].mean() > 0.9


def test_near_horizon_table():
    table = RM.near_horizon_table()
    assert RM.min_elevation_for("eye2sky-AURIC", table) == 5.0
    assert RM.min_elevation_for("b0268", table) == 0.0
    assert RM.min_elevation_for("ccsn-various", table) == 0.0
