"""P046: transforms keep image and geometry consistent (ADR-004)."""

import cv2
import numpy as np

from stratia.geometry import raymap as RM
from stratia.geometry import transforms as T
from stratia.geometry.ocam import OcamModel
from stratia.geometry.pose import Pose, euler_to_rotation
from stratia.geometry.sun import azel_to_enu

OLDLR = OcamModel(ss=(-642.6690459406254, 0.0, 0.0002441927468927244, 4.838462720720562e-07),
                  xc=1028.60, yc=1072.59, width=2112, height=2048)
UP = Pose(R=euler_to_rotation(0.01, -0.02, 0.9, "zyx"), source="test")
SUN_AZ, SUN_ZEN = 230.0, 48.0


def _coordinate_image(w, h):
    """Float32 image whose channels encode the pixel's own (x, y): decoding a warped image reads source coordinates."""
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return np.stack([xx, yy, np.zeros_like(xx)], axis=-1)


def _disc_mask(cam, radius=950):
    yy, xx = np.mgrid[: cam.height, : cam.width]
    return (xx - cam.yc) ** 2 + (yy - cam.xc) ** 2 < radius**2


def _chain(rng):
    return T.Compose([T.Resize(640, 620), T.Crop(60, 40, 560, 560), T.Rotate((-30, 30)), T.HorizontalFlip(), T.Resize(512)])


def test_composed_map_matches_the_warped_pixels():
    rng = np.random.default_rng(0)
    img = _coordinate_image(900, 700)
    out, state = _chain(rng)(img, T.GeoState.identity(900, 700), rng)
    assert out.shape == (512, 512, 3) and state.width == 512 and state.height == 512
    cu, cv_ = RM.patch_centres(512, 512)
    xs, ys = state.to_source(cu, cv_)
    decoded_x = cv2.remap(out[..., 0], cu.astype(np.float32), cv_.astype(np.float32), cv2.INTER_LINEAR)
    decoded_y = cv2.remap(out[..., 1], cu.astype(np.float32), cv_.astype(np.float32), cv2.INTER_LINEAR)
    covered = T.patch_fraction(state.coverage) == 1.0
    assert covered.sum() > 400
    err = np.hypot(decoded_x - xs, decoded_y - ys)[covered]
    assert err.max() < 1.0, err.max()                        # interpolation of a linear ramp is exact up to rounding
    # round trip through the inverse map
    bx, by = state.from_source(xs, ys)
    assert np.allclose(bx, cu) and np.allclose(by, cv_)


def test_cached_state_equals_direct_full_frame_resize():
    cached = T.GeoState.cached(768, 745, 2112, 2048)
    img = np.zeros((745, 768, 3), dtype=np.uint8)
    _, state = T.Resize(512)(img, cached)
    direct = RM.full_frame_affine(2112, 2048, 512, 512)
    assert np.allclose(state.affine, direct, atol=1e-9)


def _sun_patch(state, model, pose):
    sun_cam = pose.enu_to_cam(azel_to_enu(SUN_AZ, 90 - SUN_ZEN))
    su, sv = model.ray_to_pixel(sun_cam)
    ou, ov = state.from_source(su, sv)
    return int(ov // 16), int(ou // 16), ou, ov


def test_sun_stays_at_the_sun_after_any_chain():
    rng = np.random.default_rng(3)
    img = np.zeros((2048, 2112, 3), dtype=np.uint8)
    mask = _disc_mask(OLDLR)
    for _ in range(5):
        chain = T.Compose([T.RandomResizedCrop(640, scale=(0.4, 1.0)), T.Rotate((-180, 180)), T.HorizontalFlip(p=0.5),
                           T.Resize(512)])
        _, state = chain(img, T.GeoState.identity(2112, 2048), rng)
        i, j, ou, ov = _sun_patch(state, OLDLR, UP)
        if not (0 <= i < 32 and 0 <= j < 32) or not state.coverage[int(ov), int(ou)]:
            continue                                            # the Sun left the crop: nothing to check
        rm, mask_out = T.geometry_for(state, OLDLR, UP, SUN_AZ, SUN_ZEN, mask)
        assert rm.shape == (4, 32, 32) and mask_out.shape == (512, 512)
        if rm[3, i, j] == 0:
            continue                                            # masked patch (near the disc edge)
        assert RM.sun_distance_deg(rm, SUN_ZEN)[i, j] < 6.0
        assert abs(RM.azimuth_from_sun_deg(rm)[i, j]) < 15.0


def test_flip_mirrors_the_zenith_map_exactly():
    img = np.zeros((2048, 2112, 3), dtype=np.uint8)
    _, base = T.Resize(512)(img, T.GeoState.identity(2112, 2048))
    _, flipped = T.HorizontalFlip()(np.zeros((512, 512, 3), np.uint8), base)
    rm0, _ = T.geometry_for(base, OLDLR, UP, SUN_AZ, SUN_ZEN)
    rm1, _ = T.geometry_for(flipped, OLDLR, UP, SUN_AZ, SUN_ZEN)
    z0, z1 = RM.zenith_angle_deg(rm0), RM.zenith_angle_deg(rm1)
    assert np.allclose(np.nan_to_num(z1, nan=-1), np.nan_to_num(z0[:, ::-1], nan=-1), atol=1e-9)
    # the Sun-relative azimuth flips sign for a mirrored layout only where the direction itself is mirrored: here
    # directions are carried, not mirrored, so the azimuth field is the mirror image too
    a0, a1 = RM.azimuth_from_sun_deg(rm0), RM.azimuth_from_sun_deg(rm1)
    assert np.allclose(np.nan_to_num(a1, nan=0), np.nan_to_num(a0[:, ::-1], nan=0), atol=1e-9)


def test_rotation_rotates_the_zenith_map():
    img = np.zeros((2048, 2112, 3), dtype=np.uint8)
    _, base = T.Resize(512)(img, T.GeoState.identity(2112, 2048))
    rm0, _ = T.geometry_for(base, OLDLR, UP, SUN_AZ, SUN_ZEN)
    # dense zenith-angle image at 512 px from the base state, rotated like the image by the same transform
    u, v = np.meshgrid(np.arange(512, dtype=float), np.arange(512, dtype=float))
    xs, ys = base.to_source(u, v)
    dense = np.degrees(np.arccos(np.clip(UP.cam_to_enu(OLDLR.pixel_to_ray(xs, ys))[..., 2], -1, 1))).astype(np.float32)
    rotated_dense, rot_state = T.Rotate(37.0)(dense, base)
    rm1, _ = T.geometry_for(rot_state, OLDLR, UP, SUN_AZ, SUN_ZEN)
    z1 = RM.zenith_angle_deg(rm1)
    sampled = rotated_dense.reshape(32, 16, 32, 16)[:, 7:9, :, 7:9].mean(axis=(1, 3))   # value at the patch centre
    both = (rm1[3] > 0) & (T.patch_fraction(rot_state.coverage) == 1.0) & (sampled < 80)
    assert both.sum() > 300
    assert np.abs(z1[both] - sampled[both]).max() < 1.0


def test_padding_is_invalid_in_ray_map_and_mask():
    img = np.zeros((2048, 2112, 3), dtype=np.uint8)
    _, state = T.Compose([T.Resize(512), T.Crop(-128, -128, 512, 512)])(img, T.GeoState.identity(2112, 2048))
    rm, mask_out = T.geometry_for(state, OLDLR, UP, SUN_AZ, SUN_ZEN, _disc_mask(OLDLR))
    assert rm[3, :8, :].sum() == 0 and rm[3, :, :8].sum() == 0 and rm[3, 8:, 8:].sum() > 0
    assert not mask_out[:128, :].any() and not mask_out[:, :128].any() and mask_out[128:, 128:].any()
    assert np.all(rm[:3][:, rm[3] == 0] == 0)


def test_random_resized_crop_box_is_inside_the_image():
    rng = np.random.default_rng(5)
    t = T.RandomResizedCrop(224, scale=(0.3, 1.0), ratio=(0.5, 2.0))
    for _ in range(200):
        x0, y0, w, h = t.sample_box(640, 480, rng)
        assert 0 <= x0 and 0 <= y0 and x0 + w <= 640 and y0 + h <= 480 and w > 0 and h > 0
    out, state = t(np.zeros((480, 640, 3), np.uint8), T.GeoState.identity(640, 480), rng)
    assert out.shape == (224, 224, 3) and state.coverage.all()


def test_uncalibrated_camera_gives_zero_ray_map_but_a_coverage_mask():
    from stratia.geometry.cameras import UnknownCamera

    _, state = T.Compose([T.Resize(512), T.Rotate(20.0)])(np.zeros((400, 600, 3), np.uint8), T.GeoState.identity(600, 400))
    rm, mask_out = T.geometry_for(state, UnknownCamera(600, 400), None, None, None)
    assert not rm.any() and mask_out.dtype == bool and 0.5 < mask_out.mean() < 1.0
