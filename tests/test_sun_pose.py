"""P044: Sun position, ENU conversions, rotation fitting, Eye2Sky orientation readings and Sun-disc detection."""

from datetime import UTC, datetime

import numpy as np
import pytest

from stratia.geometry import pose as P
from stratia.geometry.ocam import OcamModel
from stratia.geometry.sun import azel_to_enu, enu_to_azel, sun_aligned_frame, sun_position
from stratia.geometry.sun_detect import detect_sun

OLDLR = OcamModel(ss=(-642.6690459406254, 0.0, 0.0002441927468927244, 4.838462720720562e-07),
                  xc=1028.60, yc=1072.59, width=2112, height=2048)


def test_sun_position_oldenburg_solstice_noon():
    # Solar noon at 8.167 E on 21 June 2022 is about 11:29 UTC; elevation 90 - 53.15 + 23.44 = 60.3 deg.
    sp = sun_position([datetime(2022, 6, 21, 11, 29, tzinfo=UTC)], 53.15137, 8.16701, 30)
    assert abs(sp.azimuth.iloc[0] - 180) < 1.5 and abs(sp.elevation.iloc[0] - 60.3) < 0.3
    assert sp.apparent_elevation.iloc[0] >= sp.elevation.iloc[0]
    naive = sun_position(["2022-06-21 11:29"], 53.15137, 8.16701)
    assert np.isclose(naive.azimuth.iloc[0], sp.azimuth.iloc[0])


def test_enu_round_trip_and_axes():
    assert np.allclose(azel_to_enu(0, 0), [0, 1, 0]) and np.allclose(azel_to_enu(90, 0), [1, 0, 0])
    assert np.allclose(azel_to_enu(123, 90), [0, 0, 1], atol=1e-12)
    az = np.array([0.0, 45.0, 180.0, 359.0])
    el = np.array([5.0, 30.0, 60.0, 89.0])
    az2, el2 = enu_to_azel(azel_to_enu(az, el))
    assert np.allclose(az2, az) and np.allclose(el2, el)


def test_sun_aligned_frame_puts_the_sun_on_x():
    r = sun_aligned_frame(sun_azimuth_deg=250.0)
    assert np.allclose(np.linalg.det(r), 1.0)
    s = r @ azel_to_enu(250.0, 40.0)
    assert abs(s[1]) < 1e-12 and s[0] > 0 and np.isclose(s[2], np.sin(np.radians(40.0)))
    assert np.allclose(r @ np.array([0, 0, 1.0]), [0, 0, 1])


def test_euler_helpers_and_zyx_round_trip():
    r = P.euler_to_rotation(0.1, -0.4, 2.0, "zyx")
    assert np.allclose(r, P.rotation_z(2.0) @ P.rotation_y(-0.4) @ P.rotation_x(0.1))
    assert np.allclose(np.linalg.det(r), 1.0)
    assert np.allclose(P.rotation_to_euler_zyx(r), (0.1, -0.4, 2.0))


def test_fit_rotation_recovers_truth_despite_outliers():
    rng = np.random.default_rng(1)
    truth = P.euler_to_rotation(-0.02, 3.13, 1.4, "zyx")
    cam = rng.normal(size=(300, 3))
    cam /= np.linalg.norm(cam, axis=1, keepdims=True)
    enu = cam @ truth.T
    enu[:30] = rng.normal(size=(30, 3))                       # 10 % garbage observations
    enu[:30] /= np.linalg.norm(enu[:30], axis=1, keepdims=True)
    r, inlier, err = P.fit_rotation(cam, enu)
    assert P.rotation_distance_deg(r, truth) < 1e-6
    assert inlier[30:].all() and not inlier[:30].any()
    assert np.median(err[inlier]) < 1e-9
    with pytest.raises(ValueError):
        P.fit_rotation(cam[:2], enu[:2])


def test_eye2sky_candidates_are_rotations_and_distinct():
    assert len(P.EYE2SKY_CANDIDATES) == 6 * 8 * 4 * 2 * 2
    ext = (-0.0107, 3.1405, 1.3656)                            # AURIC's declared orientation
    mats = [P.eye2sky_rotation(ext, c) for c in P.EYE2SKY_CANDIDATES[:64]]
    assert all(np.allclose(m @ m.T, np.eye(3)) and np.isclose(np.linalg.det(m), 1.0) for m in mats)
    # Any upward-looking reading must put the optical axis near the zenith for pitch ~ pi.
    up = sum(abs(m[2, 2]) > 0.99 for m in mats)
    assert up > 0


def test_pose_records_round_trip_and_fallback(tmp_path):
    f = tmp_path / "poses.yaml"
    r = P.euler_to_rotation(0.0, np.pi - 0.01, 1.0, "zyx")
    P.save_pose_records({"X_1.yaml": {"status": "ok", "R": r.tolist()}, "Y_1.yaml": {"status": "excluded", "R": r.tolist()}},
                        {"eye2sky_convention": "zyx|+r+p+y|ENU|std|c2w"}, f)
    recs, meta = P.load_pose_records(f), P.load_pose_meta(f)
    assert meta["eye2sky_convention"] == "zyx|+r+p+y|ENU|std|c2w"
    fitted = P.pose_for("X_1.yaml", records=recs)
    assert fitted.source == "fitted" and np.allclose(fitted.R, r)
    assert P.pose_for("Y_1.yaml", records=recs) is None                 # excluded, no declared orientation
    assert P.pose_for("Z_1.yaml", external_orientation=None, records=recs) is None
    assert P.load_pose_records(tmp_path / "missing.yaml") == {}


def _synthetic_sky(u_sun, v_sun, radius=9, size=(600, 700)):
    h, w = size
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    yy, xx = np.mgrid[:h, :w]
    mask = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 < (0.45 * min(h, w)) ** 2
    img[((xx - u_sun) ** 2 + (yy - v_sun) ** 2) <= radius**2] = 255        # the Sun
    img[10:40, 10:200] = 255                                              # burned-in text block outside the mask
    img[300:302, 100:101] = 255                                           # a 2-pixel speck
    img[~mask] = 0
    img[10:40, 10:200] = 255
    return img, mask


def test_detect_sun_synthetic():
    img, mask = _synthetic_sky(420.0, 250.0)
    det = detect_sun(img, mask)
    assert det.ok and abs(det.u - 420) < 0.5 and abs(det.v - 250) < 0.5 and det.candidates == 1
    assert not detect_sun(img, mask, threshold=256).ok                     # nothing saturated
    two, _ = _synthetic_sky(420.0, 250.0)
    yy, xx = np.mgrid[:600, :700]
    two[((xx - 200) ** 2 + (yy - 400) ** 2) <= 9**2] = 255                  # a second disc of the same size
    assert not detect_sun(two, mask).ok
    assert not detect_sun(img, mask, min_area=10_000).ok
    big, _ = _synthetic_sky(420.0, 250.0, radius=30)                        # a glow-sized blob is refused
    assert not detect_sun(big, mask).ok and detect_sun(big, mask, max_area=5000).ok


def test_end_to_end_sun_pipeline_on_a_synthetic_station():
    """Project the real Sun track through OLDLR's model under a chosen pose, detect it and recover the pose."""
    truth = P.euler_to_rotation(0.03, -0.02, 1.5, "zyx")                   # optical axis 2 deg off the zenith
    times = [datetime(2022, 5, 10, h, m, tzinfo=UTC) for h in range(6, 17) for m in (0, 30)]
    sp = sun_position(times, 53.15137, 8.16701, 30)
    enu = azel_to_enu(sp.azimuth.to_numpy(), sp.apparent_elevation.to_numpy())
    cam_rays = enu @ truth                                                  # ENU -> camera (R^T)
    u, v = OLDLR.ray_to_pixel(cam_rays)
    keep = np.isfinite(u)
    det_u, det_v = [], []
    for uu, vv in zip(u[keep], v[keep], strict=True):
        img = np.zeros((2048 // 2, 2112 // 2, 3), dtype=np.uint8)              # half resolution, as the script reads
        yy, xx = np.mgrid[: 2048 // 2, : 2112 // 2]
        img[((xx - (uu - 0.5) / 2) ** 2 + (yy - (vv - 0.5) / 2) ** 2) <= 10**2] = 255
        d = detect_sun(img)
        assert d.ok, (uu, vv)
        det_u.append(2 * d.u + 0.5), det_v.append(2 * d.v + 0.5)
    rays = OLDLR.pixel_to_ray(np.array(det_u), np.array(det_v))
    assert keep.sum() >= 15
    r, inlier, err = P.fit_rotation(rays, enu[keep])
    assert P.rotation_distance_deg(r, truth) < 0.15 and np.median(err) < 0.15
