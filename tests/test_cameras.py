"""P043: camera models on one interface, cross-checked against OpenCV and the shared test vectors."""

import json
from pathlib import Path

import numpy as np
import pytest

from stratia.geometry.cameras import (
    MODEL_KINDS,
    CameraModel,
    OpenCVCamera,
    UnknownCamera,
    camera_for,
    load_camera_registry,
    load_cloudscope_model,
    nominal_fisheye,
    registry_entry,
    validate_cloudscope_model,
)
from stratia.geometry.ocam import OcamModel

REPO = Path(__file__).resolve().parents[1]

OLDLR = OcamModel(ss=(-642.6690459406254, 0.0, 0.0002441927468927244, 4.838462720720562e-07),
                  xc=1028.60, yc=1072.59, width=2112, height=2048)
FISHEYE = OpenCVCamera("opencv_fisheye", 640, 480, 300.0, 302.0, 325.0, 236.0, (-0.05, 0.01, -0.002, 0.0005))
PINHOLE = OpenCVCamera("opencv_pinhole", 640, 480, 520.0, 518.0, 322.0, 241.0, (-0.20, 0.05, 0.001, -0.0005, 0.0))


def _grid(cam, n=25, margin=0.05):
    u = np.linspace(margin * cam.width, (1 - margin) * cam.width, n)
    v = np.linspace(margin * cam.height, (1 - margin) * cam.height, n)
    uu, vv = np.meshgrid(u, v)
    return uu.ravel(), vv.ravel()


@pytest.mark.parametrize("cam", [OLDLR, FISHEYE, PINHOLE], ids=["ocam", "fisheye", "pinhole"])
def test_interface_and_round_trip(cam):
    assert isinstance(cam, CameraModel) and cam.kind in MODEL_KINDS and cam.calibrated
    u, v = _grid(cam)
    rays = cam.pixel_to_ray(u, v)
    assert rays.shape == (u.size, 3) and np.allclose(np.linalg.norm(rays, axis=-1), 1.0)
    uu, vv = cam.ray_to_pixel(rays)
    err = np.hypot(uu - u, vv - v)
    assert np.isfinite(err).all() and err.max() < 0.01, err.max()


@pytest.mark.parametrize("cam", [FISHEYE, PINHOLE], ids=["fisheye", "pinhole"])
def test_opencv_models_match_cv2(cam):
    cv2 = pytest.importorskip("cv2")
    u, v = _grid(cam, n=15)
    rays = cam.pixel_to_ray(u, v)
    k = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
    d = np.array(cam.distortion, dtype=float)
    pts = rays.reshape(-1, 1, 3)
    if cam.model == "opencv_fisheye":
        proj, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), k, d.reshape(4, 1))
        undist = cv2.fisheye.undistortPoints(np.stack([u, v], -1).reshape(-1, 1, 2), k, d.reshape(4, 1),
                                             criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12))
    else:
        proj, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), k, d)
        undist = cv2.undistortPointsIter(np.stack([u, v], -1).reshape(-1, 1, 2), k, d, None, None,
                                         (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12))
    assert np.allclose(proj.reshape(-1, 2), np.stack([u, v], -1), atol=1e-3)
    ours = rays[:, :2] / rays[:, 2:3]
    assert np.allclose(undist.reshape(-1, 2), ours, atol=1e-6)


def test_axes_follow_adr_012():
    # The pinhole's tangential terms (p1, p2) bend an on-axis-row pixel slightly off the axis: 1e-3 for it.
    for cam, tol in ((FISHEYE, 1e-9), (PINHOLE, 1e-3)):
        centre = cam.pixel_to_ray(cam.cx, cam.cy)
        right = cam.pixel_to_ray(cam.cx + 100, cam.cy)
        down = cam.pixel_to_ray(cam.cx, cam.cy + 100)
        assert np.allclose(centre, [0, 0, 1], atol=1e-9)
        assert right[0] > 0 and abs(right[1]) < tol and down[1] > 0 and abs(down[0]) < tol


def test_directions_behind_the_camera_do_not_project():
    u, v = FISHEYE.ray_to_pixel(np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]))
    assert np.isnan(u).all() and np.isnan(v).all()


def test_unknown_camera_is_explicitly_uncalibrated():
    cam = UnknownCamera(640, 480)
    assert cam.kind == "unknown" and not cam.calibrated
    assert np.isnan(cam.pixel_to_ray(np.array([1.0, 2.0]), np.array([3.0, 4.0]))).all()
    assert all(np.isnan(x).all() for x in cam.ray_to_pixel(np.array([[0.0, 0.0, 1.0]])))


def test_nominal_fisheye_field_of_view():
    cam = nominal_fisheye(4656, 3496, 105.0, "b0268", "nominal")
    left = cam.pixel_to_ray(0.0, cam.cy)
    assert abs(np.degrees(np.arccos(left[2])) - 52.5) < 0.05


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError):
        OpenCVCamera("opencv_fisheye", 640, 480, 300, 300, 320, 240, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        OpenCVCamera("opencv_pinhole", 640, 480, 300, 300, 320, 240, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        OpenCVCamera("equirectangular", 640, 480, 300, 300, 320, 240, (0.0,) * 4)


def test_cloudscope_file_validation_and_loading(tmp_path):
    good = json.loads((REPO / "configs" / "cameras" / "b0268_nominal.camera.json").read_text())
    assert validate_cloudscope_model(good) == []
    cam = load_cloudscope_model(REPO / "configs" / "cameras" / "b0268_nominal.camera.json")
    assert cam.kind == "opencv_fisheye" and (cam.width, cam.height) == (4656, 3496)
    bad = dict(good, model="opencv_pinhole")                       # 4 coefficients are legal for a pinhole
    assert validate_cloudscope_model(bad) == []
    bad = dict(good, distortion=[0.0] * 5)                         # 5 on a fisheye are not
    assert any("fisheye" in p for p in validate_cloudscope_model(bad))
    bad = dict(good, schema="cloudscope.camera_model/2")
    assert any("schema" in p for p in validate_cloudscope_model(bad))
    bad = {k: v for k, v in good.items() if k != "fit"}
    assert validate_cloudscope_model(bad) == ["missing field 'fit'"]
    f = tmp_path / "broken.json"
    f.write_text(json.dumps(dict(good, intrinsics=dict(good["intrinsics"], fx=0))))
    with pytest.raises(ValueError):
        load_cloudscope_model(f)


def test_registry_resolves_every_manifest_camera_type():
    reg = load_camera_registry()
    assert registry_entry("eye2sky-AURIC", reg).model == "ocam"
    assert registry_entry("eye2sky-WESTE", reg).model == "ocam"
    b = registry_entry("b0268", reg)
    assert b.model == "opencv_fisheye" and b.provisional
    assert registry_entry("ccsn-various", reg).model == "unknown"
    assert registry_entry("never-seen", reg).model == "unknown"
    assert isinstance(camera_for("mgcd-asi", 1024, 1024), UnknownCamera)
    assert camera_for("eye2sky-AURIC", 2112, 2048, calib_id=None).kind == "unknown"   # no calibration selected
    cam = camera_for("b0268", 4656, 3496)
    assert cam.kind == "opencv_fisheye"
    with pytest.raises(ValueError):
        camera_for("b0268", 1920, 1080)                              # the nominal model is for the full frame


def test_shared_test_vectors():
    """The vectors file is what CloudScope must reproduce (ADR-012): recompute every entry here."""
    vectors = json.loads((REPO / "tests" / "vectors" / "camera_test_vectors.json").read_text(encoding="utf-8"))
    assert vectors["conventions"]["camera_frame"].startswith("+x toward +u")
    for case in vectors["cases"]:
        p = case["camera"]
        if p["kind"] == "ocam":
            cam = OcamModel(ss=tuple(p["ss"]), xc=p["xc"], yc=p["yc"], c=p["c"], d=p["d"], e=p["e"],
                            width=p["width"], height=p["height"])
        else:
            cam = OpenCVCamera(p["kind"], p["width"], p["height"], p["fx"], p["fy"], p["cx"], p["cy"], tuple(p["distortion"]))
        for s in case["samples"]:
            ray = cam.pixel_to_ray(s["u"], s["v"])
            assert np.allclose(ray, s["ray"], atol=1e-9), (case["name"], s)
            u, v = cam.ray_to_pixel(np.array(s["ray"]))
            assert abs(u - s["u"]) < 1e-6 and abs(v - s["v"]) < 1e-6, (case["name"], s)
