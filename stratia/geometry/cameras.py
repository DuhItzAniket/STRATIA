"""Camera models with one interface (P043): pixel <-> viewing direction.

Models
------
* ``OcamModel`` (``stratia.geometry.ocam``): Scaramuzza/OCamCalib, the Eye2Sky all-sky imagers.
* ``OpenCVCamera``: OpenCV's pinhole (k1 k2 p1 p2 k3) and equidistant fisheye (k1..k4) models, read from the
  CloudScope camera-model file (schema ``cloudscope.camera_model/1``, mirrored in ``schemas/camera_model.schema.json``).
  Both projections are written out in NumPy so the package does not depend on OpenCV for geometry; the tests
  cross-check them against ``cv2``.
* ``UnknownCamera``: the placeholder for images without a calibration (CCSN, MGCD, Montenegro, ...). It knows only
  the image size; ``calibrated`` is False and the ray map for it is all zeros (contract v1, docs/contract.md).

Conventions (CloudScope ADR-012, docs/contract.md): pixel (u, v) from the top-left pixel centre, u right, v down;
camera frame +x toward +u, +y toward +v, +z along the optical axis into the scene; rays are unit vectors.
``ray_to_pixel`` returns NaN where a direction is outside the model's range (behind the camera, beyond the fisheye's
monotonic range).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

CLOUDSCOPE_SCHEMA = "cloudscope.camera_model/1"
MODEL_KINDS = ("ocam", "opencv_pinhole", "opencv_fisheye", "unknown")


@runtime_checkable
class CameraModel(Protocol):
    """What every camera model offers. ``kind`` is one of MODEL_KINDS."""

    width: int
    height: int

    @property
    def kind(self) -> str: ...

    @property
    def calibrated(self) -> bool: ...

    def pixel_to_ray(self, u: np.ndarray, v: np.ndarray) -> np.ndarray: ...

    def ray_to_pixel(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


# ----------------------------------------------------------------------------------------------- unknown camera
@dataclass(frozen=True)
class UnknownCamera:
    """No calibration: directions are unknown. pixel_to_ray returns NaN rays; ray_to_pixel NaN pixels."""

    width: int
    height: int

    @property
    def kind(self) -> str:
        return "unknown"

    @property
    def calibrated(self) -> bool:
        return False

    def pixel_to_ray(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return np.full(np.broadcast(u, np.asarray(v, dtype=float)).shape + (3,), np.nan)

    def ray_to_pixel(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shape = np.asarray(rays, dtype=float).shape[:-1]
        return np.full(shape, np.nan), np.full(shape, np.nan)


# ----------------------------------------------------------------------------------------------- OpenCV models
@dataclass(frozen=True)
class OpenCVCamera:
    """OpenCV pinhole (distortion k1 k2 p1 p2 k3) or equidistant fisheye (k1 k2 k3 k4), zero skew."""

    model: str                      # "opencv_pinhole" | "opencv_fisheye"
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]
    camera_id: str = ""
    calibration_id: str = ""
    rms_px: float = float("nan")

    def __post_init__(self) -> None:
        if self.model not in ("opencv_pinhole", "opencv_fisheye"):
            raise ValueError(f"unknown OpenCV model {self.model!r}")
        n = len(self.distortion)
        if self.model == "opencv_fisheye" and n != 4:
            raise ValueError(f"opencv_fisheye needs 4 distortion coefficients, got {n}")
        if self.model == "opencv_pinhole" and n not in (4, 5):
            raise ValueError(f"opencv_pinhole needs 4 or 5 distortion coefficients, got {n}")
        if self.fx <= 0 or self.fy <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("focal lengths and image size must be positive")

    @property
    def kind(self) -> str:
        return self.model

    @property
    def calibrated(self) -> bool:
        return True

    # -- normalised <-> distorted image coordinates -------------------------------------------------------
    def _distort(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model == "opencv_fisheye":
            k1, k2, k3, k4 = self.distortion
            r = np.hypot(a, b)
            theta = np.arctan(r)
            t2 = theta * theta
            theta_d = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
            with np.errstate(invalid="ignore", divide="ignore"):
                scale = np.where(r > 0, theta_d / r, 1.0)
            return a * scale, b * scale
        k = list(self.distortion) + [0.0] * (5 - len(self.distortion))
        k1, k2, p1, p2, k3 = k
        r2 = a * a + b * b
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = a * radial + 2 * p1 * a * b + p2 * (r2 + 2 * a * a)
        yd = b * radial + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b
        return xd, yd

    def _undistort(self, xd: np.ndarray, yd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model == "opencv_fisheye":
            k1, k2, k3, k4 = self.distortion
            theta_d = np.hypot(xd, yd)
            theta = theta_d.copy()
            for _ in range(20):                                   # Newton on theta_d(theta), as cv::fisheye does
                t2 = theta * theta
                f = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4) - theta_d
                df = 1 + 3 * k1 * t2 + 5 * k2 * t2**2 + 7 * k3 * t2**3 + 9 * k4 * t2**4
                theta = theta - f / df
            with np.errstate(invalid="ignore", divide="ignore"):
                scale = np.where(theta_d > 0, np.tan(theta) / theta_d, 1.0)
            bad = (theta < 0) | (theta >= np.pi / 2)              # beyond the model's range
            scale = np.where(bad, np.nan, scale)
            return xd * scale, yd * scale
        k = list(self.distortion) + [0.0] * (5 - len(self.distortion))
        k1, k2, p1, p2, k3 = k
        a, b = xd.copy(), yd.copy()
        for _ in range(20):                                       # fixed point, as cv::undistortPoints does
            r2 = a * a + b * b
            inv_radial = 1.0 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
            da = 2 * p1 * a * b + p2 * (r2 + 2 * a * a)
            db = p1 * (r2 + 2 * b * b) + 2 * p2 * a * b
            a = (xd - da) * inv_radial
            b = (yd - db) * inv_radial
        return a, b

    # -- public interface ----------------------------------------------------------------------------------
    def pixel_to_ray(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        a, b = self._undistort((u - self.cx) / self.fx, (v - self.cy) / self.fy)
        rays = np.stack([a, b, np.ones_like(a)], axis=-1)
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    def ray_to_pixel(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rays = np.asarray(rays, dtype=float)
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            a, b = x / z, y / z
        xd, yd = self._distort(a, b)
        u, v = self.fx * xd + self.cx, self.fy * yd + self.cy
        behind = z <= 1e-9                                       # OpenCV's models stop at 90 deg off-axis
        return np.where(behind, np.nan, u), np.where(behind, np.nan, v)

    @property
    def max_theta_deg(self) -> float:
        """Largest off-axis angle that still projects inside the image (sampled along the image border)."""
        t = np.linspace(0, 1, 401)
        w, h = self.width - 1, self.height - 1
        border_u = np.concatenate([t * w, np.full_like(t, w), t * w, np.zeros_like(t)])
        border_v = np.concatenate([np.zeros_like(t), t * h, np.full_like(t, h), t * h])
        rays = self.pixel_to_ray(border_u, border_v)
        return float(np.nanmax(np.degrees(np.arccos(np.clip(rays[..., 2], -1, 1)))))


# ----------------------------------------------------------------------------------------------- CloudScope file
_REQUIRED = ("schema", "model", "image", "intrinsics", "distortion", "fit", "camera", "calibration_id", "calibrated_utc",
             "software")


def validate_cloudscope_model(d: dict) -> list[str]:
    """Checks mirroring schemas/camera_model.schema.json (no jsonschema dependency). Returns the problems found."""
    problems = [f"missing field {k!r}" for k in _REQUIRED if k not in d]
    if problems:
        return problems
    if d["schema"] != CLOUDSCOPE_SCHEMA:
        problems.append(f"schema is {d['schema']!r}, expected {CLOUDSCOPE_SCHEMA!r}")
    if d["model"] not in ("opencv_pinhole", "opencv_fisheye"):
        problems.append(f"model {d['model']!r} is not opencv_pinhole or opencv_fisheye")
    img, intr, dist = d["image"], d["intrinsics"], d["distortion"]
    for k in ("width", "height"):
        if not isinstance(img.get(k), int) or img[k] < 1:
            problems.append(f"image.{k} must be a positive integer")
    for k in ("fx", "fy"):
        if not isinstance(intr.get(k), int | float) or intr[k] <= 0:
            problems.append(f"intrinsics.{k} must be positive")
    for k in ("cx", "cy"):
        if not isinstance(intr.get(k), int | float):
            problems.append(f"intrinsics.{k} must be a number")
    if not isinstance(dist, list) or not all(isinstance(x, int | float) for x in dist):
        problems.append("distortion must be a list of numbers")
    elif d["model"] == "opencv_fisheye" and len(dist) != 4:
        problems.append(f"opencv_fisheye needs exactly 4 distortion coefficients, got {len(dist)}")
    elif d["model"] == "opencv_pinhole" and len(dist) not in (4, 5):
        problems.append(f"opencv_pinhole needs 4 or 5 distortion coefficients, got {len(dist)}")
    fit = d["fit"]
    if not isinstance(fit, dict) or "rms_px" not in fit or "views" not in fit or "board" not in fit:
        problems.append("fit must have rms_px, views and board")
    cam = d["camera"]
    if not isinstance(cam, dict) or not isinstance(cam.get("id"), str) or not isinstance(cam.get("name"), str):
        problems.append("camera must have string id and name")
    return problems


def load_cloudscope_model(path: str | Path) -> OpenCVCamera:
    """Read a ``cloudscope.camera_model/1`` JSON file into an OpenCVCamera."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_cloudscope_model(d)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    intr = d["intrinsics"]
    return OpenCVCamera(model=d["model"], width=int(d["image"]["width"]), height=int(d["image"]["height"]),
                        fx=float(intr["fx"]), fy=float(intr["fy"]), cx=float(intr["cx"]), cy=float(intr["cy"]),
                        distortion=tuple(float(x) for x in d["distortion"]), camera_id=str(d["camera"]["id"]),
                        calibration_id=str(d["calibration_id"]), rms_px=float(d["fit"]["rms_px"]))


# ----------------------------------------------------------------------------------------------- resolver
REPO_ROOT = Path(__file__).resolve().parents[2]


def nominal_fisheye(width: int, height: int, hfov_deg: float, camera_id: str, calibration_id: str) -> OpenCVCamera:
    """Equidistant fisheye with zero distortion whose horizontal field of view is `hfov_deg` (a datasheet model)."""
    f = (width / 2.0) / np.radians(hfov_deg / 2.0)
    return OpenCVCamera(model="opencv_fisheye", width=width, height=height, fx=f, fy=f,
                        cx=(width - 1) / 2.0, cy=(height - 1) / 2.0, distortion=(0.0, 0.0, 0.0, 0.0),
                        camera_id=camera_id, calibration_id=calibration_id)


@dataclass(frozen=True)
class CameraEntry:
    camera_id: str
    model: str                  # one of MODEL_KINDS
    source: str = ""            # file or pattern the model is read from
    provisional: bool = False   # True: a nominal/datasheet model, not a calibration
    note: str = ""


def load_camera_registry(path: str | Path | None = None) -> dict[str, CameraEntry]:
    import yaml

    p = Path(path) if path else REPO_ROOT / "configs" / "cameras.yaml"
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))["cameras"]
    out = {}
    for cid, d in raw.items():
        model = d.get("model", "unknown")
        if model not in MODEL_KINDS:
            raise ValueError(f"configs/cameras.yaml: {cid}: unknown model {model!r}")
        out[cid] = CameraEntry(camera_id=cid, model=model, source=d.get("source", ""),
                               provisional=bool(d.get("provisional", False)), note=d.get("note", ""))
    return out


def registry_entry(camera_id: str, registry: dict[str, CameraEntry]) -> CameraEntry:
    """Exact id first, then prefix patterns such as ``eye2sky-*``; unknown when nothing matches."""
    if camera_id in registry:
        return registry[camera_id]
    for key, entry in registry.items():
        if key.endswith("*") and camera_id.startswith(key[:-1]):
            return entry
    return CameraEntry(camera_id=camera_id, model="unknown", note="not in configs/cameras.yaml")


def camera_for(camera_id: str, width: int, height: int, calib_id: str | None = None,
               data_root: str | Path | None = None, registry: dict[str, CameraEntry] | None = None) -> CameraModel:
    """The camera model for a manifest row (camera_id, width, height, calib_id)."""
    registry = registry or load_camera_registry()
    entry = registry_entry(camera_id, registry)
    if entry.model == "unknown":
        return UnknownCamera(width=width, height=height)
    if entry.model == "ocam":
        if not calib_id or data_root is None:
            return UnknownCamera(width=width, height=height)
        from ..data.eye2sky import load_calibration

        station = camera_id.split("-", 1)[1]
        cal = load_calibration(Path(data_root) / entry.source.format(station=station, calib_id=calib_id))
        return cal.model if cal is not None else UnknownCamera(width=width, height=height)
    src = Path(entry.source)
    if not src.is_absolute():
        src = REPO_ROOT / src
    model = load_cloudscope_model(src)
    if (model.width, model.height) != (width, height):
        raise ValueError(f"{camera_id}: model is for {model.width}x{model.height}, image is {width}x{height}")
    return model
