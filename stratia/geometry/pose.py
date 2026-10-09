"""Camera pose: the rotation from the camera frame to East-North-Up (P044).

A ``Pose`` holds ``R`` (3x3) with ``enu = R @ cam``. Poses come from two sources:

* **fitted** against the Sun: Sun discs detected in the images (``sun_detect``) give camera rays, pvlib gives the
  Sun's ENU direction at the same times, and ``fit_rotation`` (Kabsch, with outlier trimming) gives the rotation
  that maps one onto the other. These are stored in ``configs/camera_poses.yaml`` with their residuals.
* **declared** by a calibration file. Eye2Sky's ``external_orientation`` is ``[roll, pitch, yaw]`` in radians with a
  convention the files do not state; ``EYE2SKY_CANDIDATES`` enumerates the plausible readings and
  ``scripts/sun_validation.py`` measures which one agrees with the Sun. The winner is ``EYE2SKY_CONVENTION``.

Rotation helpers follow the right-hand rule; ``euler_to_rotation(roll, pitch, yaw, "zyx")`` is
``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (roll applied first).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
POSES_FILE = REPO_ROOT / "configs" / "camera_poses.yaml"

# ----------------------------------------------------------------------------------------------- rotations
def rotation_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rotation_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotation_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


_AXIS = {"x": rotation_x, "y": rotation_y, "z": rotation_z}


def euler_to_rotation(roll: float, pitch: float, yaw: float, order: str = "zyx") -> np.ndarray:
    """R = R_order[0](angle of that axis) @ R_order[1](...) @ R_order[2](...); roll->x, pitch->y, yaw->z."""
    angle = {"x": roll, "y": pitch, "z": yaw}
    r = np.eye(3)
    for axis in order:
        r = r @ _AXIS[axis](angle[axis])
    return r


def rotation_to_euler_zyx(r: np.ndarray) -> tuple[float, float, float]:
    """(roll, pitch, yaw) with R = Rz(yaw) Ry(pitch) Rx(roll); pitch in (-pi/2, pi/2) branch."""
    r = np.asarray(r, dtype=float)
    pitch = float(np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0])))
    roll = float(np.arctan2(r[2, 1], r[2, 2]))
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    return roll, pitch, yaw


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between unit vectors (..., 3)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def rotation_distance_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    """Geodesic angle between two rotations."""
    c = (np.trace(np.asarray(r1).T @ np.asarray(r2)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def kabsch(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Rotation R minimising sum w |R src - dst|^2 for unit vectors src, dst (n, 3)."""
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=float)
    h = (src * w[:, None]).T @ dst
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def fit_rotation(cam_rays: np.ndarray, enu_vectors: np.ndarray, trim_factor: float = 3.0, floor_deg: float = 0.5,
                 iterations: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kabsch with outlier trimming: after each fit, observations further than max(trim_factor * median, floor)
    from the fit are dropped. Returns (R, inlier mask, errors in degrees for all observations)."""
    cam, enu = np.asarray(cam_rays, dtype=float), np.asarray(enu_vectors, dtype=float)
    if len(cam) < 3:
        raise ValueError("at least three observations are needed to fit a rotation")
    inlier = np.ones(len(cam), dtype=bool)
    r = kabsch(cam, enu)
    for _ in range(iterations):
        err = angular_error_deg(cam @ r.T, enu)
        limit = max(trim_factor * float(np.median(err[inlier])), floor_deg)
        new = err <= limit
        if new.sum() < 3 or np.array_equal(new, inlier):
            break
        inlier = new
        r = kabsch(cam[inlier], enu[inlier])
    err = angular_error_deg(cam @ r.T, enu)
    return r, inlier, err


# ----------------------------------------------------------------------------------------------- Eye2Sky readings
# Base frames the declared angles may refer to, as rows = (x, y, z) axes expressed in ENU.
_BASE_FRAMES = {
    "ENU": np.eye(3),
    "NED": np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float),
    "NWU": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float),
    "ESD": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float),
}
# Camera frames the rotation may act on: the standard frame (ADR-012) or OCamCalib's (x along rows, y along
# columns, z = F(rho) pointing away from the scene). cam_std = M @ cam_ocam with M = [[0,1,0],[1,0,0],[0,0,-1]].
_CAM_FRAMES = {"std": np.eye(3), "ocam": np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)}


def _candidate_names():
    for order in ("".join(p) for p in permutations("xyz")):
        for sign in product((1, -1), repeat=3):
            signs = "".join(f"{'+' if sg > 0 else '-'}{n}" for sg, n in zip(sign, "rpy", strict=True))
            for base in _BASE_FRAMES:
                for cam in _CAM_FRAMES:
                    for direction in ("c2w", "w2c"):
                        yield f"{order}|{signs}|{base}|{cam}|{direction}"


def eye2sky_rotation(external_orientation, convention: str) -> np.ndarray:
    """R (camera std frame -> ENU) for Eye2Sky's [roll, pitch, yaw] under a named candidate convention.

    Name: ``<order>|<signs>|<base frame>|<camera frame>|<direction>``. The Euler rotation is built in the base
    frame (``base`` maps base coordinates to ENU), acts on vectors in the named camera frame, and is read as
    camera-to-world (``c2w``) or world-to-camera (``w2c``, i.e. transposed).
    """
    order, signs, base, cam, direction = convention.split("|")
    roll, pitch, yaw = (float(v) for v in external_orientation)
    sr, sp, sy = (1.0 if c == "+" else -1.0 for c in (signs[0], signs[2], signs[4]))
    r_base = euler_to_rotation(sr * roll, sp * pitch, sy * yaw, order)
    if direction == "w2c":
        r_base = r_base.T
    base_to_enu = _BASE_FRAMES[base].T            # columns = base axes in ENU
    cam_to_named = _CAM_FRAMES[cam].T             # std -> named camera frame (inverse of named -> std)
    return base_to_enu @ r_base @ cam_to_named


EYE2SKY_CANDIDATES = tuple(_candidate_names())

def eye2sky_convention(records_meta: dict | None = None) -> str | None:
    """The reading that agreed with the Sun on the stations with images (P044): stored in configs/camera_poses.yaml
    (``meta.eye2sky_convention``) by scripts/sun_validation.py; None until measured."""
    meta = load_pose_meta() if records_meta is None else records_meta
    return meta.get("eye2sky_convention")


# ----------------------------------------------------------------------------------------------- Pose records
@dataclass(frozen=True)
class Pose:
    R: np.ndarray                       # camera (std frame) -> ENU
    source: str = "declared"            # "fitted" | "declared" | "identity"
    calib_id: str = ""

    def cam_to_enu(self, rays: np.ndarray) -> np.ndarray:
        return np.asarray(rays, dtype=float) @ self.R.T

    def enu_to_cam(self, vectors: np.ndarray) -> np.ndarray:
        return np.asarray(vectors, dtype=float) @ self.R

    @property
    def optical_axis_elevation_deg(self) -> float:
        return float(np.degrees(np.arcsin(np.clip(self.R[2, 2], -1, 1))))


def _load_pose_file(path: str | Path | None = None) -> dict:
    import yaml

    p = Path(path) if path else POSES_FILE
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_pose_records(path: str | Path | None = None) -> dict[str, dict]:
    """configs/camera_poses.yaml -> {calib_id: record}; empty when the file does not exist."""
    return dict(_load_pose_file(path).get("poses", {}))


def load_pose_meta(path: str | Path | None = None) -> dict:
    return dict(_load_pose_file(path).get("meta", {}))


def save_pose_records(records: dict[str, dict], meta: dict, path: str | Path | None = None) -> None:
    import yaml

    p = Path(path) if path else POSES_FILE
    doc = {"meta": meta, "poses": records}
    p.write_text("# Camera poses fitted against the Sun (P044, scripts/sun_validation.py). Do not edit by hand.\n"
                 + yaml.safe_dump(doc, sort_keys=False, width=120), encoding="utf-8", newline="\n")


def pose_for(calib_id: str, external_orientation=None, records: dict[str, dict] | None = None) -> Pose | None:
    """Fitted pose when one with status 'ok' exists for calib_id; else the declared Eye2Sky orientation under the
    established convention; None when neither is available."""
    records = load_pose_records() if records is None else records
    rec = records.get(calib_id)
    if rec and rec.get("status") == "ok":
        return Pose(R=np.array(rec["R"], dtype=float), source="fitted", calib_id=calib_id)
    convention = eye2sky_convention()
    if external_orientation is not None and convention:
        return Pose(R=eye2sky_rotation(external_orientation, convention), source="declared", calib_id=calib_id)
    return None
