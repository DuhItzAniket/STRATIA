"""Consumer-view synthesis (P049): re-project an all-sky frame into a pointed 105-degree view.

An Eye2Sky fisheye frame with a fitted pose (P044) sees the whole sky; a B0268-like virtual camera (P043 nominal
model, scaled) pointed at a random azimuth and elevation sees a part of it. For every virtual pixel the viewing
direction goes camera -> ENU -> source camera -> source pixel, and ``cv2.remap`` samples the frame. The ray map
of the view comes from the virtual camera and pose through the ordinary P045 generator, so the synthetic sample
carries exactly the geometry the model will see from the real B0268.

Labels are inherited from the ceilometer at the source station (``inherit_label``): the cloud-base *height* of the
zenith measurement is assigned to the view under the **flat-layer assumption** (a base height is a height, not a
slant range, and the layer is taken as horizontally uniform over the few kilometres the view covers). The join to
the P036 target table (nearest 30 s grid point within 15 s) is provisional until P047 fixes the pairing rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from .cameras import CameraModel, OpenCVCamera
from .pose import Pose
from .raymap import ray_map
from .sun import azel_to_enu

SITE_CEILOMETER = {"OLDLR": "CDLRA", "OLUOL": "CDLRA", "OLWIN": "CDLRA", "WESTE": "CDLRB"}
_RAY_CACHE: dict[CameraModel, np.ndarray] = {}


def camera_rays(cam: CameraModel) -> np.ndarray:
    """Unit rays (H, W, 3) of every pixel of `cam`, cached per camera (the virtual camera never changes)."""
    if cam not in _RAY_CACHE:
        u, v = np.meshgrid(np.arange(cam.width, dtype=float), np.arange(cam.height, dtype=float))
        _RAY_CACHE[cam] = cam.pixel_to_ray(u, v)
    return _RAY_CACHE[cam]


@dataclass(frozen=True)
class Pointing:
    azimuth_deg: float          # of the optical axis, from north clockwise
    elevation_deg: float        # of the optical axis, from the horizon
    roll_deg: float = 0.0       # rotation about the optical axis, positive = image turns clockwise on screen


def pointing_pose(p: Pointing) -> Pose:
    """Pose (camera -> ENU) of a camera whose optical axis points at (azimuth, elevation) with image-up toward the sky."""
    z = azel_to_enu(p.azimuth_deg, p.elevation_deg)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)                              # image right = horizontal, perpendicular to the axis
    n = np.linalg.norm(x)
    if n < 1e-9:                                     # looking straight up: pick north as image-up
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / n
    # ENU x (east) cross z gives "right" for a camera facing north; the general formula above is up x z, whose
    # sign makes +x point to the right when facing along z with the horizon level
    x = -x
    y = np.cross(z, x)                               # image down = toward the ground
    r = np.stack([x, y, z], axis=1)
    if p.roll_deg:
        c, s = np.cos(np.radians(p.roll_deg)), np.sin(np.radians(p.roll_deg))
        r = r @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Pose(R=r, source="synthetic")


def random_pointing(rng: np.random.Generator, elevation=(15.0, 70.0), roll=(-3.0, 3.0)) -> Pointing:
    return Pointing(float(rng.uniform(0.0, 360.0)), float(rng.uniform(*elevation)), float(rng.uniform(*roll)))


def scaled_camera(cam: OpenCVCamera, width: int) -> OpenCVCamera:
    """The same lens at a smaller image size (same field of view)."""
    s = width / cam.width
    return OpenCVCamera(cam.model, width, int(round(cam.height * s)), cam.fx * s, cam.fy * s,
                        (cam.cx + 0.5) * s - 0.5, (cam.cy + 0.5) * s - 0.5, cam.distortion,
                        camera_id=cam.camera_id, calibration_id=cam.calibration_id + f"@{width}")


@dataclass
class View:
    image: np.ndarray            # (h, w, 3) uint8
    valid: np.ndarray            # (h, w) bool: sampled from inside the source mask and above the horizon
    map_x: np.ndarray            # (h, w) float32 source column per virtual pixel (NaN outside)
    map_y: np.ndarray


def render_view(src_img: np.ndarray, src_cam: CameraModel, src_pose: Pose, virt_cam: CameraModel, virt_pose: Pose,
                src_mask: np.ndarray | None = None) -> View:
    enu = virt_pose.cam_to_enu(camera_rays(virt_cam))
    above = enu[..., 2] >= 0.0
    xs, ys = src_cam.ray_to_pixel(src_pose.enu_to_cam(enu))
    inside = (np.isfinite(xs) & np.isfinite(ys) & (xs >= 0) & (xs <= src_cam.width - 1)
              & (ys >= 0) & (ys <= src_cam.height - 1))
    valid = inside & above
    if src_mask is not None:
        ix = np.clip(np.rint(np.nan_to_num(xs)).astype(int), 0, src_mask.shape[1] - 1)
        iy = np.clip(np.rint(np.nan_to_num(ys)).astype(int), 0, src_mask.shape[0] - 1)
        valid &= src_mask[iy, ix]
    map_x = np.where(valid, xs, -1e4).astype(np.float32)
    map_y = np.where(valid, ys, -1e4).astype(np.float32)
    img = cv2.remap(src_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img[~valid] = 0
    return View(image=img, valid=valid, map_x=np.where(valid, xs, np.nan).astype(np.float32),
                map_y=np.where(valid, ys, np.nan).astype(np.float32))


def view_ray_map(virt_cam: CameraModel, virt_pose: Pose, sun_azimuth_deg: float, sun_zenith_deg: float,
                 valid: np.ndarray, out_size: int = 512, min_elevation_deg: float = 0.0) -> np.ndarray:
    """Contract ray map of the synthetic view (its own camera and pose; the rendered validity as the mask)."""
    return ray_map(virt_cam, virt_pose, sun_azimuth_deg, sun_zenith_deg, mask=valid, out_size=out_size,
                   min_elevation_deg=min_elevation_deg)


def source_footprint(virt_cam: CameraModel, virt_pose: Pose, src_cam: CameraModel, src_pose: Pose, n: int = 48):
    """The virtual image border in source pixels (for figures): (us, vs) with NaN where it leaves the source."""
    w, h = virt_cam.width - 1, virt_cam.height - 1
    t = np.linspace(0, 1, n)
    bu = np.concatenate([t * w, np.full(n, w), (1 - t) * w, np.zeros(n)])
    bv = np.concatenate([np.zeros(n), t * h, np.full(n, h), (1 - t) * h])
    enu = virt_pose.cam_to_enu(virt_cam.pixel_to_ray(bu, bv))
    us, vs = src_cam.ray_to_pixel(src_pose.enu_to_cam(enu))
    below = enu[..., 2] < 0
    return np.where(below, np.nan, us), np.where(below, np.nan, vs)


def inherit_label(targets: pd.DataFrame, utc, max_gap_s: float = 15.0) -> dict | None:
    """The ceilometer target nearest in time (P036 table rows of one site). None when no grid point is within
    `max_gap_s` or the window carried no label. Provisional pairing rule until P047."""
    if targets is None or not len(targets):
        return None
    t = pd.Timestamp(utc)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    times = targets["time"]
    i = int(np.argmin(np.abs((times - t).to_numpy().astype("timedelta64[ms]").astype(np.int64))))
    row = targets.iloc[i]
    gap = abs((row["time"] - t).total_seconds())
    if gap > max_gap_s or not bool(row["valid"]):
        return None
    return {"ceilometer": row["ceilometer"], "target_time": row["time"], "gap_s": gap, "label": row["label"],
            "etage_wmo": row["etage_wmo"], "cbh_m": float(row["cbh_m"]) if pd.notna(row["cbh_m"]) else None,
            "cbh2_m": float(row["cbh2_m"]) if pd.notna(row["cbh2_m"]) else None, "layers": float(row["layers"]),
            "confidence": float(row["confidence"]), "mixed": bool(row["mixed"]), "assumption": "flat layer"}
