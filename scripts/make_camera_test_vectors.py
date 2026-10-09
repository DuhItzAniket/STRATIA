"""Shared camera test vectors (P043, CloudScope ADR-012).

    python scripts/make_camera_test_vectors.py

Writes tests/vectors/camera_test_vectors.json: pixel -> ray and ray -> pixel samples for one real Eye2Sky OCamCalib
calibration (OLDLR, valid from 2022-04-22) and for the synthetic OpenCV fisheye and pinhole models CloudScope's own
calibration tests use. CloudScope copies the file and must reproduce every number; `tests/test_cameras.py` recomputes
them here. P044/P045 append the Sun-direction cases (pixel -> azimuth/elevation at a reference time).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.geometry.cameras import OpenCVCamera  # noqa: E402
from stratia.geometry.ocam import OcamModel  # noqa: E402

OUT = Path("tests/vectors/camera_test_vectors.json")

CAMERAS = {
    "eye2sky_OLDLR_20220422_ocam": OcamModel(ss=(-642.6690459406254, 0.0, 0.0002441927468927244, 4.838462720720562e-07),
                                             xc=1028.60, yc=1072.59, width=2112, height=2048),
    "synthetic_opencv_fisheye_640x480": OpenCVCamera("opencv_fisheye", 640, 480, 300.0, 302.0, 325.0, 236.0,
                                                     (-0.05, 0.01, -0.002, 0.0005)),
    "synthetic_opencv_pinhole_640x480": OpenCVCamera("opencv_pinhole", 640, 480, 520.0, 518.0, 322.0, 241.0,
                                                     (-0.20, 0.05, 0.001, -0.0005, 0.0)),
}


def params(cam) -> dict:
    if isinstance(cam, OcamModel):
        return {"kind": "ocam", "ss": list(cam.ss), "xc": cam.xc, "yc": cam.yc, "c": cam.c, "d": cam.d, "e": cam.e,
                "width": cam.width, "height": cam.height,
                "note": "OCamCalib: xc is the centre ROW, yc the centre COLUMN (0-based); F(rho) = sum ss[i] rho^i"}
    return {"kind": cam.model, "width": cam.width, "height": cam.height, "fx": cam.fx, "fy": cam.fy, "cx": cam.cx,
            "cy": cam.cy, "distortion": list(cam.distortion)}


def main() -> None:
    rng = np.random.default_rng(43)
    cases = []
    for name, cam in CAMERAS.items():
        if isinstance(cam, OcamModel):
            r = 900 * np.sqrt(rng.uniform(0, 1, 12))
            a = rng.uniform(0, 2 * np.pi, 12)
            u, v = cam.yc + r * np.cos(a), cam.xc + r * np.sin(a)
            u, v = np.concatenate([[cam.yc], u]), np.concatenate([[cam.xc], v])
        else:
            u = np.concatenate([[cam.cx], rng.uniform(0.05 * cam.width, 0.95 * cam.width, 12)])
            v = np.concatenate([[cam.cy], rng.uniform(0.05 * cam.height, 0.95 * cam.height, 12)])
        u, v = np.round(u, 3), np.round(v, 3)
        rays = cam.pixel_to_ray(u, v)
        uu, vv = cam.ray_to_pixel(rays)
        assert np.hypot(uu - u, vv - v).max() < 1e-6
        samples = [{"u": float(a), "v": float(b), "ray": [float(x) for x in r]} for a, b, r in zip(u, v, rays, strict=True)]
        cases.append({"name": name, "camera": params(cam), "samples": samples})
    doc = {
        "schema": "stratia.camera_test_vectors/1",
        "conventions": {
            "pixel": "(u, v) from the top-left pixel centre, u right (column), v down (row); 0-based",
            "camera_frame": "+x toward +u, +y toward +v, +z along the optical axis into the scene; rays are unit vectors "
                            "(CloudScope ADR-012, STRATIA docs/contract.md)",
            "tolerance": "ray components within 1e-9; pixel round trip within 1e-6 px",
        },
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}: {len(cases)} cameras, {sum(len(c['samples']) for c in cases)} samples")


if __name__ == "__main__":
    main()
