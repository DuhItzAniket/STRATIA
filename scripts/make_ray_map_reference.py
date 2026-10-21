"""Cross-repository ray-map vectors (P096 / CloudScope P067).

    python scripts/make_ray_map_reference.py [--out tests/fixtures/stratia_dummy/ray_map_reference.json]

Computes the contract's ray map with STRATIA's reference implementation for the nominal B0268 camera
(configs/cameras/b0268_nominal.camera.json) under two poses given in CloudScope's convention (azimuth, elevation,
roll; the roll about the optical axis first, then the pointing; CloudScope `camera_orientation` +
`SkyProjection`), so that CloudScope's `ai::ray_map` can be checked against these numbers to 1e-5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from stratia import contract as C  # noqa: E402
from stratia.geometry.cameras import camera_for  # noqa: E402
from stratia.geometry.pose import Pose  # noqa: E402
from stratia.geometry.raymap import meta_vector, ray_map  # noqa: E402

CASES = [
    {"name": "zenith_rolled", "azimuth_deg": 30.0, "elevation_deg": 90.0, "roll_deg": 10.0, "sun_azimuth_deg": 135.0,
     "sun_apparent_elevation_deg": 50.0},
    {"name": "tilted_to_horizon", "azimuth_deg": 210.0, "elevation_deg": 35.0, "roll_deg": -5.0, "sun_azimuth_deg": 80.0,
     "sun_apparent_elevation_deg": 20.0},
]


def cloudscope_rotation(azimuth_deg: float, elevation_deg: float, roll_deg: float) -> np.ndarray:
    """camera -> ENU as CloudScope composes it: columns are the camera axes (+x right, +y down, +z optical axis)
    expressed in ENU for a level camera looking along (azimuth, elevation), applied after a roll about +z."""
    az, el, roll = np.radians([azimuth_deg, elevation_deg, roll_deg])
    forward = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    right = np.array([np.cos(az), -np.sin(az), 0.0])
    down = np.array([np.sin(el) * np.sin(az), np.sin(el) * np.cos(az), -np.cos(el)])
    level = np.stack([right, down, forward], axis=1)
    c, s = np.cos(roll), np.sin(roll)
    about_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return level @ about_z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures" / "stratia_dummy" / "ray_map_reference.json")
    a = ap.parse_args()
    camera = camera_for("b0268", 4656, 3496)
    doc = {"description": "Contract ray maps of the nominal B0268 camera (configs/cameras/b0268_nominal.camera.json) "
                          "from STRATIA's reference implementation for poses in CloudScope's convention; CloudScope "
                          "ai::ray_map must match to 1e-5 (P067)",
           "camera": "b0268_nominal.camera.json", "size": C.DEFAULT_INPUT_SIZE, "patch": C.PATCH_SIZE, "cases": []}
    for case in CASES:
        r = cloudscope_rotation(case["azimuth_deg"], case["elevation_deg"], case["roll_deg"])
        zen = 90.0 - case["sun_apparent_elevation_deg"]
        rm = ray_map(camera, Pose(R=r, source="declared"), case["sun_azimuth_deg"], zen)
        doc["cases"].append({**case, "valid_count": int(rm[3].sum()), "meta": [round(float(v), 7) for v in meta_vector(zen)],
                             "ray_map": [[round(float(v), 6) for v in channel.ravel()] for channel in rm]})
    a.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{a.out} ({a.out.stat().st_size} bytes); valid patches: "
          + ", ".join(f"{c['name']} {c['valid_count']}" for c in doc["cases"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
