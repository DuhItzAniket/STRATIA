"""B0268 Sun-track pose fit: where the Sun actually sits in the logged frames against where the declared zenith
pointing puts it (follow-up to P083/P084 and the CloudScope mask work of 2026-11-05).

    python scripts/b0268_sun_track.py                      # the three adaptation days
    python scripts/b0268_sun_track.py --days 2026-10-18    # one day

The camera's pointing is a declared zenith with no measured rotation about it, so the ray maps STRATIA built for the
B0268 (P083, P084, P090) carry the right zenith angle per pixel but an arbitrary azimuth. This script detects the
Sun in every logged frame (the brightest saturated plateau near the ring of the Sun's zenith angle, as the CloudScope
glare rule does), turns the detections into camera-frame rays through the nominal fisheye model of P042, and fits
the camera-to-ENU rotation against the Sun vectors of the sidecars (Kabsch with trimming, as P044 did for Eye2Sky).
The locked day 2026-10-20 is excluded by default: its frames stay untouched by everything but P090.

Outputs: data/b0268_sun_track.parquet (one row per frame), docs/data/b0268_sun_track.md, and a pose record
`BLR01_b0268_nominal` in configs/camera_poses.yaml (status "ok" under the P044 rule, else "insufficient"; no
existing code looks that record up, so nothing changes until an adaptation asks for it).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stratia.geometry.cameras import camera_for  # noqa: E402
from stratia.geometry.pose import (  # noqa: E402
    angular_error_deg,
    fit_rotation,
    load_pose_meta,
    load_pose_records,
    rotation_to_euler_zyx,
    save_pose_records,
)
from stratia.geometry.sun import azel_to_enu  # noqa: E402

DATA = ROOT.parent.parent / "DATA" / "B0268" / "BLR01"
DAYS = ("2026-10-18", "2026-10-19", "2026-10-21")
REDUCE = 4  # detection on the quarter-resolution decode
BLUR = 31  # plateau window at the reduced size (about 124 px on the sensor)
GATE_DEG = 25.0  # the plateau must sit this close to the ring of the Sun's zenith angle under the declared pointing
SATURATED = 0.90
MIN_ELEVATION = 10.0
STATUS_RULE = {"min_inliers": 15, "min_span_hours": 3.0, "max_median_deg": 1.0, "max_p90_deg": 2.0}


def find_sun(luminance: np.ndarray, zenith_deg: np.ndarray, sun_zenith_deg: float, gate_deg: float = GATE_DEG,
             blur: int = BLUR, saturated: float = SATURATED) -> tuple[int, int, float] | None:
    """(column, row, plateau luminance) of the brightest `blur` x `blur` plateau among the pixels whose zenith angle
    is within `gate_deg` of the Sun's; None when no plateau there is saturated."""
    import cv2

    blurred = cv2.blur(luminance.astype(np.float32), (blur, blur), borderType=cv2.BORDER_REFLECT)
    band = np.isfinite(zenith_deg) & (np.abs(zenith_deg - sun_zenith_deg) <= gate_deg)
    if not band.any():
        return None
    cand = np.where(band, blurred, -1.0)
    row, col = np.unravel_index(int(cand.argmax()), cand.shape)
    value = float(cand[row, col])
    if value < saturated:
        return None
    return int(col), int(row), value


def reduced_rays(camera, width: int, height: int, reduce: int) -> np.ndarray:
    """Camera-frame unit rays at the centres of the reduce x reduce blocks of a (height, width) sensor."""
    cols = np.arange(width // reduce) * reduce + (reduce - 1) / 2.0
    rows = np.arange(height // reduce) * reduce + (reduce - 1) / 2.0
    u, v = np.meshgrid(cols, rows)
    return camera.pixel_to_ray(u, v)


def detections(days: tuple[str, ...], data: Path = DATA) -> pd.DataFrame:
    import cv2

    rows = []
    rays_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for day in days:
        for jpg in sorted((data / day).glob("*.jpg")):
            side = json.loads(jpg.with_suffix(".json").read_text(encoding="utf-8"))
            sun = side["sun"]
            az, el = float(sun["azimuth_deg"]), float(sun["apparent_elevation_deg"])
            rec = {"day": day, "file": jpg.name, "utc": side["capture"]["utc"], "sun_azimuth_deg": az,
                   "sun_elevation_deg": el, "detected": False}
            if el < MIN_ELEVATION:
                rows.append(rec)
                continue
            bgr = cv2.imread(str(jpg), cv2.IMREAD_REDUCED_COLOR_4)
            full_w, full_h = int(side["camera"]["requested"]["width"]), int(side["camera"]["requested"]["height"])
            key = (full_w, full_h)
            if key not in rays_cache:
                camera = camera_for("b0268", full_w, full_h)
                rays = reduced_rays(camera, full_w, full_h, REDUCE)[: bgr.shape[0], : bgr.shape[1]]
                zenith = np.degrees(np.arccos(np.clip(rays[..., 2], -1.0, 1.0)))
                rays_cache[key] = (rays, zenith)
            rays, zenith = rays_cache[key]
            lum = bgr.max(axis=2).astype(np.float32) / 255.0
            found = find_sun(lum, zenith, 90.0 - el)
            if found is None:
                rows.append(rec)
                continue
            col, row, value = found
            ray = rays[row, col]
            rec.update({"detected": True, "u": col * REDUCE + (REDUCE - 1) / 2.0, "v": row * REDUCE + (REDUCE - 1) / 2.0,
                        "plateau": value, "ray_x": float(ray[0]), "ray_y": float(ray[1]), "ray_z": float(ray[2]),
                        "cam_zenith_deg": float(zenith[row, col]),
                        "cam_azimuth_deg": float(np.degrees(np.arctan2(ray[1], ray[0])) % 360.0)})
            rows.append(rec)
    return pd.DataFrame(rows)


def fit(det: pd.DataFrame) -> dict:
    d = det[det.detected].copy()
    cam = d[["ray_x", "ray_y", "ray_z"]].to_numpy(dtype=float)
    enu = azel_to_enu(d.sun_azimuth_deg.to_numpy(dtype=float), d.sun_elevation_deg.to_numpy(dtype=float))
    out = {"n_frames": int(len(det)), "n_detections": int(len(d))}
    if len(d) < 3:
        out["status"] = "insufficient"
        return out
    r, inlier, err = fit_rotation(cam, enu)
    declared = angular_error_deg(cam, enu)  # the declared pointing is the identity rotation
    roll, pitch, yaw = (float(np.degrees(a)) for a in rotation_to_euler_zyx(r))
    yaw_simple = (d.sun_azimuth_deg.to_numpy() - d.cam_azimuth_deg.to_numpy()) % 360.0
    times = pd.to_datetime(d.utc, utc=True)
    span_h = float((times.max() - times.min()).total_seconds() / 3600.0)
    ok = (int(inlier.sum()) >= STATUS_RULE["min_inliers"] and span_h >= STATUS_RULE["min_span_hours"]
          and float(np.median(err[inlier])) <= STATUS_RULE["max_median_deg"]
          and float(np.percentile(err[inlier], 90)) <= STATUS_RULE["max_p90_deg"])
    out.update({
        "status": "ok" if ok else "insufficient",
        "R": r.tolist(), "roll_deg": roll, "pitch_deg": pitch, "yaw_deg": yaw,
        "yaw_simple_median_deg": float(np.median(yaw_simple)), "yaw_simple_iqr_deg": float(np.subtract(*np.percentile(yaw_simple, [75, 25]))),  # noqa: E501
        "n_inliers": int(inlier.sum()), "span_hours": span_h,
        "declared_median_deg": float(np.median(declared)), "declared_p90_deg": float(np.percentile(declared, 90)),
        "fitted_median_deg": float(np.median(err[inlier])), "fitted_p90_deg": float(np.percentile(err[inlier], 90)),
        "optical_axis_elevation_deg": float(np.degrees(np.arcsin(np.clip(r[2, 2], -1, 1)))),
    })
    det.loc[d.index, "fitted_err_deg"] = err
    det.loc[d.index, "declared_err_deg"] = declared
    det.loc[d.index, "inlier"] = inlier
    return out


def focal_scan(det: pd.DataFrame, scales=(0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5)) -> list[dict]:
    """The fit repeated with the nominal focal length scaled: does the residual come from the intrinsics? (With the
    plateau detector the focal and the axis tilt are degenerate, so this is a diagnostic, not a calibration.)"""
    import dataclasses

    d = det[det.detected]
    if len(d) < 3:
        return []
    cam0 = camera_for("b0268", 4656, 3496)
    enu = azel_to_enu(d.sun_azimuth_deg.to_numpy(dtype=float), d.sun_elevation_deg.to_numpy(dtype=float))
    out = []
    for s in scales:
        cam = dataclasses.replace(cam0, fx=cam0.fx * s, fy=cam0.fy * s)
        rays = cam.pixel_to_ray(d.u.to_numpy(dtype=float), d.v.to_numpy(dtype=float))
        ok = np.isfinite(rays).all(axis=1)
        if ok.sum() < 3:
            continue
        r, inl, err = fit_rotation(rays[ok], enu[ok])
        _, _, yaw = (float(np.degrees(a)) for a in rotation_to_euler_zyx(r))
        out.append({"scale": s, "n": int(ok.sum()), "median_deg": float(np.median(err[inl])), "p90_deg": float(np.percentile(err[inl], 90)),  # noqa: E501
                    "yaw_deg": yaw, "axis_elevation_deg": float(np.degrees(np.arcsin(np.clip(r[2, 2], -1, 1))))})
    return out


def write_sheet(det: pd.DataFrame, result: dict, data: Path, path: Path) -> None:
    """Twelve frames (four per day): the detected plateau (red), the Sun under the declared identity pose (blue) and
    under the fitted pose (green)."""
    import cv2

    if "R" not in result:
        return
    r_fit = np.array(result["R"], dtype=float)
    cam = camera_for("b0268", 4656, 3496)
    tiles = []
    for _, part in det[det.detected].groupby("day"):
        for i in np.linspace(0, len(part) - 1, 4).astype(int):
            row = part.iloc[i]
            img = cv2.imread(str(data / row.day / row.file), cv2.IMREAD_REDUCED_COLOR_8)
            s = 1 / 8.0
            enu = azel_to_enu(np.array([row.sun_azimuth_deg]), np.array([row.sun_elevation_deg]))[0]
            for pose_r, colour in ((np.eye(3), (255, 128, 0)), (r_fit, (0, 220, 0))):
                u, v = cam.ray_to_pixel((pose_r.T @ enu)[None, :])
                if np.isfinite(u[0]) and np.isfinite(v[0]):
                    cv2.circle(img, (int(u[0] * s), int(v[0] * s)), 14, colour, 2)
            cv2.circle(img, (int(row.u * s), int(row.v * s)), 9, (0, 0, 255), 2)
            cv2.putText(img, f"{row.file[6:21]} el {row.sun_elevation_deg:.0f} err {row.fitted_err_deg:.1f}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.resize(img, (388, 291)))
    rows = [np.concatenate(tiles[i : i + 4], axis=1) for i in range(0, len(tiles), 4)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 80])


def write_doc(det: pd.DataFrame, result: dict, per_day: dict[str, dict], scan: list[dict], path: Path) -> None:
    f = lambda v, n=2: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"  # noqa: E731
    lines = ["# B0268 Sun track and pose (2026-11-05)", "",
             "The Sun found in every logged frame (brightest saturated plateau within 25° of the ring of the Sun's zenith "
             "angle under the declared zenith pointing; quarter-resolution decode, 31-px window) as a camera-frame ray "
             "through the nominal fisheye model of P042, fitted against the sidecars' Sun vectors with P044's trimmed "
             "Kabsch. The declared pointing is the identity rotation; its residual is what the P083/P084/P090 ray maps "
             "carried.", "",
             "| Days | Frames | Detections | Inliers | Span (h) | Declared median / p90 (°) | Fitted median / p90 (°) | Roll / pitch / yaw (°) | Yaw by azimuth difference, median (IQR) | Status |",  # noqa: E501
             "|---|---|---|---|---|---|---|---|---|---|"]
    for label, res in [("all", result)] + list(per_day.items()):
        if res.get("status") == "insufficient" and "R" not in res:
            lines.append(f"| {label} | {res['n_frames']} | {res['n_detections']} | – | – | – | – | – | – | insufficient |")
            continue
        identifiable = res['span_hours'] >= STATUS_RULE['min_span_hours']
        euler = (f"{f(res['roll_deg'], 1)} / {f(res['pitch_deg'], 1)} / {f(res['yaw_deg'], 1)}" if identifiable
                 else 'not identifiable (arc < 3 h)')
        lines.append(
            f"| {label} | {res['n_frames']} | {res['n_detections']} | {res['n_inliers']} | {f(res['span_hours'], 1)} | "
            f"{f(res['declared_median_deg'])} / {f(res['declared_p90_deg'])} | {f(res['fitted_median_deg'])} / {f(res['fitted_p90_deg'])} | "  # noqa: E501
            f"{euler} | {f(res['yaw_simple_median_deg'], 1)} ({f(res['yaw_simple_iqr_deg'], 1)}) | {res['status']} |")  # noqa: E501
    lines += ["", f"Status rule (P044): inliers ≥ {STATUS_RULE['min_inliers']}, span ≥ {STATUS_RULE['min_span_hours']} h, median ≤ "  # noqa: E501
              f"{STATUS_RULE['max_median_deg']}°, p90 ≤ {STATUS_RULE['max_p90_deg']}° on the inliers. The nominal model is the "
              "datasheet fisheye (f = 2540.7 px, no measured distortion), and the detector takes the centre of a saturated "
              "plateau that can span 40° of sky, so a residual well above the Eye2Sky 0.1° says 'intrinsics and detector' "
              "as much as 'pose'; the yaw is the robust part (it only needs the Sun's azimuth in the picture).", ""]
    if scan:
        lines += ["## Focal scale", "", "The fit with the nominal focal length scaled. The residual falls slowly as the focal grows "  # noqa: E501
                  "while the inferred axis tilt grows with it: focal and tilt are degenerate under this detector, so the "
                  "scan is a diagnostic and the nominal focal is kept; a checkerboard calibration (P031) is what resolves it.", "",  # noqa: E501
                  "| Focal × | n | Median (°) | p90 (°) | Yaw (°) | Axis elevation (°) |", "|---|---|---|---|---|---|"]
        lines += [f"| {s['scale']:.1f} | {s['n']} | {f(s['median_deg'])} | {f(s['p90_deg'])} | {f(s['yaw_deg'], 1)} | {f(s['axis_elevation_deg'], 1)} |"  # noqa: E501
                  for s in scan]
        lines.append("")
    lines += ["![detections against the declared and fitted poses](figures/b0268_sun_track.jpg)", "",
              "Twelve frames, four per day: red = detected plateau, blue = the Sun under the declared identity pose (off the "
              "glare in every frame), green = under the fitted pose.", "",
              "Rows in `data/b0268_sun_track.parquet`; the pose record `BLR01_b0268_nominal` in `configs/camera_poses.yaml` "
              "(nothing reads it yet: the P083/P084 ray maps and the locked-day evaluation used the declared identity "
              "pose and stay as reported)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", nargs="+", default=list(DAYS))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--no-pose-record", action="store_true", help="do not write configs/camera_poses.yaml")
    args = ap.parse_args()
    if "2026-10-20" in args.days:
        print("refusing the locked day 2026-10-20: its frames are read by P090 only", file=sys.stderr)
        return 2
    det = detections(tuple(args.days), Path(args.data))
    det["fitted_err_deg"], det["declared_err_deg"], det["inlier"] = np.nan, np.nan, False
    result = fit(det)
    per_day = {}
    for day, part in det.groupby("day"):
        part = part.copy()
        per_day[day] = fit(part)
    (ROOT / "data").mkdir(exist_ok=True)
    det.to_parquet(ROOT / "data" / "b0268_sun_track.parquet", index=False)
    (ROOT / "docs" / "data").mkdir(parents=True, exist_ok=True)
    scan = focal_scan(det)
    write_sheet(det, result, Path(args.data), ROOT / "docs" / "data" / "figures" / "b0268_sun_track.jpg")
    write_doc(det, result, per_day, scan, ROOT / "docs" / "data" / "b0268_sun_track.md")
    if not args.no_pose_record and "R" in result:
        records = load_pose_records()
        meta = load_pose_meta()
        records["BLR01_b0268_nominal"] = {
            "station": "BLR01", "camera": "b0268", "model": "nominal datasheet fisheye (P042, provisional)",
            "status": result["status"], "R": result["R"],
            "roll_deg": result["roll_deg"], "pitch_deg": result["pitch_deg"], "yaw_deg": result["yaw_deg"],
            "n_detections": result["n_detections"], "n_inliers": result["n_inliers"], "span_hours": result["span_hours"],
            "median_err_deg": result["fitted_median_deg"], "p90_err_deg": result["fitted_p90_deg"],
            "declared_median_err_deg": result["declared_median_deg"], "days": list(args.days),
            "generated_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
            "source": "scripts/b0268_sun_track.py",
        }
        save_pose_records(records, meta)
    print(json.dumps({k: v for k, v in result.items() if k != "R"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
