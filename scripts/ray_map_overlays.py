"""Ray-map overlays and statistics (P045).

    python scripts/ray_map_overlays.py [--stations AURIC BARSE OLDLR WESTE] [--time 2022-04-01T12:00:00]

For every station with images and a pose (configs/camera_poses.yaml, P044): take the frame nearest `--time`, build
the contract's ray map (512 px input, 16 px patches) with the fitted pose, the camera mask and the near-horizon
floor, and draw: zenith-angle contours, the azimuth-from-Sun field as arrows, the valid flag, and the Sun's own
projected position. Writes docs/data/figures/raymap_<station>.jpg and docs/data/ray_maps.md (valid shares,
zenith-angle coverage, Sun patch check per station).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.eye2sky import load_mask, mask_path, parse_image_name, select_calibration, station_calibrations  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.geometry import raymap as RM  # noqa: E402
from stratia.geometry.pose import load_pose_records, pose_for  # noqa: E402
from stratia.geometry.sun import azel_to_enu, sun_position  # noqa: E402

FIG_DIR = Path("docs/data/figures")


def nearest_frame(root: Path, station: str, when: datetime) -> Path | None:
    day = root / f"{when:%Y/%m/%d}"
    dirs = [p for p in day.glob(f"ASI_*_{station}") if p.is_dir()]
    if not dirs:
        return None
    hour = dirs[0] / station / f"{when:%Y/%m/%d/%H}"
    frames = sorted(hour.glob("*.jpg"))
    if not frames:
        return None
    stamps = [abs((parse_image_name(f).utc.replace(tzinfo=None) - when).total_seconds()) for f in frames]
    return frames[int(np.argmin(stamps))]


def overlay(station: str, frame: Path, cal, pose, mask, min_el: float) -> tuple[Path, dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = parse_image_name(frame)
    sp = sun_position([name.utc], cal.latitude, cal.longitude, cal.altitude)
    az, zen = float(sp.azimuth.iloc[0]), float(sp.zenith.iloc[0])
    rm = RM.ray_map(cal.model, pose, az, zen, mask, min_elevation_deg=min_el)
    img = cv2.cvtColor(cv2.imdecode(np.fromfile(str(frame), dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    small[~RM.resized_mask(mask, 512)] = 0
    cu, cv_ = RM.patch_centres(512, 512)
    zenith = RM.zenith_angle_deg(rm)
    az_rel = RM.azimuth_from_sun_deg(rm)
    # the Sun's own pixel in the 512 image
    su, sv = cal.model.ray_to_pixel(pose.enu_to_cam(azel_to_enu(az, 90 - zen)))
    a = RM.full_frame_affine(cal.model.width, cal.model.height)
    inv = np.linalg.inv(a)
    sx, sy = inv[0, 0] * su + inv[0, 2], inv[1, 1] * sv + inv[1, 2]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    ax = axes[0]
    ax.imshow(small)
    cs = ax.contour(cu, cv_, np.nan_to_num(zenith, nan=180.0), levels=[15, 30, 45, 60, 75, 85], colors="cyan", linewidths=0.8)
    ax.clabel(cs, fmt="%d°", fontsize=7)
    ax.plot(sx, sy, "o", mfc="none", mec="yellow", ms=14, mew=2, label="Sun (projected)")
    ax.set_title(f"{station} {name.utc:%Y-%m-%d %H:%M} UTC: zenith angle")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax = axes[1]
    ax.imshow(small)
    valid = rm[3] > 0
    # arrows: the horizontal direction toward the Sun, as seen in image coordinates, every second patch
    step = 2
    # image-plane direction of "toward the Sun" at each patch: numerical derivative of the projection along x_sun
    enu_x = np.array([np.sin(np.radians(az)), np.cos(np.radians(az)), 0.0])
    rays = pose.cam_to_enu(cal.model.pixel_to_ray(*RM.apply_affine(a, cu, cv_)))
    nudged = rays + 0.02 * enu_x
    nudged /= np.linalg.norm(nudged, axis=-1, keepdims=True)
    pu, pv = cal.model.ray_to_pixel(pose.enu_to_cam(nudged))
    hu = (inv[0, 0] * pu + inv[0, 2]) - cu
    hv = (inv[1, 1] * pv + inv[1, 2]) - cv_
    norm = np.hypot(hu, hv)
    ok = valid & np.isfinite(norm) & (norm > 0)
    hu, hv = np.where(ok, hu / np.where(norm > 0, norm, 1), 0), np.where(ok, hv / np.where(norm > 0, norm, 1), 0)
    sel = np.zeros_like(ok)
    sel[::step, ::step] = True
    sel &= ok
    ax.quiver(cu[sel], cv_[sel], hu[sel], hv[sel], np.abs(az_rel[sel]), cmap="coolwarm", scale=40, width=0.004, clim=(0, 180))
    ax.plot(sx, sy, "o", mfc="none", mec="yellow", ms=14, mew=2)
    ax.set_title("direction toward the Sun's azimuth (colour: |azimuth from Sun|)")
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax = axes[2]
    ax.imshow(valid, cmap="gray", interpolation="nearest", extent=(0, 512, 512, 0))
    ax.set_title(f"valid flag: {valid.mean():.0%} of patches (mask, horizon >= {min_el:.0f}°)")
    fig.tight_layout()
    out = FIG_DIR / f"raymap_{station}.jpg"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    # Sun-patch check: the patch containing the projected Sun should look at the Sun
    i, j = int(sy // 16), int(sx // 16)
    sun_ok = 0 <= i < 32 and 0 <= j < 32 and valid[i, j]
    stats = {"frame": str(frame.name), "utc": f"{name.utc:%Y-%m-%d %H:%M:%S}", "sun_azimuth": round(az, 2),
             "sun_zenith": round(zen, 2),
             "valid_share": round(float(valid.mean()), 3), "zenith_min": round(float(np.nanmin(zenith)), 2),
             "zenith_max": round(float(np.nanmax(zenith)), 2), "pose_source": pose.source,
             "sun_patch_distance_deg": round(float(RM.sun_distance_deg(rm, zen)[i, j]), 2) if sun_ok else None,
             "sun_patch_azimuth_deg": round(float(az_rel[i, j]), 2) if sun_ok else None}
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--time", default="2022-04-01T12:00:00")
    a = ap.parse_args()
    t0 = time.perf_counter()
    when = datetime.fromisoformat(a.time)
    root = Path(load_paths()["data_root"]) / "Eye2Sky"
    meta = root / "asi_meta"
    records = load_pose_records()
    stations = a.stations or sorted({p.name.rsplit("_", 1)[1] for p in root.glob("2022/*/*/ASI_*") if p.is_dir()})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows, figs = [], []
    for station in stations:
        frame = nearest_frame(root, station, when)
        if frame is None:
            print(f"{station}: no frame near {when}")
            continue
        cals = station_calibrations(meta, station)
        cal = select_calibration(cals, parse_image_name(frame).utc)
        pose = pose_for(cal.file, cal.external_orientation, records)
        if pose is None:
            print(f"{station}: no pose for {cal.file}")
            continue
        mask = load_mask(mask_path(meta, cal))
        min_el = RM.min_elevation_for(f"eye2sky-{station}")
        fig, stats = overlay(station, frame, cal, pose, mask, min_el)
        stats["station"], stats["calib_id"] = station, cal.file
        rows.append(stats)
        figs.append(fig)
        print(f"{station}: {stats}", flush=True)
    lines = ["# Ray maps (P045)", "",
             f"Generated by `scripts/ray_map_overlays.py` (frames nearest {when:%Y-%m-%d %H:%M} UTC).", "",
             "Ray map per docs/contract.md: 512 x 512 input, 16 px patches, channels = unit direction in the Sun-aligned frame "
             "(x toward the Sun's azimuth, z up) + valid flag. Validity = calibrated camera, >= 50 % of the patch footprint "
             "inside the camera mask, elevation >= the near-horizon floor (`configs/near_horizon.csv`). Poses from "
             "`configs/camera_poses.yaml` (P044).", "",
             "| Station | Calibration | Frame (UTC) | Pose | Sun az / zenith | Valid patches | Zenith range | "
             "Sun patch: distance, azimuth |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sun = (f"{r['sun_patch_distance_deg']} deg, {r['sun_patch_azimuth_deg']} deg" if r["sun_patch_distance_deg"] is not None
               else "Sun not in a valid patch")
        lines.append(f"| {r['station']} | {r['calib_id']} | {r['utc']} | {r['pose_source']} | "
                     f"{r['sun_azimuth']} / {r['sun_zenith']} | "
                     f"{r['valid_share']:.1%} | {r['zenith_min']}-{r['zenith_max']} deg | {sun} |")
    lines += ["", "The Sun patch columns are the check that the Sun-aligned frame is oriented correctly: the patch holding the "
              "projected Sun must look within a few degrees of the Sun (a patch is ~6 deg wide at the image centre) with "
              "azimuth-from-Sun near 0.", "", "## Figures", ""] + [f"- `{f.as_posix()}`" for f in figs]
    Path("docs/data/ray_maps.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: docs/data/ray_maps.md, {len(figs)} figures")


if __name__ == "__main__":
    main()
