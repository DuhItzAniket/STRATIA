"""Sun-based validation of Eye2Sky calibrations and camera poses (P044).

    python scripts/sun_validation.py [--stations AURIC BARSE] [--day-step 7] [--minute-step 5] [--min-elevation 5]

For every station with images on disk: sample frames (every `day-step`-th day, every `minute-step` minutes, Sun at
least `min-elevation` degrees up), detect the Sun disc, turn its pixel into a camera ray with the station's
OCamCalib model, compute the Sun's East-North-Up direction with pvlib, and
  1. fit the rotation camera -> ENU (Kabsch with outlier trimming) and report its residuals;
  2. measure which reading of the calibration file's `external_orientation` [roll, pitch, yaw] agrees with the Sun;
  3. check the manifest's Sun columns against stratia.geometry.sun.

Writes data/sun_detections.parquet (ignored), configs/camera_poses.yaml (committed: fitted poses + residuals +
the established convention), docs/data/sun_validation.md and docs/data/figures/sun_validation_<station>.jpg.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.eye2sky import load_mask, mask_path, parse_image_name, select_calibration, station_calibrations  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.geometry.pose import (  # noqa: E402
    EYE2SKY_CANDIDATES,
    angular_error_deg,
    eye2sky_rotation,
    fit_rotation,
    rotation_distance_deg,
    rotation_to_euler_zyx,
    save_pose_records,
)
from stratia.geometry.sun import azel_to_enu, sun_position  # noqa: E402
from stratia.geometry.sun_detect import detect_sun  # noqa: E402

FIG_DIR = Path("docs/data/figures")
# A pose is "ok" with at least min_inliers clean detections spanning at least min_span_hours of a day (one clear day
# gives about 20 at 5-minute sampling), median error <= 1 deg and p90 <= 2 deg.
STATUS_RULE = {"min_inliers": 15, "min_span_hours": 3.0, "max_median_deg": 1.0, "max_p90_deg": 2.0}


def station_days(root: Path, station: str) -> list[Path]:
    return sorted(p for p in root.glob(f"2022/*/*/ASI_*_{station}") if p.is_dir())


def day_frames(day_dir: Path, minute_step: int) -> list[Path]:
    out = []
    for f in sorted(day_dir.rglob("*.jpg")):
        stamp = f.name[:14]
        if len(stamp) == 14 and stamp.isdigit() and stamp[12:14] == "00" and int(stamp[10:12]) % minute_step == 0:
            out.append(f)
    return out


def read_half(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_2)
    if img is None:
        raise OSError(f"cannot decode {path}")
    return img


def detect_station(root: Path, meta: Path, station: str, day_step: int, minute_step: int, min_el: float) -> pd.DataFrame:
    days = station_days(root, station)
    chosen = days[::day_step] if days else []
    if len(chosen) <= 3:                                  # a station with few days: sample them densely
        minute_step = min(minute_step, 2)
    cals = station_calibrations(meta, station)
    masks: dict[str, np.ndarray] = {}
    rows = []
    for day in chosen:
        frames = day_frames(day, minute_step)
        if not frames:
            continue
        names = [parse_image_name(f) for f in frames]
        by_cal: dict[str, list[int]] = {}
        for i, n in enumerate(names):
            cal = select_calibration(cals, n.utc)
            if cal is not None:
                by_cal.setdefault(cal.file, []).append(i)
        for cal_file, idx in by_cal.items():
            cal = next(c for c in cals if c.file == cal_file)
            sp = sun_position([names[i].utc for i in idx], cal.latitude, cal.longitude, cal.altitude)
            if cal_file not in masks:
                m = load_mask(mask_path(meta, cal)).astype(np.uint8)
                masks[cal_file] = cv2.resize(m, (m.shape[1] // 2, m.shape[0] // 2), interpolation=cv2.INTER_NEAREST) > 0
            for k, i in enumerate(idx):
                el = float(sp.apparent_elevation.iloc[k])
                if el < min_el:
                    continue
                det = detect_sun(read_half(frames[i]), masks[cal_file])
                rows.append({"station": station, "calib_id": cal_file, "utc": names[i].utc,
                             "file": str(frames[i].relative_to(root)),
                             "detected": det.ok, "reason": det.reason, "u": 2 * det.u + 0.5, "v": 2 * det.v + 0.5,
                             "area_half": det.area, "sun_azimuth": float(sp.azimuth.iloc[k]),
                             "sun_elevation": float(sp.elevation.iloc[k]), "sun_apparent_elevation": el})
    df = pd.DataFrame(rows)
    if len(df):
        print(f"{station}: {len(chosen)} days sampled, {len(df):,} frames with the Sun >= {min_el} deg, "
              f"{int(df.detected.sum()):,} detections", flush=True)
    return df


def fit_calibration(cal, det: pd.DataFrame) -> dict:
    u, v = det.u.to_numpy(), det.v.to_numpy()
    rays = cal.model.pixel_to_ray(u, v)
    enu = azel_to_enu(det.sun_azimuth.to_numpy(), det.sun_apparent_elevation.to_numpy())
    r, inlier, err = fit_rotation(rays, enu)
    e_in = err[inlier]
    median, p90 = float(np.median(e_in)), float(np.percentile(e_in, 90))
    hod = (det.utc.dt.hour + det.utc.dt.minute / 60.0).to_numpy()[inlier]        # time of day of the inliers
    span_hours = float(hod.max() - hod.min()) if inlier.sum() else 0.0
    ok = (inlier.sum() >= STATUS_RULE["min_inliers"] and span_hours >= STATUS_RULE["min_span_hours"]
          and median <= STATUS_RULE["max_median_deg"] and p90 <= STATUS_RULE["max_p90_deg"])
    cands = {}
    for name in EYE2SKY_CANDIDATES:
        rc = eye2sky_rotation(cal.external_orientation, name)
        cands[name] = float(np.median(angular_error_deg(rays[inlier] @ rc.T, enu[inlier])))
    return {"R": r, "inlier": inlier, "err": err, "median": median, "p90": p90, "max": float(e_in.max()),
            "span_hours": span_hours, "status": "ok" if ok else "excluded", "candidates": cands}


def figure(station: str, det: pd.DataFrame, fits: dict[str, dict], cals, root: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    d = det[det.detected]
    # background frame: the highest Sun among the clean inliers (a clear frame, not a hazy one)
    clean = pd.concat([d[d.calib_id == cid][f["inlier"] & (f["err"] < 0.3)] for cid, f in fits.items()])
    pick = clean if len(clean) else d
    sample = pick.iloc[int(pick.sun_apparent_elevation.argmax())] if len(pick) else None
    ax = axes[0]
    if sample is not None:
        img = cv2.cvtColor(read_half(root / sample.file), cv2.COLOR_BGR2RGB)
        ax.imshow(img, extent=(0, img.shape[1] * 2, img.shape[0] * 2, 0))
    for calib_id, f in fits.items():
        dd = d[d.calib_id == calib_id]
        cal = next(c for c in cals if c.file == calib_id)
        enu = azel_to_enu(dd.sun_azimuth.to_numpy(), dd.sun_apparent_elevation.to_numpy())
        pu, pv = cal.model.ray_to_pixel(enu @ f["R"])
        ax.scatter(dd.u, dd.v, s=6, label=f"detected ({calib_id})")
        ax.scatter(pu, pv, s=14, marker="x", label=f"projected, fitted pose (median {f['median']:.2f} deg)")
        axes[1].scatter(dd.utc.dt.hour + dd.utc.dt.minute / 60, f["err"], s=6, label=calib_id)
    ax.set_title(f"{station}: Sun detections over the sampled days")
    ax.legend(fontsize=7, loc="lower right")
    axes[1].set_xlabel("hour (UTC)")
    axes[1].set_ylabel("angular error of the fitted pose (deg)")
    axes[1].set_ylim(0, max(2.0, axes[1].get_ylim()[1]))
    axes[1].legend(fontsize=7)
    out = FIG_DIR / f"sun_validation_{station}.jpg"
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def manifest_check() -> str:
    p = Path("data/manifest.parquet")
    if not p.exists():
        return "manifest not present; skipped"
    m = pd.read_parquet(p)
    m = m[(m.dataset == "eye2sky") & m.utc.notna()].sample(300, random_state=0)
    worst = 0.0
    for (lat, lon), g in m.groupby(["latitude", "longitude"]):
        sp = sun_position(g.utc, lat, lon)
        worst = max(worst, float(np.abs(sp.zenith.to_numpy() - g.sun_zenith_deg.to_numpy()).max()),
                    float(np.abs(sp.azimuth.to_numpy() - g.sun_azimuth_deg.to_numpy()).max()))
    return f"300 Eye2Sky manifest rows: max |difference| {worst:.2e} deg in zenith and azimuth"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--day-step", type=int, default=7)
    ap.add_argument("--minute-step", type=int, default=5)
    ap.add_argument("--min-elevation", type=float, default=5.0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    root = Path(load_paths()["data_root"]) / "Eye2Sky"
    meta = root / "asi_meta"
    stations = a.stations or sorted({p.name.rsplit("_", 1)[1] for p in root.glob("2022/*/*/ASI_*") if p.is_dir()})
    print(f"stations with images: {stations}", flush=True)

    frames, records, cand_tables, figs = [], {}, {}, []
    for station in stations:
        det = detect_station(root, meta, station, a.day_step, a.minute_step, a.min_elevation)
        if not len(det):
            continue
        frames.append(det)
        cals = station_calibrations(meta, station)
        fits = {}
        for calib_id, g in det[det.detected].groupby("calib_id"):
            if len(g) < 3:
                continue
            cal = next(c for c in cals if c.file == calib_id)
            f = fit_calibration(cal, g)
            fits[calib_id] = f
            cand_tables[calib_id] = f["candidates"]
            roll, pitch, yaw = rotation_to_euler_zyx(f["R"])
            records[calib_id] = {
                "station": station, "status": f["status"], "n_sampled": int((det.calib_id == calib_id).sum()),
                "n_detected": int(len(g)), "n_inliers": int(f["inlier"].sum()), "inlier_span_hours": round(f["span_hours"], 2),
                "median_err_deg": round(f["median"], 4), "p90_err_deg": round(f["p90"], 4),
                "max_inlier_err_deg": round(f["max"], 4),
                "R": [[round(float(x), 10) for x in row] for row in f["R"]],
                "euler_zyx_deg": [round(float(np.degrees(x)), 4) for x in (roll, pitch, yaw)],
                "optical_axis_elevation_deg": round(float(np.degrees(np.arcsin(np.clip(f["R"][2, 2], -1, 1)))), 3),
                "declared_external_orientation": [float(x) for x in cal.external_orientation],
                "days": [str(g.utc.min().date()), str(g.utc.max().date())],
            }
            print(f"  {calib_id}: {len(g)} detections, {int(f['inlier'].sum())} inliers, median {f['median']:.3f} deg, "
                  f"p90 {f['p90']:.3f} deg -> {f['status']}", flush=True)
        if fits:
            figs.append(figure(station, det, fits, cals, root))

    if not records:
        print("no calibration could be fitted (no images with a visible Sun?)")
        return
    all_det = pd.concat(frames, ignore_index=True)
    all_det.to_parquet("data/sun_detections.parquet", index=False)

    # The declared-orientation reading: mean over calibrations of the median error, lowest wins.
    cand = pd.DataFrame(cand_tables)                     # rows = candidate names, columns = calib_id
    cand["mean"] = cand.mean(axis=1)
    cand = cand.sort_values("mean", kind="stable").sort_index(kind="stable").sort_values("mean", kind="stable")
    winner = str(cand.index[0])
    for calib_id, rec in records.items():
        cal = next(c for c in station_calibrations(meta, rec["station"]) if c.file == calib_id)
        rec["declared_convention_median_err_deg"] = round(float(cand.loc[winner, calib_id]), 4)
        rec["rotation_distance_fitted_vs_declared_deg"] = round(
            rotation_distance_deg(np.array(rec["R"]), eye2sky_rotation(cal.external_orientation, winner)), 4)
    meta_out = {"generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"), "eye2sky_convention": winner,
                "convention_mean_median_err_deg": round(float(cand["mean"].iloc[0]), 4),
                "runner_up": str(cand.index[1]), "runner_up_mean_median_err_deg": round(float(cand["mean"].iloc[1]), 4),
                "candidates_tested": int(len(cand)), "status_rule": STATUS_RULE,
                "sampling": {"day_step": a.day_step, "minute_step": a.minute_step, "min_apparent_elevation_deg": a.min_elevation}}
    save_pose_records(records, meta_out)
    consistency = manifest_check()
    write_report(records, cand, meta_out, all_det, consistency, figs)
    print(f"convention: {winner} (mean median error {meta_out['convention_mean_median_err_deg']:.3f} deg; runner-up "
          f"{meta_out['runner_up']} at {meta_out['runner_up_mean_median_err_deg']:.3f} deg)")
    print(f"manifest check: {consistency}")
    print(f"done in {time.perf_counter() - t0:.0f} s: configs/camera_poses.yaml, docs/data/sun_validation.md, "
          f"{len(figs)} figures", flush=True)


def write_report(records, cand, meta, det, consistency, figs) -> None:
    smp = meta["sampling"]
    rule = meta["status_rule"]
    lines = [
        "# Sun-based calibration validation (P044)", "",
        f"Generated {meta['generated_utc']} by `scripts/sun_validation.py` (sampling: every {smp['day_step']}th day, "
        f"every {smp['minute_step']} min, Sun >= {smp['min_apparent_elevation_deg']} deg).", "",
        "## Method", "",
        "The Sun disc is detected as the one saturated, disc-like blob inside the camera mask; its pixel becomes a camera "
        "ray through the station's OCamCalib model (P013), pvlib's SPA gives the Sun's apparent East-North-Up direction "
        "at the frame time, and a rotation camera -> ENU is fitted by Kabsch with outlier trimming. The declared "
        "`external_orientation` [roll, pitch, yaw] of the calibration files is read under every plausible convention "
        f"({meta['candidates_tested']} candidates: six Euler orders x eight sign patterns x four base frames x two camera "
        "frames x camera-to-world or world-to-camera) and the reading with the lowest mean median error wins.", "",
        "## Fitted poses", "",
        "| Calibration | Station | Days | Sampled | Detected | Inliers | Median err | p90 err | Declared reading err | "
        "Fitted vs declared | Axis elevation | Status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cid, r in records.items():
        lines.append(f"| {cid} | {r['station']} | {r['days'][0]} to {r['days'][1]} | {r['n_sampled']} | {r['n_detected']} | "
                     f"{r['n_inliers']} | {r['median_err_deg']:.3f} deg | {r['p90_err_deg']:.3f} deg | "
                     f"{r['declared_convention_median_err_deg']:.3f} deg | "
                     f"{r['rotation_distance_fitted_vs_declared_deg']:.3f} deg | {r['optical_axis_elevation_deg']:.2f} deg | "
                     f"**{r['status']}** |")
    rate = det.detected.mean()
    reasons = det[~det.detected].reason.value_counts().to_dict()
    lines += [
        "", f"Detection rate over all sampled frames: {rate:.1%} ({int(det.detected.sum()):,} of {len(det):,}); "
        f"undetected frames by reason: {reasons}.", "",
        f"Status rule: at least {rule['min_inliers']} inliers spanning >= {rule['min_span_hours']} h, "
        f"median <= {rule['max_median_deg']} deg, "
        f"p90 <= {rule['max_p90_deg']} deg.", "",
        "## Reading of the declared orientation", "",
        f"Winner: `{meta['eye2sky_convention']}` (mean median error {meta['convention_mean_median_err_deg']:.3f} deg); "
        f"runner-up `{meta['runner_up']}` at {meta['runner_up_mean_median_err_deg']:.3f} deg.", "",
        "| Candidate | " + " | ".join(c for c in cand.columns if c != "mean") + " | Mean |",
        "|---|" + "---|" * (len(cand.columns)),
    ]
    for name, row in cand.head(8).iterrows():
        cells = " | ".join(f"{row[c]:.3f}" for c in cand.columns if c != "mean")
        lines.append(f"| `{name}` | {cells} | {row['mean']:.3f} |")
    lines += [
        "", "Name: `<Euler order>|<signs of roll, pitch, yaw>|<base frame>|<camera frame>|<direction>`; the Euler rotation "
        "`R_order[0] R_order[1] R_order[2]` is built in the base frame, acts on the named camera frame (ADR-012 standard "
        "or OCamCalib's x-along-rows, y-along-columns, z away from the scene) and is read camera-to-world (`c2w`) or "
        "world-to-camera (`w2c`).", "",
        "## Manifest consistency", "",
        f"`sun_zenith_deg` / `sun_azimuth_deg` versus `stratia.geometry.sun.sun_position`: {consistency}.", "",
        "## Figures", "",
    ]
    lines += [f"- `{f.as_posix()}`" for f in figs]
    lines += [
        "", "## Decisions", "",
        "- Ray maps (P045) use the **fitted** pose where one has status `ok` and the declared orientation under the "
        "winning reading elsewhere (`stratia.geometry.pose.pose_for`); the table above is the expected accuracy of each.",
        "- A calibration whose fitted pose is `excluded` is not used for cloud-base-height pairing (P047) until it is "
        "understood.",
        "- Stations without images on disk (all except those above) carry the declared reading; its accuracy on the "
        "tested stations is the column *Declared reading err*. Rerun this script when images of a new station arrive "
        "(OLDLR).",
    ]
    Path("docs/data/sun_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
