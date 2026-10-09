"""Consumer-view synthesis set (P049).

    python scripts/synthesize_views.py [--stations OLDLR WESTE AURIC BARSE] [--per-day 6] [--views 3] [--width 1024]

For every station with a fitted pose (configs/camera_poses.yaml): sample `per-day` frames per day on disk (Sun at
least 10 deg up, spread over the day), render `views` B0268-like 105-degree views per frame at random pointings
(elevation 15-70 deg, any azimuth, roll +-3 deg) with the scaled nominal B0268 model, write them as JPEGs under
<cache_root>/synthetic_views/<station>/, and index them in data/synthetic_views.parquet (ignored): pointing, Sun,
validity share, and the ceilometer label inherited from data/ceilometer_targets.parquet at the two ceilometer
sites (flat-layer assumption; provisional pairing until P047). Writes docs/data/synthetic_views.md and
docs/data/figures/synthetic_views.jpg (source frames with the view footprints, the views, their zenith contours).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.eye2sky import load_mask, mask_path, parse_image_name, select_calibration, station_calibrations  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.geometry import synthesis as S  # noqa: E402
from stratia.geometry.cameras import load_cloudscope_model  # noqa: E402
from stratia.geometry.pose import load_pose_records, pose_for  # noqa: E402
from stratia.geometry.raymap import zenith_angle_deg  # noqa: E402
from stratia.geometry.sun import sun_position  # noqa: E402

FIG = Path("docs/data/figures/synthetic_views.jpg")
DOC = Path("docs/data/synthetic_views.md")
B0268_FILE = Path("configs/cameras/b0268_nominal.camera.json")


def day_frames(day_dir: Path) -> list[Path]:
    return sorted(day_dir.rglob("*.jpg"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--per-day", type=int, default=6)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--day-step", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    paths = load_paths()
    root = Path(paths["data_root"]) / "Eye2Sky"
    meta = root / "asi_meta"
    out_root = Path(paths["cache_root"]) / "synthetic_views"
    records = load_pose_records()
    virt = S.scaled_camera(load_cloudscope_model(B0268_FILE), a.width)
    targets = pd.read_parquet("data/ceilometer_targets.parquet") if Path("data/ceilometer_targets.parquet").exists() else None
    rng = np.random.default_rng(a.seed)
    stations = a.stations or sorted({p.name.rsplit("_", 1)[1] for p in root.glob("2022/*/*/ASI_*") if p.is_dir()})
    rows, gallery = [], []
    for station in stations:
        cals = station_calibrations(meta, station)
        site_targets = None
        if targets is not None and station in S.SITE_CEILOMETER:
            site_targets = targets[targets.ceilometer == S.SITE_CEILOMETER[station]].reset_index(drop=True)
        days = sorted(p for p in root.glob(f"2022/*/*/ASI_*_{station}") if p.is_dir())[:: a.day_step]
        masks = {}
        n_station = 0
        for day in days:
            frames = day_frames(day)
            if not frames:
                continue
            names = [parse_image_name(f) for f in frames]
            cal = select_calibration(cals, names[0].utc)
            if cal is None:
                continue
            pose = pose_for(cal.file, cal.external_orientation, records)
            if pose is None or pose.source != "fitted":
                print(f"{station} {day.name}: no fitted pose for {cal.file}; skipped")
                continue
            sp = sun_position([n.utc for n in names], cal.latitude, cal.longitude, cal.altitude)
            ok = np.nonzero(sp.elevation.to_numpy() >= 10.0)[0]
            if len(ok) == 0:
                continue
            pick = ok[np.linspace(0, len(ok) - 1, a.per_day).round().astype(int)]
            if cal.file not in masks:
                masks[cal.file] = load_mask(mask_path(meta, cal))
            mask = masks[cal.file]
            for i in pick:
                buf = np.fromfile(str(frames[i]), dtype=np.uint8)
                src = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                label = S.inherit_label(site_targets, names[i].utc) if site_targets is not None else None
                for k in range(a.views):
                    p = S.random_pointing(rng)
                    vpose = S.pointing_pose(p)
                    view = S.render_view(src, cal.model, pose, virt, vpose, mask)
                    if view.valid.mean() < 0.5:
                        continue
                    rel = Path(station) / f"{names[i].utc:%Y%m%d%H%M%S}_{k}.jpg"
                    dst = out_root / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    ok_enc, enc = cv2.imencode(".jpg", cv2.cvtColor(view.image, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if ok_enc:
                        enc.tofile(str(dst))
                    row = {"station": station, "calib_id": cal.file, "source_file": str(frames[i].relative_to(root)),
                           "utc": names[i].utc, "view_file": str(rel.as_posix()), "azimuth_deg": p.azimuth_deg,
                           "elevation_deg": p.elevation_deg, "roll_deg": p.roll_deg, "width": virt.width, "height": virt.height,
                           "sun_azimuth_deg": float(sp.azimuth.iloc[i]), "sun_zenith_deg": float(sp.zenith.iloc[i]),
                           "valid_share": float(view.valid.mean()), "label": None, "cbh_m": None, "confidence": None}
                    if label is not None:
                        row.update({"label": label["label"], "cbh_m": label["cbh_m"], "confidence": label["confidence"],
                                    "label_gap_s": label["gap_s"], "label_assumption": label["assumption"]})
                    rows.append(row)
                    n_station += 1
                    want = station in S.SITE_CEILOMETER or len(gallery) < 2
                    if len(gallery) < 4 and k == 0 and view.valid.mean() > 0.8 and want:
                        gallery.append((station, src, cal, pose, p, vpose, view, row))
        print(f"{station}: {n_station} views from {len(days)} days", flush=True)
    df = pd.DataFrame(rows)
    index = Path("data/synthetic_views.parquet")
    if index.exists():                                   # keep the stations this run did not touch
        old = pd.read_parquet(index)
        df = pd.concat([old[~old.station.isin(stations)], df], ignore_index=True)
    df.to_parquet(index, index=False)
    figure(gallery, virt)
    report(df, a, virt)
    print(f"done in {time.perf_counter() - t0:.0f} s: {len(df)} views -> {out_root}, data/synthetic_views.parquet, {DOC}, {FIG}")


def figure(gallery, virt) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not gallery:
        return
    fig, axes = plt.subplots(len(gallery), 3, figsize=(13, 3.6 * len(gallery)))
    axes = np.atleast_2d(axes)
    for r, (station, src, cal, pose, p, vpose, view, row) in enumerate(gallery):
        small = cv2.resize(src, (512, 512), interpolation=cv2.INTER_AREA)
        us, vs = S.source_footprint(virt, vpose, cal.model, pose)
        axes[r, 0].imshow(small)
        axes[r, 0].plot(us * 512 / cal.model.width, vs * 512 / cal.model.height, "-", color="yellow", lw=1.2)
        axes[r, 0].set_title(f"{station} {row['utc']:%Y-%m-%d %H:%M}: view footprint "
                             f"(az {p.azimuth_deg:.0f}, el {p.elevation_deg:.0f})", fontsize=8)
        axes[r, 1].imshow(view.image)
        axes[r, 1].set_title(f"synthetic 105 deg view, valid {row['valid_share']:.0%}", fontsize=8)
        rm = S.view_ray_map(virt, vpose, row["sun_azimuth_deg"], row["sun_zenith_deg"], view.valid, 512)
        zen = zenith_angle_deg(rm)
        axes[r, 2].imshow(cv2.resize(view.image, (512, 512), interpolation=cv2.INTER_AREA))
        cu = np.arange(32) * 16 + 7.5
        cs = axes[r, 2].contour(cu, cu, np.nan_to_num(zen, nan=180.0), levels=[20, 40, 60, 80], colors="cyan",
                                linewidths=0.8)
        axes[r, 2].clabel(cs, fmt="%d°", fontsize=7)
        lab = f"label {row['label']} ({row['cbh_m']:.0f} m)" if row.get("label") else "no ceilometer at this station"
        axes[r, 2].set_title(f"ray map: zenith angle; {lab}", fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=75)
    plt.close(fig)


def report(df: pd.DataFrame, a, virt) -> None:
    lines = ["# Consumer-view synthesis (P049)", "",
             f"Generated by `scripts/synthesize_views.py` (per day {a.per_day} frames, {a.views} views each, "
             f"{virt.width} x {virt.height} px, nominal B0268 lens at 105 deg horizontal field of view, elevation 15-70 deg, "
             "any azimuth, roll +-3 deg; views with less than 50 % valid pixels discarded).", "",
             "| Station | Views | Days | With ceilometer label | Label mix | Mean valid share |", "|---|---|---|---|---|---|"]
    for station, g in df.groupby("station"):
        labelled = g.label.notna()
        mix = g[labelled].label.value_counts().to_dict() if labelled.any() else "-"
        lines.append(f"| {station} | {len(g)} | {g.utc.dt.date.nunique()} | {int(labelled.sum())} | {mix} | "
                     f"{g.valid_share.mean():.1%} |")
    lines += ["", f"Total: {len(df)} views; index `data/synthetic_views.parquet` (ignored); images under "
              "`<cache_root>/synthetic_views/`.",
              "", "## Labels", "",
              "Views at OLDLR and WESTE inherit the ceilometer target of the source frame (nearest 30 s grid point of "
              "`data/ceilometer_targets.parquet` within 15 s, valid windows only): the cloud-base **height** of the zenith "
              "measurement is assigned to the pointed view under the flat-layer assumption (a base height is a height, not "
              "a slant range; the layer is taken as uniform over the few kilometres the view spans). This pairing rule is "
              "provisional until P047 fixes it; the `label_assumption` column marks every inherited label.", "",
              "## Visual check", "", f"`{FIG.as_posix()}`: source frame with the view's footprint (yellow), the rendered view, "
              "and its ray-map zenith contours with the inherited label."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
