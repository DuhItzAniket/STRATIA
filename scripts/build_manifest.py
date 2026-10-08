"""Build and validate the unified manifest (P021).

    python scripts/build_manifest.py

Writes data/manifest.parquet (git-ignored) and docs/data/manifest_summary.md. Exit code 1 if validation fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.b0268 import ingest_folder  # noqa: E402
from stratia.data.eye2sky import parse_image_name, select_calibration, station_calibrations  # noqa: E402
from stratia.data.manifest import empty_columns, validate  # noqa: E402
from stratia.data.mgcd import load_mgcd  # noqa: E402
from stratia.data.montenegro import load_annotations, load_items, soft_labels  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.data.segmentation import build_index  # noqa: E402
from stratia.data.temporal import times_from_filenames  # noqa: E402

LICENCES = {"ccsn": "CC0-1.0", "mgcd": "research use; terms by agreement", "swimcat": "CC-BY-NC-4.0",
            "swimseg": "CC-BY-NC-4.0", "swinseg": "CC-BY-NC-4.0", "swinyseg": "CC-BY-NC-4.0", "shwimseg": "CC-BY-NC-4.0",
            "almeria": "CC-BY-4.0", "montenegro": "CC-BY-4.0", "eye2sky": "CDLA-Sharing-1.0", "b0268": "owner"}
# The manifest holds nine Eye2Sky days per station (the audit and the image cache were built on them, P021-P032);
# the April-July download on disk is ingested for the ceilometer-site work when those images arrive (P047).
EYE2SKY_DAYS = ("2022-04-01", "2022-04-09")


def ccsn(root: Path) -> pd.DataFrame:
    files = sorted((root / "CCSN" / "CCSN_v2").glob("*/*.jpg"))
    return pd.DataFrame({"dataset": "ccsn", "image_file": [f.relative_to(root).as_posix() for f in files],
                         "camera_id": "ccsn-various", "camera_type": "consumer_photo",
                         "source_label": [f.parent.name for f in files], "official_split": "none"})


def mgcd(root: Path) -> pd.DataFrame:
    df = load_mgcd(root / "MGCD" / "MGCD")
    return pd.DataFrame({"dataset": "mgcd", "image_file": "MGCD/MGCD/" + df.image_file, "camera_id": "mgcd-asi",
                         "camera_type": "fisheye_asi", "source_label": df.class_name, "official_split": df.split})


def swimcat(root: Path) -> pd.DataFrame:
    files = sorted((root / "SWIMCAT" / "swimcat").glob("*/images/*"))
    return pd.DataFrame({"dataset": "swimcat", "image_file": [f.relative_to(root).as_posix() for f in files],
                         "camera_id": "wsi-singapore", "camera_type": "wsi_patch",
                         "source_label": [f.parent.parent.name for f in files], "official_split": "none"})


def segmentation(root: Path) -> pd.DataFrame:
    idx = build_index(root)
    almeria = idx.dataset == "almeria"
    # Almería's only timestamps are in its file names (asi_001_170328164030.jpg = 2017-03-28 16:40:30; the test set's
    # 20230126135101_00160.jpg); the time zone is not stated, so they are taken as UTC for day blocking (P027, P032),
    # and no sun position is derived from them.
    utc = times_from_filenames(idx.image_file).dt.tz_localize("UTC").where(almeria)
    return pd.DataFrame({"dataset": idx.dataset, "image_file": idx.image_file,
                         "camera_id": ("almeria-" + idx.camera).where(almeria, idx.camera.str.lower()),
                         "camera_type": almeria.map({True: "fisheye_asi", False: "wsi_crop"}),
                         "seg_file": idx.mask_file, "has_layers": idx.has_layers, "official_split": idx.split,
                         "utc": utc, "time_source": pd.Series("file name (time zone not stated, taken as UTC)",
                                                             index=idx.index).where(almeria & utc.notna())})


def montenegro(root: Path) -> pd.DataFrame:
    base = root / "Montenegro"
    soft = soft_labels(load_annotations(base)).merge(load_items(base), on="item_id")
    utc = (pd.to_datetime(soft.local_time) - pd.Timedelta(hours=1)).dt.tz_localize("UTC")
    return pd.DataFrame({"dataset": "montenegro", "image_file": "Montenegro/" + soft.image_file,
                         "camera_id": "montenegro-lowcost", "camera_type": "fixed_lowcost", "site": "Montenegro",
                         "utc": utc, "time_source": "local time, stated UTC+1 (daylight saving not verified)",
                         "source_label": soft.altitude_class_majority, "official_split": "none",
                         "n_raters": soft.n_raters, "oktas_dist": soft.N_dist, "h_dist": soft.h_dist,
                         "cl_dist": soft.CL_dist, "cm_dist": soft.CM_dist, "ch_dist": soft.CH_dist})


def eye2sky(root: Path) -> pd.DataFrame:
    base, meta = root / "Eye2Sky", root / "Eye2Sky" / "asi_meta"
    rows, cals = [], {}
    for f in sorted((base / "2022").rglob("*.jpg")):
        n = parse_image_name(f)
        if not (EYE2SKY_DAYS[0] <= pd.Timestamp(n.utc).strftime("%Y-%m-%d") <= EYE2SKY_DAYS[1]):
            continue
        if n.station not in cals:
            cals[n.station] = station_calibrations(meta, n.station)
        cal = select_calibration(cals[n.station], n.utc)
        rows.append({"dataset": "eye2sky", "image_file": f.relative_to(root).as_posix(), "camera_id": f"eye2sky-{n.station}",
                     "camera_type": "fisheye_asi", "site": n.station, "utc": pd.Timestamp(n.utc),
                     "time_source": "file name (UTC)", "calib_id": cal.file if cal else None,
                     "latitude": cal.latitude if cal else None, "longitude": cal.longitude if cal else None,
                     "official_split": "none"})
    return pd.DataFrame(rows)


def b0268(root: Path) -> pd.DataFrame:
    df = ingest_folder(root / "My Sample", root)
    return pd.DataFrame({"dataset": "b0268", "image_file": df.image_file, "camera_id": "b0268", "camera_type": "wide_angle_usb",
                         "site": "BLR", "utc": df.utc, "time_source": df.time_source, "latitude": df.latitude,
                         "longitude": df.longitude, "official_split": "none"})


def add_sun(df: pd.DataFrame) -> pd.DataFrame:
    df["sun_zenith_deg"], df["sun_azimuth_deg"] = None, None
    known = df.utc.notna() & df.latitude.notna() & df.longitude.notna()
    for (lat, lon), g in df[known].groupby(["latitude", "longitude"]):
        sp = pvlib.solarposition.get_solarposition(pd.DatetimeIndex(g.utc), lat, lon)
        df.loc[g.index, "sun_zenith_deg"] = sp.zenith.to_numpy()
        df.loc[g.index, "sun_azimuth_deg"] = sp.azimuth.to_numpy()
    df["sun_zenith_deg"] = df.sun_zenith_deg.astype(float)
    df["sun_azimuth_deg"] = df.sun_azimuth_deg.astype(float)
    return df


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = Path(load_paths()["data_root"])
    parts = [ccsn(root), mgcd(root), swimcat(root), segmentation(root), montenegro(root), eye2sky(root), b0268(root)]
    df = pd.concat(parts, ignore_index=True)
    df["utc"] = pd.to_datetime(df["utc"], utc=True)
    df["licence"] = df.dataset.map(LICENCES)
    df["sample_id"] = df.dataset + ":" + df.image_file
    df["group_id"] = df.sample_id
    df["has_layers"] = df.get("has_layers", False)
    df["has_layers"] = df.has_layers.fillna(False).astype(bool)
    inv = pd.read_parquet("data/inventory.parquet", columns=["relpath", "width", "height"])
    inv = inv.rename(columns={"relpath": "image_file"})
    df = df.drop(columns=[c for c in ("width", "height") if c in df]).merge(inv, on="image_file", how="left")
    df = add_sun(df)
    df = empty_columns(df)
    df["width"], df["height"] = df.width.astype("Int64"), df.height.astype("Int64")
    problems = validate(df)
    df.to_parquet("data/manifest.parquet", index=False)

    g = df.groupby("dataset")
    summary = pd.DataFrame({"rows": g.size(), "camera_types": g.camera_type.agg(lambda s: ", ".join(sorted(set(s)))),
                            "with_utc": g.utc.agg(lambda s: int(s.notna().sum())),
                            "with_sun": g.sun_zenith_deg.agg(lambda s: int(s.notna().sum())),
                            "with_seg": g.seg_file.agg(lambda s: int(s.notna().sum())),
                            "with_soft_labels": g.n_raters.agg(lambda s: int(s.notna().sum())),
                            "official_splits": g.official_split.agg(lambda s: json.dumps(s.value_counts().to_dict()))})
    L = ["# Manifest summary", "",
         f"Generated by `scripts/build_manifest.py` (P021): {len(df):,} rows, {len(df.columns)} columns; "
         f"validation: {'passed' if not problems else 'FAILED'}.", "", summary.to_markdown(), ""]
    if problems:
        L += ["## Validation problems", ""] + [f"- {p}" for p in problems]
    Path("docs/data/manifest_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
