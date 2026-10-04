"""Unified sample manifest (P021): one row per image across all datasets, with a validated schema.

Columns filled later are present from the start (null until their phase): genus_set / etage_set (P034
ontology mapping), group_id refinement (P025 near-duplicates, P027 temporal blocks), cbh_station (P047
ceilometer pairing), split (P038 split generator).
"""

from __future__ import annotations

import json

import pandas as pd

CAMERA_TYPES = {"fisheye_asi", "consumer_photo", "fixed_lowcost", "wide_angle_usb", "wsi_patch", "wsi_crop"}
SPLITS = {"train", "val", "test", "none"}

# column -> (kind, nullable); kinds: str, int, float, bool, utc, json
SCHEMA: dict[str, tuple[str, bool]] = {
    "sample_id": ("str", False), "dataset": ("str", False), "image_file": ("str", False),
    "camera_id": ("str", False), "camera_type": ("str", False), "site": ("str", True),
    "utc": ("utc", True), "time_source": ("str", True),
    "latitude": ("float", True), "longitude": ("float", True),
    "sun_zenith_deg": ("float", True), "sun_azimuth_deg": ("float", True),
    "calib_id": ("str", True), "width": ("int", False), "height": ("int", False),
    "source_label": ("str", True), "official_split": ("str", True),
    "genus_set": ("json", True), "etage_set": ("json", True),
    "n_raters": ("int", True), "oktas_dist": ("json", True), "h_dist": ("json", True),
    "cl_dist": ("json", True), "cm_dist": ("json", True), "ch_dist": ("json", True),
    "seg_file": ("str", True), "has_layers": ("bool", False),
    "cbh_station": ("str", True), "licence": ("str", False), "group_id": ("str", False), "split": ("str", True),
}


def empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any schema column that is missing, as nulls (or False for has_layers)."""
    for col, (kind, _) in SCHEMA.items():
        if col not in df:
            df[col] = False if kind == "bool" else None
    return df[list(SCHEMA)]


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems = [f"missing column {c}" for c in SCHEMA if c not in df]
    if problems:
        return problems
    for col, (kind, nullable) in SCHEMA.items():
        s = df[col]
        if not nullable and s.isna().any():
            problems.append(f"{col}: {int(s.isna().sum())} nulls in a non-nullable column")
        v = s.dropna()
        if kind == "utc" and len(v) and (getattr(v.dt, "tz", None) is None or str(v.dt.tz) != "UTC"):
            problems.append(f"{col}: timestamps must be timezone-aware UTC")
        if kind == "json":
            bad = [x for x in v if not _is_json(x)]
            if bad:
                problems.append(f"{col}: {len(bad)} values are not valid JSON")
    if df.sample_id.duplicated().any():
        problems.append(f"sample_id: {int(df.sample_id.duplicated().sum())} duplicates")
    if not df.camera_type.isin(CAMERA_TYPES).all():
        problems.append(f"camera_type: unknown values {sorted(set(df.camera_type) - CAMERA_TYPES)}")
    for col in ("official_split", "split"):
        v = df[col].dropna()
        if not v.isin(SPLITS).all():
            problems.append(f"{col}: unknown values {sorted(set(v) - SPLITS)}")
    lat, lon, zen = df.latitude.dropna(), df.longitude.dropna(), df.sun_zenith_deg.dropna()
    if ((lat < -90) | (lat > 90)).any() or ((lon < -180) | (lon > 180)).any():
        problems.append("latitude/longitude out of range")
    if ((zen < 0) | (zen > 180)).any():
        problems.append("sun_zenith_deg out of range")
    if ((df.width <= 0) | (df.height <= 0)).any():
        problems.append("non-positive image size")
    return problems


def _is_json(x) -> bool:
    try:
        json.loads(x)
        return True
    except (TypeError, ValueError):
        return False
