"""Ingest of the owner's Arducam B0268 captures (P019).

Two sources:
* CloudScope interim sky logger output: image + JSON sidecar (schema `cloudscope.sky_logger.frame/1`), with UTC
  time, camera read-back, site, pointing and Sun position. The image's SHA-256 is checked against the sidecar.
* Legacy frames without sidecars (the 25 Windows Camera captures of 2026-09-27): EXIF DateTimeOriginal +
  SubsecTimeOriginal in local time (converted with the given UTC offset, IST +05:30 by default) and EXIF GPS.

Privacy: EXIF GPS can reveal where the owner lives. Coordinates are rounded to LOCATION_DECIMALS (0.01 deg,
about 1 km), which is ample for Sun geometry; exact coordinates are never stored.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from PIL import ExifTags, Image

SIDECAR_SCHEMA = "cloudscope.sky_logger.frame/1"
LOCATION_DECIMALS = 2
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def _round_loc(x: float | None) -> float | None:
    return None if x is None else round(float(x), LOCATION_DECIMALS)


def _exif(path: Path) -> dict:
    ex = Image.open(path).getexif()
    sub = {str(ExifTags.TAGS.get(k, k)): v for k, v in ex.get_ifd(0x8769).items()}
    gps = {str(ExifTags.GPSTAGS.get(k, k)): v for k, v in ex.get_ifd(0x8825).items()}
    return {"sub": sub, "gps": gps}


def _gps_deg(gps: dict) -> tuple[float | None, float | None]:
    if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
        return None, None

    def deg(t):
        return float(t[0]) + float(t[1]) / 60 + float(t[2]) / 3600

    lat, lon = deg(gps["GPSLatitude"]), deg(gps["GPSLongitude"])
    if gps.get("GPSLatitudeRef", "N") == "S":
        lat = -lat
    if gps.get("GPSLongitudeRef", "E") == "W":
        lon = -lon
    return lat, lon


def _from_sidecar(img: Path, sc: dict) -> dict:
    if sc.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"{img.name}: unsupported sidecar schema {sc.get('schema')!r}")
    data = img.read_bytes()
    cam = sc.get("camera", {}).get("readback", {})
    site, sun, pointing = sc.get("site") or {}, sc.get("sun") or {}, sc.get("pointing") or {}
    return {
        "utc": pd.Timestamp(sc["capture"]["utc"]).tz_convert("UTC"),
        "time_source": "sidecar: " + sc["capture"].get("time_source", "host clock"),
        "sha256_ok": hashlib.sha256(data).hexdigest() == sc.get("sha256"),
        "exposure_readback": cam.get("exposure"), "gain_readback": cam.get("gain"),
        "exposure_seconds": cam.get("exposure_seconds_estimate"),
        "latitude": _round_loc(site.get("latitude")), "longitude": _round_loc(site.get("longitude")),
        "location_source": "sidecar (rounded)" if site.get("latitude") is not None else None,
        "pointing_elevation_deg": pointing.get("elevation_deg"), "pointing_azimuth_deg": pointing.get("azimuth_deg"),
        "pointing_source": pointing.get("source"),
        "sun_elevation_deg": sun.get("elevation_deg"), "sun_azimuth_deg": sun.get("azimuth_deg"),
        "clipped_fraction": (sc.get("stats") or {}).get("clipped_fraction"),
    }


def _from_exif(img: Path, utc_offset_hours: float) -> dict:
    ex = _exif(img)
    rec = {"sha256_ok": None, "exposure_readback": None, "gain_readback": None, "exposure_seconds": None,
           "pointing_elevation_deg": None, "pointing_azimuth_deg": None, "pointing_source": None,
           "sun_elevation_deg": None, "sun_azimuth_deg": None, "clipped_fraction": None}
    dto = ex["sub"].get("DateTimeOriginal")
    if dto:
        local = datetime.strptime(dto, "%Y:%m:%d %H:%M:%S")
        sub = ex["sub"].get("SubsecTimeOriginal")
        if sub:
            local += timedelta(milliseconds=int(str(sub)[:3].ljust(3, "0")))
        tz = timezone(timedelta(hours=utc_offset_hours))
        rec["utc"] = pd.Timestamp(local.replace(tzinfo=tz)).tz_convert("UTC")
        rec["time_source"] = f"exif local time, assumed UTC{utc_offset_hours:+g}h"
    else:
        rec["utc"] = pd.Timestamp(datetime.fromtimestamp(img.stat().st_mtime, tz=UTC))
        rec["time_source"] = "file modification time"
    lat, lon = _gps_deg(ex["gps"])
    rec.update(latitude=_round_loc(lat), longitude=_round_loc(lon),
               location_source="EXIF GPS (rounded)" if lat is not None else None)
    return rec


def ingest_folder(folder: str | Path, data_root: str | Path, utc_offset_hours: float = 5.5) -> pd.DataFrame:
    """One row per image under `folder` (recursive), sidecar first, EXIF fallback."""
    folder, data_root = Path(folder), Path(data_root)
    rows = []
    for img in sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT):
        sidecar = img.with_suffix(".json")
        rec = _from_sidecar(img, json.loads(sidecar.read_text(encoding="utf-8"))) if sidecar.exists() \
            else _from_exif(img, utc_offset_hours)
        with Image.open(img) as im:
            rec["width"], rec["height"] = im.size
        rec["image_file"] = img.relative_to(data_root).as_posix()
        rec["has_sidecar"] = sidecar.exists()
        rows.append(rec)
    return pd.DataFrame(rows)
