import hashlib
import json

import pandas as pd
from PIL import Image

from stratia.data.b0268 import SIDECAR_SCHEMA, ingest_folder


def _sidecar_image(folder, name="BLR01_20261004T063015_123Z", tamper=False):
    img = folder / f"{name}.jpg"
    Image.new("RGB", (64, 48), (120, 160, 220)).save(img)
    digest = hashlib.sha256(img.read_bytes()).hexdigest()
    sc = {"schema": SIDECAR_SCHEMA, "file": img.name, "sha256": "0" * 64 if tamper else digest,
          "capture": {"utc": "2026-10-04T06:30:15.123+00:00", "time_source": "host clock (not GPS-disciplined)"},
          "site": {"id": "BLR01", "latitude": 12.971598, "longitude": 77.594566, "altitude_m": 920},
          "pointing": {"source": "declared_by_operator", "elevation_deg": 90, "azimuth_deg": None},
          "camera": {"readback": {"exposure": -7.0, "gain": 0.0, "exposure_seconds_estimate": 2 ** -7}},
          "sun": {"elevation_deg": 72.6, "azimuth_deg": 179.7}, "stats": {"clipped_fraction": 0.01}}
    img.with_suffix(".json").write_text(json.dumps(sc), encoding="utf-8")


def _exif_image(folder):
    img = folder / "b0268_legacy.jpg"
    exif = Image.Exif()
    sub = exif.get_ifd(0x8769)
    sub[36867] = "2026:09:27 15:44:21"      # DateTimeOriginal (local)
    sub[37521] = "390"                      # SubsecTimeOriginal
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2], gps[3], gps[4] = "N", (12.0, 55.0, 48.0), "E", (77.0, 30.0, 45.0)
    Image.new("RGB", (32, 24)).save(img, exif=exif)


def test_sidecar_ingest_and_rounding(tmp_path):
    _sidecar_image(tmp_path)
    r = ingest_folder(tmp_path, tmp_path).iloc[0]
    assert r.sha256_ok and r.has_sidecar and r.utc == pd.Timestamp("2026-10-04 06:30:15.123", tz="UTC")
    assert (r.latitude, r.longitude) == (12.97, 77.59)          # rounded to 0.01 deg for privacy
    assert r.pointing_elevation_deg == 90 and r.exposure_seconds == 2 ** -7 and (r.width, r.height) == (64, 48)


def test_sidecar_checksum_mismatch_is_reported(tmp_path):
    _sidecar_image(tmp_path, tamper=True)
    assert not ingest_folder(tmp_path, tmp_path).iloc[0].sha256_ok


def test_exif_fallback_converts_local_time_and_rounds_gps(tmp_path):
    _exif_image(tmp_path)
    r = ingest_folder(tmp_path, tmp_path, utc_offset_hours=5.5).iloc[0]
    assert r.utc == pd.Timestamp("2026-09-27 10:14:21.390", tz="UTC") and not r.has_sidecar
    assert (r.latitude, r.longitude) == (12.93, 77.51) and r.location_source == "EXIF GPS (rounded)"
    assert "UTC+5.5h" in r.time_source
