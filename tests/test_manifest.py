import json

import pandas as pd

from stratia.data.manifest import SCHEMA, empty_columns, validate


def _valid_frame() -> pd.DataFrame:
    df = pd.DataFrame({
        "sample_id": ["a:1.jpg", "b:2.jpg"], "dataset": ["a", "b"], "image_file": ["1.jpg", "2.jpg"],
        "camera_id": ["cam", "cam"], "camera_type": ["fisheye_asi", "wide_angle_usb"],
        "utc": pd.to_datetime(["2022-06-01 12:00", None], utc=True), "latitude": [53.15, None], "longitude": [8.17, None],
        "sun_zenith_deg": [35.0, None], "width": [10, 20], "height": [10, 20], "licence": ["CC0", "owner"],
        "group_id": ["a:1.jpg", "b:2.jpg"], "official_split": ["train", "none"], "oktas_dist": [json.dumps({"8": 1.0}), None],
    })
    return empty_columns(df)


def test_valid_frame_passes_and_has_all_columns():
    df = _valid_frame()
    assert list(df.columns) == list(SCHEMA) and validate(df) == []


def test_detects_common_problems():
    df = _valid_frame()
    df.loc[1, "sample_id"] = "a:1.jpg"                 # duplicate id
    df.loc[0, "camera_type"] = "drone"                 # unknown camera type
    df.loc[0, "official_split"] = "holdout"            # unknown split
    df.loc[0, "oktas_dist"] = "{not json"              # broken JSON
    df.loc[0, "latitude"] = 123.0                      # out of range
    problems = " | ".join(validate(df))
    for needle in ("sample_id", "camera_type", "official_split", "oktas_dist", "latitude"):
        assert needle in problems, needle


def test_naive_timestamps_rejected():
    df = _valid_frame()
    df["utc"] = pd.to_datetime(["2022-06-01 12:00", None])
    assert any("UTC" in p for p in validate(df))
