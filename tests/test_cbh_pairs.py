"""P047: image-ceilometer pairing, zenith geometry, day splits and the lock entry."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from stratia.data import test_lock as tl
from stratia.geometry.ocam import OcamModel
from stratia.geometry.pose import Pose, euler_to_rotation
from stratia.labels import cbh_pairs as cp

OLDLR = OcamModel(ss=(-642.6690459406254, 0.0, 0.0002441927468927244, 4.838462720720562e-07),
                  xc=1028.60, yc=1072.59, width=2112, height=2048)


def _targets(start="2022-04-01 10:00:00", n=8):
    times = pd.date_range(start, periods=n, freq="30s", tz="UTC")
    labels = ["low", "low", "none", "mid", "mid", "high", "mixed", "low"]
    cbh = [800.0, 850.0, np.nan, 3000.0, 3100.0, 7500.0, 1200.0, 900.0]
    return pd.DataFrame({"ceilometer": "CDLRA", "time": times, "label": labels, "etage_wmo": labels, "cbh_m": cbh,
                         "cbh2_m": [np.nan] * n, "layers": [1.0] * n, "confidence": [1.0, 0.9, 1.0, 0.8, 1.0, 1.0, 0.4, 1.0],
                         "valid": [True, True, True, False, True, True, True, True], "mixed": [False] * 6 + [True, False],
                         "daytime": [True] * n})


def test_frames_on_disk_cadence(tmp_path):
    day = tmp_path / "2022" / "04" / "01" / "ASI_20220401_OLDLR" / "OLDLR" / "2022" / "04" / "01" / "10"
    day.mkdir(parents=True)
    for stamp in ("20220401100000", "20220401100030", "20220401100200", "20220401100330", "20220401100400"):
        (day / f"{stamp}_160.jpg").write_bytes(b"x")
    (tmp_path / "2022" / "04" / "01" / "ASI_20220401_OLDLR.zip").write_bytes(b"x")       # a leftover zip is ignored
    f = cp.frames_on_disk(tmp_path, "OLDLR", cadence_s=120)
    assert [t.strftime("%H%M%S") for t in f.utc] == ["100000", "100200", "100400"]
    assert f.image_file.iloc[0] == "Eye2Sky/2022/04/01/ASI_20220401_OLDLR/OLDLR/2022/04/01/10/20220401100000_160.jpg"
    assert cp.frames_on_disk(tmp_path, "WESTE").empty


def test_join_targets_nearest_within_gap_and_valid():
    frames = pd.DataFrame({"station": "OLDLR", "image_file": [f"f{i}" for i in range(4)],
                           "utc": pd.to_datetime(["2022-04-01 10:00:10", "2022-04-01 10:01:25", "2022-04-01 10:01:50",
                                                  "2022-04-01 10:09:00"], utc=True)})
    j = cp.join_targets(frames, _targets(), max_gap_s=15.0)
    assert list(j.paired) == [True, False, True, False]        # 10:01:25 -> 10:01:30 is invalid; 10:09 has no grid point
    assert j.label.iloc[0] == "low" and j.gap_s.iloc[0] == 10
    assert j.label.iloc[2] == "mid" and j.cbh_m.iloc[2] == 3100.0 and j.gap_s.iloc[2] == 10
    assert cp.join_targets(frames.iloc[0:0], _targets()).empty


def test_zenith_geometry_for_an_upward_camera():
    pose = Pose(R=euler_to_rotation(0.0, 0.0, 0.7, "zyx"), source="test")      # axis exactly at the zenith
    mask = np.ones((2048, 2112), dtype=bool)
    g = cp.zenith_geometry(OLDLR, pose, mask, roi_deg=10.0)
    assert abs(g.u - OLDLR.yc) < 1e-6 and abs(g.v - OLDLR.xc) < 1e-6 and g.inside_mask
    assert 100 < g.roi_radius_px < 130                       # ~11 px per degree near the centre of the Q25
    tilted = Pose(R=euler_to_rotation(0.0, np.radians(3.0), 0.0, "zyx"), source="test")
    g2 = cp.zenith_geometry(OLDLR, tilted, mask)
    assert abs(g2.u - OLDLR.yc) > 20 and g2.inside_mask        # the zenith moved off the optical centre
    mask[:, :] = False
    assert not cp.zenith_geometry(OLDLR, pose, mask).inside_mask


def test_assign_days_inherits_and_balances():
    days = [dt.date(2022, 4, 1) + dt.timedelta(d) for d in range(60)]
    inherited = {days[0]: "train", days[2]: "test", days[6]: "val"}
    strata = {d: ("low" if i % 3 else "high") for i, d in enumerate(days)}
    out = cp.assign_days(days, inherited, {"train": 0.7, "val": 0.1, "test": 0.2}, seed=0, strata=strata)
    assert out[days[0]] == "train" and out[days[2]] == "test" and out[days[6]] == "val"
    rest = [d for d in days if d not in inherited]
    shares = pd.Series([out[d] for d in rest]).value_counts(normalize=True)
    assert abs(shares["train"] - 0.7) < 0.08 and abs(shares["test"] - 0.2) < 0.08
    # every stratum sees every split
    for stratum in ("low", "high"):
        got = {out[d] for d in rest if strata[d] == stratum}
        assert got == {"train", "val", "test"}
    again = cp.assign_days(days, inherited, {"train": 0.7, "val": 0.1, "test": 0.2}, seed=0, strata=strata)
    assert again == out                                       # deterministic


def _pairs():
    utc = pd.to_datetime(["2022-04-01 10:00:00", "2022-04-01 10:00:30", "2022-04-01 10:02:30", "2022-04-02 10:00:00",
                          "2022-04-02 10:03:30", "2022-04-02 22:00:00"], utc=True)
    return pd.DataFrame({"station": ["OLDLR"] * 3 + ["WESTE"] * 3, "image_file": [f"Eye2Sky/f{i}.jpg" for i in range(6)],
                         "utc": utc, "paired": [True, True, True, True, True, True], "daytime": [True] * 5 + [False],
                         "zenith_inside_mask": [True, True, False, True, True, True],
                         "label": ["low", "low", "high", "mid", "mid", "none"], "cbh_m": [800, 850, 7500, 3000, 3100, np.nan],
                         "confidence": 1.0, "mixed": False, "gap_s": 0.0, "ceilometer": "CDLRA", "time": utc})


def test_split_table_day_share_and_summary():
    pairs = cp.add_day_label_share(_pairs())
    assert pairs.day_label_share.iloc[0] == 2 / 3 and pairs.day_label_share.iloc[2] == 1 / 3
    strata = cp.day_strata(pairs)
    assert strata[dt.date(2022, 4, 1)] == "low" and strata[dt.date(2022, 4, 2)] == "mid"
    day_split = {dt.date(2022, 4, 1): "train", dt.date(2022, 4, 2): "test"}
    splits = cp.split_table(pairs, day_split)
    assert len(splits) == 4                                   # the masked-out zenith and the night frame are gone
    assert splits.sample_id.iloc[0] == "eye2sky:Eye2Sky/f0.jpg"
    assert set(splits[splits.split == "test"].station) == {"WESTE"}
    s = cp.summary(pairs, splits)
    assert s["per_station"]["OLDLR"]["pairs"] == 3 and s["per_station"]["OLDLR"]["labels"]["low"] == 2
    assert s["days_by_split"] == {"test": 1, "train": 1}
    md = cp.report_markdown(s, 120, 15.0, ["decision"], ["fig.png"])
    assert "| OLDLR |" in md and "| test | 1 |" in md


def test_lock_covers_the_cbh_protocol(tmp_path):
    splits = Path(tmp_path)
    pd.DataFrame({"sample_id": ["a", "b", "c"], "station": ["OLDLR", "OLDLR", "WESTE"], "day": ["d1", "d2", "d3"],
                  "label": ["low", "mid", "low"], "split": ["test", "train", "test"]}).to_parquet(splits / "cbh_pairs.parquet")
    entries = tl.test_entries(splits)
    assert set(entries) == {"cbh_pairs/OLDLR/test", "cbh_pairs/WESTE/test"}
    assert entries["cbh_pairs/OLDLR/test"]["n"] == 1
    lock = tl.build_lock(splits, 1, 0, "abc")
    assert tl.verify_lock(lock, splits) == []
