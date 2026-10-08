import numpy as np
import pandas as pd

from stratia.labels import ceilometer_targets as ct


def _records():
    """Records every 15 s: 10 min of cloud at 1,500 m, 10 min clear, 10 min half cloudy, with a flagged record and a
    second layer in the first block."""
    n = 120
    t = pd.Timestamp("2022-06-01 12:00", tz="UTC") + pd.to_timedelta(np.arange(n) * 15, unit="s")
    cbh1 = np.full(n, np.nan)
    cbh1[:40] = 1500 + np.arange(40) * 5.0                 # 1,500 .. 1,695 m
    cbh1[80:120:2] = 6500.0                                # every other record cloudy at 6.5 km
    cbh2 = np.full(n, np.nan)
    cbh2[:40] = 4000.0
    layers = (~np.isnan(cbh1)).astype(int) + (~np.isnan(cbh2)).astype(int)
    qc = np.zeros(n, bool)
    qc[2] = True                                            # a flagged record inside the first block
    return pd.DataFrame({"ceilometer": "A", "time": t, "cbh1": cbh1, "cbh2": cbh2, "n_layers": layers, "qc_any": qc})


def test_targets_follow_the_window_rule():
    rec = _records()
    times = pd.DatetimeIndex([pd.Timestamp("2022-06-01 12:00:30", tz="UTC"),    # cloud block
                              pd.Timestamp("2022-06-01 12:15:00", tz="UTC"),    # clear block
                              pd.Timestamp("2022-06-01 12:25:00", tz="UTC"),    # half cloudy block
                              pd.Timestamp("2022-06-01 13:30:00", tz="UTC")])   # no records
    t = ct.targets_at(rec, times, half_window_s=30)
    first = t.iloc[0]
    assert first.n_records == 5 and first.n_clean == 4 and abs(first.qc_share - 0.2) < 1e-9   # record 2 is flagged
    assert first.label == "low" and first.cloud_share == 1.0 and first.confidence == 1.0
    assert 1500 <= first.cbh_m <= 1520 and first.cbh_min_m == 1500 and first.cbh_max_m == 1520 and first.cbh2_m == 4000
    assert first.layers == 2.0 and first.etage_wmo == "low" and not first.mixed
    clear = t.iloc[1]
    assert clear.label == "none" and clear.cloud_share == 0.0 and clear.confidence == 1.0 and np.isnan(clear.cbh_m)
    half = t.iloc[2]
    assert half.mixed and 0.4 <= half.cloud_share <= 0.6 and half.label in {"high", "mixed"} and half.cbh_m == 6500
    assert half.etage_wmo in {"mid", None}                     # 6.5 km is 'high' at 6 km but 'mid' at the WMO 7 km boundary
    empty = t.iloc[3]
    assert not empty.valid and pd.isna(empty.label) and np.isnan(empty.confidence) and empty.n_records == 0


def test_etage_thresholds_grid_and_summary():
    assert ct.etage_of([np.nan, 1999, 2000, 5999, 6000, 6999]).tolist() == ["none", "low", "mid", "mid", "high", "high"]
    assert ct.etage_of([6500], ct.WMO_EDGES_M).tolist() == ["mid"]
    rec = _records()
    grid = ct.time_grid(rec, step_s=30)
    assert len(grid) == 2880 and grid[0] == pd.Timestamp("2022-06-01 00:00", tz="UTC") and grid.tz is not None
    t = ct.targets_at(rec, grid)
    t["daytime"] = ct.daytime(grid, 53.15, 8.17)
    assert 0.5 < t.daytime.mean() < 0.8                                   # June at 53 N: a long day
    s = ct.summary(t)
    assert s["all"]["grid_points"] == 2880 and 0 < s["all"]["valid_share"] < 0.05
    assert abs(sum(s["all"]["labels"].values()) - 1) < 1e-9
    text = ct.report_markdown({"A": s}, 30, ["decided"])
    assert "## A" in text and "| all |" in text and "- decided" in text
