import numpy as np
import pandas as pd
from PIL import Image

from stratia.data import ceilometer_qc as cq
from stratia.data import imbalance as ib


def test_distribution_and_imbalance_metrics():
    d = ib.distribution(pd.Series(["a", "a", "b", None, "c", "a"]))
    assert d.value.tolist() == ["a", "b", "c"] and d["count"].tolist() == [3, 1, 1]
    assert abs(d.share.sum() - 1) < 1e-9
    m = ib.imbalance_metrics(d["count"].values)
    assert m["classes"] == 3 and m["ratio"] == 3.0 and 0 < m["entropy"] < 1 and 1 < m["effective_classes"] < 3
    even = ib.imbalance_metrics(np.array([5, 5]))
    assert even["classes"] == 2 and even["ratio"] == 1.0 and abs(even["entropy"] - 1) < 1e-9
    assert abs(even["effective_classes"] - 2) < 1e-9
    ordered = ib.distribution(pd.Series(["x"]), order=["y", "x"])
    assert ordered.value.tolist() == ["y", "x"] and ordered["count"].tolist() == [0, 1]


def test_bins():
    assert ib.cbh_bin([np.nan, 500, 2000, 5999, 6000, 11000]).tolist() == ["none", "low", "mid", "mid", "high", "high"]
    assert ib.sun_bin([10, 30, 69.9, 85, 120]).tolist() == ["0-30", "30-50", "50-70", "85+", "85+"]
    assert ib.cloud_fraction_bin([0, 0.049, 0.5, 1.0]).tolist() == ["clear <5%", "clear <5%", "50-75%", "overcast >95%"]


def test_mask_cloud_fraction_reads_swim_and_almeria_masks(tmp_path):
    swim = np.zeros((10, 10), np.uint8)
    swim[:5] = 255                                            # top half cloud
    Image.fromarray(swim).save(tmp_path / "swim.png")
    r = ib.mask_cloud_fraction(("swimseg", str(tmp_path / "swim.png")))
    assert r["cloud_fraction"] == 0.5 and np.isnan(r["low"]) and r["error"] is None
    al = np.array([[0, 1, 2, 3, 4]] * 2, np.uint8)             # camera mask, sky, low, mid, high
    Image.fromarray(al).save(tmp_path / "al.png")
    r = ib.mask_cloud_fraction(("almeria", str(tmp_path / "al.png")))
    assert abs(r["cloud_fraction"] - 0.75) < 1e-9 and abs(r["low"] - 0.25) < 1e-9 and abs(r["high"] - 0.25) < 1e-9
    assert ib.mask_cloud_fraction(("swimseg", str(tmp_path / "missing.png")))["error"]


def test_sampling_rules():
    w = ib.source_weights({"big": 10000, "small": 100}, power=0.5)
    assert abs(w.natural.sum() - 1) < 1e-9 and abs(w.power.sum() - 1) < 1e-9
    assert w.set_index("source").power["small"] > w.set_index("source").natural["small"]        # small sources lifted
    assert w.set_index("source").power["small"] < 0.5                                             # but not to uniform
    cb = ib.class_balanced_weights(np.array([1000, 10, 1]))
    assert abs(cb.mean() - 1) < 1e-9 and cb[0] < cb[1] < cb[2]


def test_imbalance_report_and_figure(tmp_path):
    d = ib.distribution(pd.Series(["a", "b", "b"]))
    sources = ib.source_weights({"s1": 10, "s2": 5})
    cb = {"demo": pd.DataFrame({"class": ["a", "b"], "images": [1, 2], "weight": [1.2, 0.8]})}
    text = ib.report_markdown([("Demo", "note.", d)], sources, cb, ["decided"], "fig.png")
    assert "| b | 2 | 66.7% |" in text and "| s1 | 10 |" in text and "- decided" in text and "2 classes" in text
    out = ib.figure([("Demo", d)], tmp_path / "fig.png")
    assert out.exists() and out.stat().st_size > 3000


def _records(code: str, days: int = 2, base_m: float = 1500.0, step_s: float = 15.0, seed: int = 0) -> pd.DataFrame:
    """A synthetic clean ceilometer stream: a slowly varying base with clear gaps."""
    rng = np.random.default_rng(seed)
    n = int(days * 86400 / step_s)
    t = pd.Timestamp("2022-06-01", tz="UTC") + pd.to_timedelta(np.arange(n) * step_s, unit="s")
    cbh = base_m + 300 * np.sin(np.arange(n) / 400) + rng.normal(0, 20, n)
    cbh[(np.arange(n) // 2000) % 3 == 2] = np.nan                       # clear spells
    df = pd.DataFrame({"ceilometer": code, "time": t, "cbh1": cbh, "cbh2": np.nan, "cbh3": np.nan, "cbh4": np.nan})
    df["n_layers"] = df.cbh1.notna().astype(int)
    for flag in ("qc_rain", "qc_fog", "qc_snow", "qc_window", "qc_optics", "qc_laser_warning", "qc_error", "qc_any"):
        df[flag] = False
    return df


def test_qc_summary_counts_completeness_gaps_and_order_violations():
    df = _records("A")
    df.loc[5, "cbh2"] = 100.0                                            # a second layer below the first: a violation
    df = df.drop(index=range(100, 110))                                  # a gap of 150 s
    q = cq.qc_summary(df).iloc[0]
    assert q.days == 2 and q.records == len(df) and q.completeness < 1 and q.gaps_over_60s == 1 and q.longest_gap_s == 165
    assert q.layer_order_violations == 1 and q.duplicate_times == 0 and q.cbh1_over_max == 0 and 0 < q.cloud_present < 1


def test_agreement_falls_with_offset_and_windows_pick_a_tolerance():
    df = _records("A")
    off = cq.agreement_vs_offset(df, offsets_s=(15, 300, 3600))
    assert off.pairs.iloc[0] > 1000 and off.presence_agreement.iloc[0] > off.presence_agreement.iloc[-1]
    assert off.abs_change_median_m.iloc[0] < off.abs_change_median_m.iloc[-1] and off.within_200m.iloc[0] > 0.9
    win = cq.window_consistency(df, half_windows_s=(30, 300, 3600))
    assert win.records_per_window.iloc[0] < win.records_per_window.iloc[-1] and win.coverage.iloc[0] > 0.9
    assert win.presence_unanimous.iloc[0] >= win.presence_unanimous.iloc[-1] and (win.etage_unanimous >= 0.5).all()
    chosen = cq.choose_tolerance(win, max_drop=0.02)
    assert chosen["half_window_s"] in {30.0, 300.0, 3600.0}
    assert cq.choose_tolerance(win, max_drop=1.0)["half_window_s"] == 3600.0      # any drop allowed: the largest window


def test_cross_site_pairs_the_two_streams():
    a, b = _records("A"), _records("B", seed=1)
    b["cbh1"] = b.cbh1 + 100
    both = pd.concat([a, b], ignore_index=True)
    c = cq.cross_site(both, "A", "B")
    assert c["pairs"] == len(a) and c["presence_agreement"] == 1.0 and 50 < c["abs_difference_median_m"] < 200
    assert c["within_500m"] == 1.0 and c["etage_agreement"] > 0.9


def test_ceilometer_qc_report_builds(tmp_path):
    df = _records("A")
    qc = cq.qc_summary(df)
    off = {"A": cq.agreement_vs_offset(df, offsets_s=(15, 60))}
    win = {"A": cq.window_consistency(df, half_windows_s=(30, 120))}
    chosen = cq.choose_tolerance(win["A"])
    cross = {"pairs": 0, "presence_agreement": np.nan, "etage_agreement": np.nan, "abs_difference_median_m": np.nan,
             "within_500m": np.nan}
    text = cq.report_markdown(qc, off, win, chosen, cross, ["decided"], "fig.png")
    assert "## QC summary" in text and "| 15 s |" in text and "Chosen half-window" in text and "- decided" in text
    out = cq.figure(off, win, chosen["half_window_s"], tmp_path / "fig.png")
    assert out.exists()
