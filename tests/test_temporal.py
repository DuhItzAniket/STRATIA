import numpy as np
import pandas as pd

from stratia.data import temporal as tp


def _drifting_embeddings(n: int, step: float, seed: int = 0) -> np.ndarray:
    """Unit vectors that wander slowly: neighbours in time are alike, far frames are not."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, step, size=(n, 32)), axis=0) + rng.normal(0, 1, size=32)
    return walk / np.linalg.norm(walk, axis=1, keepdims=True)


def test_pairs_respect_the_gap_bins_within_one_series():
    t = np.arange(0, 3 * tp.DAY, 30.0)                       # three days at 30 s
    edges = np.array([30.0, 60.0, 3600.0, tp.DAY, np.inf])
    pairs = tp.sample_pairs_by_gap(t, edges, per_bin=200, rng=np.random.default_rng(1))
    assert set(pairs.bin) == {0, 1, 2, 3} and (pairs.groupby("bin").size() <= 200).all()
    assert (pairs.j > pairs.i).all()
    for b, part in pairs.groupby("bin"):
        assert (part.gap_s >= edges[b]).all() and (part.gap_s < edges[b + 1]).all()
    assert (pairs[pairs.bin == 0].gap_s == 30).all()           # the adjacent-frame bin holds only adjacent frames


def test_cross_series_pairs_use_the_absolute_gap_and_a_same_moment_bin():
    t_a = np.arange(0, 7200, 30.0)
    t_b = t_a + 5.0                                            # the other camera fires 5 s later
    edges = np.array([0.0, 30.0, 600.0, np.inf])
    pairs = tp.sample_pairs_by_gap(t_a, edges, per_bin=100, rng=np.random.default_rng(2), t_b=t_b)
    assert set(pairs.bin) == {0, 1, 2}
    same_moment = pairs[pairs.bin == 0]
    assert (np.isclose(same_moment.gap_s, 5.0) | np.isclose(same_moment.gap_s, 25.0)).all()
    for b, part in pairs.groupby("bin"):
        assert (part.gap_s >= edges[b]).all() and (part.gap_s <= edges[b + 1]).all()


def test_same_hour_other_days_pairs_are_on_different_days_at_the_same_time_of_day():
    t = np.concatenate([d * tp.DAY + np.arange(6 * 3600, 18 * 3600, 600.0) for d in range(5)])
    pairs = tp.sample_same_hour_other_days(t, per=300, rng=np.random.default_rng(3), window_s=1800.0)
    assert len(pairs) == 300 and (pairs.gap_s >= tp.DAY - 1800).all()
    tod = lambda x: np.mod(x, tp.DAY)  # noqa: E731
    assert (np.abs(tod(t[pairs.i]) - tod(t[pairs.j])) < 1800).all()
    assert tp.sample_same_hour_other_days(np.arange(0, 3600, 60.0), 10, np.random.default_rng(0)).empty   # one day only


def test_curve_decays_with_gap_and_the_decision_finds_where_it_reaches_the_baseline():
    n = 4000
    t = np.arange(n) * 30.0                                    # 33 hours at 30 s
    emb = _drifting_embeddings(n, step=0.03)
    ph = np.random.default_rng(0).integers(0, 2**63, size=n, dtype=np.int64).astype(np.uint64)
    edges = tp.DEFAULT_EDGES[:12]                              # 30 s .. 17 h
    pairs = tp.pair_measures(tp.sample_pairs_by_gap(t, edges, 400, np.random.default_rng(4)), emb, ph)
    assert pairs.cosine.between(-1.0001, 1.0001).all() and pairs.phash.between(0, 64).all()
    c = tp.curve(pairs, edges, min_pairs=50)
    assert len(c) >= 8 and c.cos_mean.iloc[0] > 0.99 and c.cos_mean.iloc[0] > c.cos_mean.iloc[-1]
    assert c.same_scene_frac.iloc[0] >= c.same_scene_frac.iloc[-1]
    far = c.cos_mean.iloc[-1]
    d = tp.decide_gap(c, base_cos=far, base_same_scene=c.same_scene_frac.iloc[-1], excess_tol=0.1)
    assert np.isfinite(d["gap_s"]) and d["gap_s"] in set(c.gap_hi)
    assert d["excess"].iloc[0] == 1.0 and d["excess"].iloc[-1] <= 0.1
    never = tp.decide_gap(c, base_cos=-1.0, base_same_scene=0.0)         # an unreachable baseline: no gap
    assert np.isnan(never["gap_s"]) and tp.fmt_gap(never["gap_s"]) == "none within the data"


def test_times_from_filenames_and_day_structure():
    files = pd.Series(["Almeria/images/asi_001_170328164030.jpg", "Almeria/test/20230126135101_00160.jpg", "no_time.jpg"])
    t = tp.times_from_filenames(files)
    assert t.iloc[0] == pd.Timestamp("2017-03-28 16:40:30") and t.iloc[1] == pd.Timestamp("2023-01-26 13:51:01")
    assert pd.isna(t.iloc[2])
    stamps = ["2022-04-01 05:00:00", "2022-04-01 05:00:30", "2022-04-02 05:00:00"]
    s = tp.day_structure(pd.Series(pd.to_datetime(stamps, utc=True)))
    assert s["days"] == 2 and s["frames"] == 3 and s["frames_per_day_median"] == 1.5 and s["first"] == "2022-04-01"
    assert tp.fmt_gap(30) == "30 s" and tp.fmt_gap(240) == "4 min"
    assert tp.fmt_gap(7200) == "2.0 h" and tp.fmt_gap(3 * tp.DAY) == "3.0 d"


def test_report_and_figure_are_produced(tmp_path):
    c = pd.DataFrame({"bin": [0, 1], "gap_lo": [30.0, 60.0], "gap_hi": [60.0, 120.0], "n": [100, 100],
                      "gap_median_s": [30.0, 80.0], "cos_mean": [0.99, 0.95], "cos_median": [0.99, 0.95],
                      "cos_p10": [0.97, 0.9], "same_scene_frac": [0.9, 0.3], "phash_mean": [2.0, 8.0]})
    b = pd.DataFrame([{"baseline": "different_day", "n": 50, "cos_mean": 0.9, "cos_p10": 0.8, "same_scene_frac": 0.05,
                       "phash_mean": 12.0},
                      {"baseline": "same_hour_other_day", "n": 50, "cos_mean": 0.92, "cos_p10": 0.85, "same_scene_frac": 0.1,
                       "phash_mean": 10.0}])
    d = tp.decide_gap(c, 0.9, 0.05)
    results = {"cam": {"curve": c, "baselines": b, "decision": d}}
    structures = {"cam": {"frames": 200, "days": 2, "frames_per_day_median": 100.0, "step_median_s": 30.0,
                          "first": "a", "last": "b"}}
    text = tp.report_markdown(results, structures, ["decided"], "fig.png")
    assert "| 30 s – 60 s | 100 | 0.990 |" in text and "baseline: different day" in text and "- decided" in text
    out = tp.plot_curves(results, tmp_path / "fig.png")
    assert out.exists() and out.stat().st_size > 5000
