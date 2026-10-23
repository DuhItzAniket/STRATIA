"""Data-free tests of `stratia.train.b0268_eval` (P083 / P084 scoring against the P041 rater targets)."""

import json

import numpy as np
import pandas as pd
import pytest

from stratia.contract import GENUS_CLASSES
from stratia.train import b0268_eval as ev

D1, D2 = "2026-10-18", "2026-10-19"


def _row(item_id, day, etage=None, genus=None, oktas=None, clear=0.0, glare=0.0, cloud=1.0):
    """One soft-target row the way `stratia.labels.b0268_labels.soft_targets` writes it (JSON strings or None)."""
    return {"item_id": item_id, "day": day, "image_file": f"B0268/{item_id}.jpg",
            "etage_share": json.dumps(etage) if etage is not None else None,
            "genus_share": json.dumps(genus) if genus is not None else None,
            "oktas_dist": json.dumps(oktas) if oktas is not None else None,
            "clear_share": clear, "glare_strong_share": glare, "cloud_share": cloud}


def _four_frames() -> pd.DataFrame:
    """Four usable frames with hand-computable majorities: r1 low+mid 4 oktas, r2 high 6 oktas, r3 clear,
    r4 low with a 1/2 okta tie (-> 1) and no genus majority."""
    return pd.DataFrame([
        _row("r1", D1, {"low": 1.0, "mid": 0.5, "high": 0.0}, {"Cu": 1.0, "Ac": 0.25}, {"4": 1.0}, glare=1.0),
        _row("r2", D1, {"low": 0.0, "mid": 0.0, "high": 1.0}, {"Ci": 1.0}, {"6": 1.0}),
        _row("r3", D2, {"low": 0.0, "mid": 0.0, "high": 0.0}, None, {"0": 1.0}, clear=1.0, cloud=0.0),
        _row("r4", D2, {"low": 0.5, "mid": 0.0, "high": 0.0}, {"Cu": 0.25, "Sc": 0.25}, {"1": 0.5, "2": 0.5}, glare=0.5),
    ])


def _predictions() -> dict[str, np.ndarray]:
    """Predictions aligned with `_four_frames()`."""
    etage = np.array([[0.9, 0.6, 0.1],      # low+mid: hit
                      [0.2, 0.1, 0.8],      # high: hit
                      [0.7, 0.1, 0.1],      # low on the clear frame: miss
                      [0.6, 0.4, 0.2]])     # low: hit
    oktas = np.zeros((4, 9))
    for i, k in enumerate([5, 6, 2, 0]):    # truth 4, 6, 0, 1 -> diffs +1, 0, +2, -1
        oktas[i, k] = 1.0
    genus = np.full((4, 12), 0.01)
    gi = GENUS_CLASSES.index
    genus[0, gi("Ac")] = 0.9                # in union {Cu, Ac}, not in majority {Cu}
    genus[1, gi("Ci")] = 0.9                # in union and majority
    genus[2, gi("clear")] = 0.95            # the extra class must not count as a genus
    genus[2, gi("Cu")] = 0.3                # (no union: excluded anyway)
    genus[3, gi("St")] = 0.9                # not in union {Cu, Sc}; no majority
    return {"etage": etage, "oktas": oktas, "genus": genus,
            "cloud_fraction": np.array([0.6, 0.75, 0.1, 0.3]),
            "glare_fraction": np.array([0.2, 0.05, 0.0, 0.1]),
            "ood": np.array([0.1, 0.3, 0.9, 0.5])}


CEILING = {"etage_set": 0.9, "oktas_within1": 0.8}


# ------------------------------------------------------------------------------------------ majority_targets


def test_majority_targets_vectors_ties_unions_and_dropped_rows():
    targets = pd.DataFrame([
        _row("f1", D1, {"low": 1.0, "mid": 0.5, "high": 0.0}, {"Cu": 1.0, "Ac": 0.5, "Ci": 0.25},
             {"4": 0.5, "5": 0.25, "6": 0.25}, glare=0.75),
        _row("f2", D1, {"low": 0.0, "mid": 0.0, "high": 1.0}, {"Ci": 1.0}, {"3": 0.5, "5": 0.5}),          # okta tie
        _row("f3", D2, {"low": 0.0, "mid": 0.0, "high": 0.0}, None, {"0": 1.0}, clear=1.0, glare=0.5, cloud=0.0),
        _row("f4", D2, None, None, None),                                                                   # unusable
        _row("f5", D2, {"low": 1.0, "mid": 0.0, "high": 0.0}, {"St": 1.0}, None, cloud=1.0),               # all obscured
    ])
    m = ev.majority_targets(targets)
    assert list(m.item_id) == ["f1", "f2", "f3", "f5"]                 # f4 (no étage share) is dropped
    assert list(m.columns[:3]) == ["item_id", "day", "image_file"] and m.image_file.iloc[0] == "B0268/f1.jpg"
    r = m.set_index("item_id")

    assert r.loc["f1", "etage_majority"].tolist() == [True, True, False]        # 0.5 counts as a majority
    assert r.loc["f1", "etage_share"].tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert r.loc["f1", "oktas_majority"] == 4 and r.loc["f1", "oktas_mean"] == pytest.approx(4.75)
    assert set(r.loc["f1", "genus_union"]) == {"Cu", "Ac", "Ci"} and set(r.loc["f1", "genus_majority"]) == {"Cu", "Ac"}
    assert not r.loc["f1", "clear"] and r.loc["f1", "glare_strong"]

    assert r.loc["f2", "etage_majority"].tolist() == [False, False, True]
    assert r.loc["f2", "oktas_majority"] == 3 and r.loc["f2", "oktas_mean"] == pytest.approx(4.0)   # tie -> lower okta
    assert r.loc["f2", "genus_union"] == ["Ci"] and r.loc["f2", "genus_majority"] == ["Ci"]
    assert not r.loc["f2", "glare_strong"]

    assert r.loc["f3", "etage_majority"].tolist() == [False, False, False]
    assert r.loc["f3", "oktas_majority"] == 0 and r.loc["f3", "oktas_mean"] == 0.0
    assert r.loc["f3", "genus_union"] == [] and r.loc["f3", "genus_majority"] == []
    assert r.loc["f3", "clear"] and r.loc["f3", "glare_strong"] and r.loc["f3", "cloud_share"] == 0.0

    assert pd.isna(r.loc["f5", "oktas_majority"]) and pd.isna(r.loc["f5", "oktas_mean"])
    assert r.loc["f5", "etage_majority"].tolist() == [True, False, False] and r.loc["f5", "genus_union"] == ["St"]


def test_majority_targets_empty_when_nothing_usable():
    assert len(ev.majority_targets(pd.DataFrame([_row("x", D1), _row("y", D1)]))) == 0


# ------------------------------------------------------------------------------------------ score


def test_score_numbers_match_hand_computation():
    truth = ev.majority_targets(_four_frames())
    s = ev.score(_predictions(), truth, ceiling=CEILING, ood_threshold=0.4)
    assert s["n"] == 4.0

    # étage: 3 of 4 exact set hits; low has one false positive (the clear frame)
    assert s["etage_set_agreement"] == 0.75
    assert s["etage_low_acc"] == 0.75 and s["etage_mid_acc"] == 1.0 and s["etage_high_acc"] == 1.0
    assert s["etage_low_pred_rate"] == 0.75 and s["etage_low_true_rate"] == 0.5
    assert s["etage_mid_pred_rate"] == 0.25 and s["etage_mid_true_rate"] == 0.25
    assert s["etage_high_pred_rate"] == 0.25 and s["etage_high_true_rate"] == 0.25
    assert s["etage_f1"] == pytest.approx((0.8 + 1.0 + 1.0) / 3)       # low: p 2/3, r 1 -> 0.8

    # oktas: diffs +1, 0, +2, -1
    assert s["oktas_within1"] == 0.75 and s["oktas_mae"] == 1.0 and s["oktas_bias"] == 0.5
    assert s["oktas_pred_mean"] == 3.25 and s["oktas_true_mean"] == 2.75

    # genus top-1 over the ten genera only: union hits r1, r2 (not r4; r3 has none), majority hit r2 (not r1)
    assert s["genus_top1_in_union"] == pytest.approx(2 / 3)
    assert s["genus_top1_in_majority"] == 0.5

    # cloud fraction against the raters' mean cover (4/8, 6/8, 0, 1.5/8)
    assert s["cloud_fraction_mae_vs_cover"] == pytest.approx(0.078125)
    assert s["cloud_fraction_bias_vs_cover"] == pytest.approx(0.078125)
    assert 0.98 < s["cloud_fraction_corr_vs_cover"] <= 1.0
    assert s["cloud_fraction_on_le1_okta"] == pytest.approx(0.2) and s["n_le1_okta"] == 2.0

    # glare: strong frames r1, r4
    assert s["glare_fraction_strong_frames"] == pytest.approx(0.15)
    assert s["glare_fraction_other_frames"] == pytest.approx(0.025)

    # OOD
    assert s["ood_mean"] == pytest.approx(0.45) and s["ood_p90"] == pytest.approx(0.78)
    assert s["ood_flagged_share"] == 0.5

    # C2-style ratios against the given ceiling
    assert s["c2_etage_ratio"] == pytest.approx(0.75 / 0.9)
    assert s["c2_oktas_ratio"] == pytest.approx(0.75 / 0.8)


def test_score_omits_optional_blocks():
    truth = ev.majority_targets(_four_frames())
    pred = _predictions()
    s = ev.score({k: pred[k] for k in ("etage", "oktas", "genus")}, truth)
    for key in ("cloud_fraction_mae_vs_cover", "n_le1_okta", "glare_fraction_strong_frames", "ood_mean",
                "ood_flagged_share", "c2_etage_ratio", "c2_oktas_ratio"):
        assert key not in s
    s = ev.score(pred, truth, ceiling=CEILING)       # no threshold -> no flagged share, ratios still there
    assert "ood_mean" in s and "ood_flagged_share" not in s and "c2_oktas_ratio" in s


def test_score_without_any_okta_majority_skips_the_okta_block():
    truth = ev.majority_targets(pd.DataFrame([
        _row("a", D1, {"low": 1.0, "mid": 0.0, "high": 0.0}, {"Cu": 1.0}, None),
        _row("b", D1, {"low": 0.0, "mid": 0.0, "high": 1.0}, {"Ci": 1.0}, None),
    ]))
    pred = {"etage": np.array([[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]]), "oktas": np.full((2, 9), 1 / 9),
            "genus": np.full((2, 12), 0.1), "cloud_fraction": np.array([0.5, 0.5])}
    s = ev.score(pred, truth, ceiling=CEILING)
    assert s["etage_set_agreement"] == 1.0 and s["c2_etage_ratio"] == pytest.approx(1 / 0.9)
    for key in ("oktas_within1", "oktas_mae", "c2_oktas_ratio", "n_le1_okta", "cloud_fraction_mae_vs_cover"):
        assert key not in s


# ------------------------------------------------------------------------------------------ per rater


def _derived():
    """Two raters on three frames, as `stratia.labels.b0268_labels.derive` writes them."""
    return pd.DataFrame([
        {"item_id": "r1", "user_id": "A", "etage_set": "low|mid", "genus_set": "Ac|Cu", "oktas": 4.0, "obscured": False,
         "cloud": True, "extra": None, "glare": "strong", "usable": True},
        {"item_id": "r1", "user_id": "B", "etage_set": "low", "genus_set": "Cu", "oktas": 6.0, "obscured": False,
         "cloud": True, "extra": None, "glare": "mild", "usable": True},
        {"item_id": "r2", "user_id": "A", "etage_set": "high", "genus_set": "Ci", "oktas": 2.0, "obscured": False,
         "cloud": True, "extra": None, "glare": "none", "usable": True},
        {"item_id": "r2", "user_id": "B", "etage_set": "high", "genus_set": None, "oktas": 3.0, "obscured": False,
         "cloud": True, "extra": None, "glare": "none", "usable": True},                       # unsure genus
        {"item_id": "r3", "user_id": "A", "etage_set": "", "genus_set": "", "oktas": 0.0, "obscured": False,
         "cloud": False, "extra": "clear", "glare": "none", "usable": True},
        {"item_id": "r3", "user_id": "B", "etage_set": None, "genus_set": None, "oktas": None, "obscured": False,
         "cloud": None, "extra": None, "glare": None, "usable": False},                         # unusable for B
    ])


def test_score_against_raters_averages_over_raters_without_tie_breaking():
    raters = ev.rater_targets(_derived())
    assert set(raters) == {"A", "B"} and len(raters["A"]) == 3 and len(raters["B"]) == 2      # B's unusable row dropped
    assert raters["B"].loc["r2", "genus"] is None and raters["A"].loc["r3", "genus"] == []
    pred = {"etage": np.array([[0.9, 0.6, 0.1], [0.1, 0.1, 0.9], [0.2, 0.1, 0.1]]),
            "oktas": np.eye(9)[[5, 2, 0]], "genus": np.full((3, 12), 0.01)}
    pred["genus"][0, GENUS_CLASSES.index("Ac")] = 0.9
    pred["genus"][1, GENUS_CLASSES.index("Cs")] = 0.9
    s = ev.score_against_raters(pred, ["r1", "r2", "r3"], raters, ceiling={"etage_set": 0.9, "oktas_within1": 0.8})
    # A: étage sets hit on r1 (low+mid), r2, r3 -> 1.0; B: r1 miss (B says low only), r2 hit -> 0.5
    assert s["etage_set_agreement_A"] == 1.0 and s["etage_set_agreement_B"] == 0.5 and s["etage_set_agreement"] == 0.75
    # oktas: A diffs +1, 0, 0 -> within1 1.0, mae 1/3, bias +1/3; B diffs -1, -1 -> within1 1.0, mae 1, bias -1
    assert s["oktas_within1"] == 1.0 and s["oktas_mae"] == pytest.approx((1 / 3 + 1.0) / 2)
    assert s["oktas_bias"] == pytest.approx((1 / 3 - 1.0) / 2)
    # genus: A scored on r1 (Ac in {Ac, Cu}: hit) and r2 (Cs not in {Ci}: miss) -> 0.5; B only on r1 (Ac not in {Cu}) -> 0
    assert s["genus_top1_in_set_A"] == 0.5 and s["genus_top1_in_set_B"] == 0.0 and s["genus_top1_in_set"] == 0.25
    assert s["n_raters"] == 2.0 and s["c2_etage_ratio"] == pytest.approx(0.75 / 0.9)
    assert s["c2_oktas_ratio"] == pytest.approx(1 / 0.8)
    df = ev.per_day_against_raters(pred, ["r1", "r2", "r3"], ["d1", "d1", "d2"], raters).set_index("day")
    assert list(df.index) == ["all", "d1", "d2"] and df.loc["d2", "n_raters"] == 1.0
    assert df.loc["d1", "etage_set_agreement"] == 0.75


# ------------------------------------------------------------------------------------------ per_day


def test_per_day_has_an_all_row_and_one_row_per_day():
    truth = ev.majority_targets(_four_frames())
    df = ev.per_day(_predictions(), truth, ceiling=CEILING, ood_threshold=0.4).set_index("day")
    assert list(df.index) == ["all", D1, D2]
    assert df.n.tolist() == [4.0, 2.0, 2.0]
    assert df.loc["all", "etage_set_agreement"] == 0.75
    assert df.loc[D1, "etage_set_agreement"] == 1.0 and df.loc[D2, "etage_set_agreement"] == 0.5
    assert df.loc[D1, "oktas_within1"] == 1.0 and df.loc[D2, "oktas_within1"] == 0.5
    assert df.loc[D1, "oktas_bias"] == 0.5 and df.loc[D2, "oktas_bias"] == 0.5
    assert df.loc[D2, "n_le1_okta"] == 2.0 and pd.isna(df.loc[D1, "n_le1_okta"])       # no <= 1-okta frame on day 1
    assert df.loc[D1, "ood_flagged_share"] == 0.0 and df.loc[D2, "ood_flagged_share"] == 1.0
    assert df.loc[D1, "c2_etage_ratio"] == pytest.approx(1 / 0.9) and df.loc["all", "c2_etage_ratio"] == pytest.approx(0.75 / 0.9)
    assert df.loc[D1, "glare_fraction_strong_frames"] == pytest.approx(0.2)
    assert df.loc[D2, "glare_fraction_strong_frames"] == pytest.approx(0.1)
    # Pins the module's `both.sum() > 2` gate in `score` (b0268_eval.py): mae / bias / corr vs cover are all skipped
    # when a day has fewer than three finite pairs, although mae and bias alone would be defined from one pair.
    assert pd.isna(df.loc[D1, "cloud_fraction_mae_vs_cover"])
    assert pd.isna(df.loc[D1, "cloud_fraction_corr_vs_cover"])
