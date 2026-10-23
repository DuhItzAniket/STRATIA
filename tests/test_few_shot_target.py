import pandas as pd

from stratia.data import splits as sp
from stratia.labels import mapping as mp

CONFIG = {"seed": 0, "few_shot_target": {"dataset": "b0268", "unit": "day", "shots_days": [0, 1, 2, 3], "repeats": 2,
                                         "test_days": 1, "min_frames_per_test_day": 3}}


def _manifest():
    rows = []
    for day, n in (("2026-10-18", 3), ("2026-10-19", 3), ("2026-10-20", 4), ("2026-10-21", 2), ("2026-09-27", 2)):
        for i in range(n):
            rows.append({"sample_id": f"b0268:{day}/{i}", "dataset": "b0268", "utc": pd.Timestamp(f"{day} 10:0{i}", tz="UTC")})
    rows.append({"sample_id": "ccsn:x", "dataset": "ccsn", "utc": pd.NaT})
    return pd.DataFrame(rows)


def test_few_shot_draws_lock_the_last_full_day_and_hold_the_rest_out():
    m = _manifest()
    labelled = pd.Series([("09-27" not in s) and d == "b0268" for s, d in zip(m.sample_id, m.dataset, strict=True)],
                         index=m.sample_id.values)
    few = sp.few_shot_target(m, labelled, CONFIG)
    assert set(few.day) == {"2026-10-18", "2026-10-19", "2026-10-20", "2026-10-21"}      # the unlabelled day is absent
    assert set(few[few.role == "test"].day) == {"2026-10-20"}                              # last day with >= 3 frames
    assert sorted(few.shots.unique()) == [0, 1, 2]                                         # k = 3 leaves nothing held out
    k0 = few[few.draw == "k0_d0"]
    assert (k0[k0.day != "2026-10-20"].role == "val").all()
    k1 = few[few.shots == 1]
    assert k1.draw.nunique() == 2                                                          # repeats = 2 of 3 combinations
    for _, part in k1.groupby("draw"):
        assert part[part.role == "train"].day.nunique() == 1 and part[part.role == "val"].day.nunique() == 2
    assert sp.verify_few_shot(few) == []
    hashes = sp.split_hashes(pd.DataFrame(columns=["source", "split", "sample_id"]),
                             pd.DataFrame(columns=["fold", "role", "sample_id"]),
                             pd.DataFrame(columns=["variant", "role", "sample_id"]), few)
    assert hashes["few_shot_target/k0_d0/test"]["n"] == 4 and "few_shot_target/k1_d0/train" in hashes
    broken = few.copy()
    broken.loc[(broken.draw == "k1_d0") & (broken.role == "val"), "role"] = "train"
    assert any("train days" in p for p in sp.verify_few_shot(broken))
    # the target's in-domain split mirrors the protocol; other sources keep their assignment; LODO never trains on it
    in_domain = pd.DataFrame({"sample_id": m.sample_id, "source": m.dataset, "split": "val",
                              "unit": range(len(m)), "stratum": "all", "test_ok": True})
    aligned = sp.align_target_with_few_shot(in_domain, few, CONFIG)
    b = aligned[aligned.source == "b0268"]
    assert set(b[b.sample_id.str.contains("2026-10-20")].split) == {"test"}
    assert set(b[~b.sample_id.str.contains("2026-10-20")].split) == {"train"}       # adaptation days and the unlabelled day
    assert (aligned[aligned.source == "ccsn"].split == "val").all()
    lodo = sp.lodo_folds(aligned, {**CONFIG, "version": 2, "lodo": {"folds": ["ccsn"]}})
    assert not lodo[lodo.role == "train"].sample_id.str.startswith("b0268").any()


def test_b0268_majority_rows_map_to_labels():
    row = pd.Series({"item_id": "f", "etage_set": ["low", "mid"], "genus_set": ["Ac", "Cu"], "cloud_present": True,
                     "extra": None})
    out = mp.map_b0268_row(row)
    assert out["genus_set"] == ["Ac", "Cu"] and out["etage_set"] == ["low", "mid"] and out["cloud_present"] is True
    assert out["excluded_reason"] is None and out["genus_alternatives"] is False
    clear = mp.map_b0268_row(pd.Series({"item_id": "g", "etage_set": [], "genus_set": None, "cloud_present": False,
                                        "extra": "clear"}))
    assert clear["extra"] == "clear" and clear["etage_set"] == [] and clear["genus_set"] == []     # known empty, as elsewhere
    assert clear["cloud_present"] is False and clear["excluded_reason"] is None
    none = mp.map_b0268_row(pd.Series({"item_id": "h", "etage_set": ["low"], "genus_set": None, "cloud_present": True,
                                       "extra": None}))
    assert none["genus_set"] is None and "no genus" in none["excluded_reason"]
