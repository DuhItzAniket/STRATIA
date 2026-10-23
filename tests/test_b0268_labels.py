from pathlib import Path

import pandas as pd
import pytest

from stratia.labels import b0268_labels as bl

FIXTURES = Path(__file__).parent / "fixtures" / "b0268_labels"


def _long(rows: list[dict]) -> pd.DataFrame:
    base = {"usable": "yes", "oktas": "3", "low_cloud": "none", "mid_cloud": "none", "high_cloud": "none",
            "lowest_base_guess": "none", "sun_glare": "none", "obstruction": "none", "notes": ""}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_derivation_follows_the_sheet_rules():
    long = _long([
        {"item_id": "f1", "user_id": "A", "oktas": "5", "low_cloud": "Cu+Sc", "mid_cloud": "Ac", "high_cloud": "Ci+Cs",
         "lowest_base_guess": "low", "sun_glare": "mild"},
        {"item_id": "f2", "user_id": "A", "oktas": "0"},                                             # clear
        {"item_id": "f3", "user_id": "A", "oktas": "2", "high_cloud": "contrail"},                   # extra class
        {"item_id": "f4", "user_id": "A", "oktas": "4", "low_cloud": "unsure", "mid_cloud": "As"},   # genus unknown
        {"item_id": "f5", "user_id": "A", "oktas": "9", "low_cloud": "St"},                          # obscured
        {"item_id": "f6", "user_id": "A", "usable": "no", "oktas": "", "low_cloud": "", "mid_cloud": "", "high_cloud": "",
         "lowest_base_guess": "", "sun_glare": "", "obstruction": "raindrops"},
    ])
    d = bl.derive(long).set_index("item_id")
    assert d.loc["f1", "etage_set"] == "low|mid|high" and d.loc["f1", "genus_set"] == "Ac|Ci|Cs|Cu|Sc"
    assert d.loc["f1", "oktas"] == 5 and bool(d.loc["f1", "cloud"]) and d.loc["f1", "glare"] == "mild"
    assert d.loc["f2", "extra"] == "clear" and d.loc["f2", "etage_set"] == "" and not bool(d.loc["f2", "cloud"])
    assert d.loc["f3", "extra"] == "contrail" and d.loc["f3", "etage_set"] == "high" and d.loc["f3", "genus_set"] == ""
    assert pd.isna(d.loc["f4", "genus_set"]) and d.loc["f4", "etage_set"] == "low|mid"   # unsure: étage yes, genus unknown
    assert d.loc["f5", "obscured"] and pd.isna(d.loc["f5", "oktas"]) and pd.isna(d.loc["f5", "cloud"])
    assert not d.loc["f6", "usable"] and pd.isna(d.loc["f6", "etage_set"]) and d.loc["f6", "obstruction"] == "raindrops"


def test_soft_targets_and_majority_sets_average_the_raters():
    long = _long([
        {"item_id": "f1", "user_id": "A", "oktas": "5", "low_cloud": "Cu", "mid_cloud": "Ac", "sun_glare": "strong"},
        {"item_id": "f1", "user_id": "B", "oktas": "6", "low_cloud": "none", "mid_cloud": "Ac", "sun_glare": "mild"},
        {"item_id": "f2", "user_id": "A", "oktas": "0"},
        {"item_id": "f2", "user_id": "B", "oktas": "0"},
    ])
    frames = pd.DataFrame({"frame_id": ["f1", "f2"], "day": ["d1", "d1"], "file": ["a.jpg", "b.jpg"]})
    t = bl.soft_targets(bl.derive(long), frames).set_index("item_id")
    assert t.loc["f1", "n_annotators"] == 2 and t.loc["f1", "glare_strong_share"] == 0.5 and t.loc["f1", "glare_any_share"] == 1.0
    assert t.loc["f1", "image_file"] == "B0268/a.jpg" and t.loc["f1", "day"] == "d1"
    etage = pd.read_json(pd.io.common.StringIO(t.loc["f1", "etage_share"]), typ="series")
    genus = pd.read_json(pd.io.common.StringIO(t.loc["f1", "genus_share"]), typ="series")
    assert etage["low"] == 0.5 and etage["mid"] == 1.0 and etage["high"] == 0.0
    assert genus["Cu"] == 0.5 and genus["Ac"] == 1.0
    assert t.loc["f2", "clear_share"] == 1.0 and t.loc["f2", "cloud_share"] == 0.0
    hard = bl.majority_sets(t.reset_index()).set_index("item_id")
    assert hard.loc["f1", "etage_set"] == ["low", "mid"] and hard.loc["f1", "genus_set"] == ["Ac", "Cu"]
    assert bool(hard.loc["f1", "cloud_present"])
    assert hard.loc["f2", "extra"] == "clear" and hard.loc["f2", "etage_set"] == [] and not bool(hard.loc["f2", "cloud_present"])


def test_agreement_table_two_raters():
    long = _long([
        {"item_id": "f1", "user_id": "A", "oktas": "5", "low_cloud": "Cu"},
        {"item_id": "f1", "user_id": "B", "oktas": "6", "low_cloud": "Cu"},
        {"item_id": "f2", "user_id": "A", "oktas": "2", "high_cloud": "Ci"},
        {"item_id": "f2", "user_id": "B", "oktas": "7", "low_cloud": "Sc"},
    ])
    d = bl.derive(long)
    table = bl.agreement_table(long, d).set_index("quantity")
    assert table.loc["oktas 0-8", "pairwise"] == 0.5 and table.loc["oktas 0-8, exact", "pairwise"] == 0.0
    assert table.loc["étage set", "pairwise"] == 0.5 and table.loc["étage set", "leave_one_out"] == 0.5
    assert table.loc["raw mid_cloud", "pairwise"] == 1.0
    per = bl.per_rater(d).set_index("rater")
    assert per.loc["A", "etage"] == 0.5 and per.loc["all", "units_n"] == 4


def test_fixture_sheets_load_and_validate():
    frames = bl.load_frames(FIXTURES)
    assert len(frames) == 393 and frames.day.nunique() == 4
    long = bl.load_raters(FIXTURES, ("A", "B"))
    assert len(long) == 786 and set(long.user_id) == {"A", "B"}
    d = bl.derive(long)
    assert d.usable.all() and d.oktas.notna().all()
    t = bl.soft_targets(d, frames)
    assert len(t) == 393 and t.n_annotators.eq(2).all() and t.image_file.str.startswith("B0268/BLR01/").all()
    table = bl.agreement_table(long, d).set_index("quantity")
    assert 0.0 < table.loc["étage set", "alpha"] <= 1.0


def test_sheet_validation_rejects_bad_values(tmp_path):
    frames = pd.DataFrame({"frame_id": ["f1"], "day": ["d"], "file": ["x.jpg"]})
    good = "idx,usable,oktas,low_cloud,mid_cloud,high_cloud,lowest_base_guess,sun_glare,obstruction,notes\n"
    (tmp_path / "ok.csv").write_text(good + "1,yes,3,Cu,none,none,low,mild,none,\n", encoding="utf-8")
    assert len(bl.load_sheet(tmp_path / "ok.csv", "A", frames)) == 1
    (tmp_path / "bad.csv").write_text(good + "1,yes,3,Cumulus,none,none,low,mild,none,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="vocabulary"):
        bl.load_sheet(tmp_path / "bad.csv", "A", frames)
    (tmp_path / "half.csv").write_text(good + "1,no,3,,,,,,none,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unusable row"):
        bl.load_sheet(tmp_path / "half.csv", "A", frames)
    obscured = bl.derive(_long([{"item_id": "f", "user_id": "A", "oktas": "9"}]))
    assert pd.isna(obscured.oktas.iloc[0]) and bool(obscured.obscured.iloc[0])
