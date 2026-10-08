import json

import numpy as np
import pandas as pd

from stratia.labels import agreement as ag
from stratia.labels.ontology import load_ontology


def test_krippendorff_alpha_on_a_hand_computed_example():
    # two raters, four units: (a,a) (b,b) (a,b) (b,b): D_o = 0.25, D_e = 30/56, alpha = 0.5333
    t = pd.DataFrame({"r1": ["a", "b", "a", "b"], "r2": ["a", "b", "b", "b"]})
    assert abs(ag.krippendorff_alpha(t, "nominal") - (1 - 0.25 / (30 / 56))) < 1e-9
    binary = pd.DataFrame({"r1": [1, 2, 1, 2], "r2": [1, 2, 2, 2]})
    assert abs(ag.krippendorff_alpha(binary, "interval") - ag.krippendorff_alpha(t, "nominal")) < 1e-9
    assert abs(ag.krippendorff_alpha(binary, "ordinal") - ag.krippendorff_alpha(t, "nominal")) < 1e-9
    perfect = pd.DataFrame({"r1": ["a", "b", "c"], "r2": ["a", "b", "c"], "r3": ["a", None, "c"]})
    assert ag.krippendorff_alpha(perfect) == 1.0
    assert np.isnan(ag.krippendorff_alpha(pd.DataFrame({"r1": ["a", "b"], "r2": [None, None]})))   # no unit with 2 values
    near = pd.DataFrame({"r1": [1, 5, 3, 7], "r2": [2, 5, 3, 8]})
    far = pd.DataFrame({"r1": [1, 5, 3, 7], "r2": [8, 5, 3, 1]})
    assert ag.krippendorff_alpha(near, "interval") > ag.krippendorff_alpha(far, "interval")


def test_pairwise_and_leave_one_out():
    t = pd.DataFrame({"a": ["x", "x", "y"], "b": ["x", "y", "y"], "c": ["x", "x", None]})
    assert abs(ag.pairwise_agreement(t) - np.mean([1.0, 1 / 3, 1.0])) < 1e-9
    loo = ag.leave_one_out(t, "nominal", min_others=2).set_index("rater")
    assert loo.loc["a", "units"] == 2 and loo.loc["a", "agreement"] == 1.0          # unit 3 has only one other
    assert loo.loc["b", "agreement"] == 0.5 and loo.loc["all", "units"] == 6
    num = pd.DataFrame({"a": [1, 5], "b": [2, 5], "c": [3, 9]})
    assert ag.leave_one_out(num, "numeric", tolerance=1.0).set_index("rater").loc["c", "agreement"] == 0.5
    assert abs(ag.pairwise_agreement(num, tolerance=1.0) - np.mean([2 / 3, 1 / 3])) < 1e-9


def test_derive_and_soft_targets_follow_the_ontology():
    ont = load_ontology()
    ann = pd.DataFrame({"item_id": [1, 1, 1, 2], "user_id": [10, 11, 12, 10],
                        "altitude_class": ["Low clouds", "Low clouds,High clouds", "Clear", "Middle clouds"],
                        "N": ["7", "8", "0", "9"], "Nh": ["2", "5", "0", "1"], "h": ["5", "6", "/", "6"],
                        "CL": ["8", "5", "0", "0"], "CM": ["0", "/", "0", "3"], "CH": ["0", "9", "0", "0"]})
    d = ag.derive(ann, ont)
    assert d.etage_set.tolist() == ["low", "high|low", "", "mid"] and d.genus_set.tolist() == ["Cu|Sc", "Cc|Sc", "", "Ac"]
    assert d.oktas.tolist() == [7, 8, 0, None] and d.obscured.tolist() == [False, False, False, True]
    assert d.h_band.tolist() == ["5", "6", None, "6"] and d.cloud.tolist() == [True, True, False, True]
    s = ag.soft_targets(d).set_index("item_id")
    e1 = json.loads(s.loc[1, "etage_share"])
    assert abs(e1["low"] - 2 / 3) < 1e-9 and abs(e1["high"] - 1 / 3) < 1e-9 and e1["mid"] == 0
    g1 = json.loads(s.loc[1, "genus_share"])
    assert abs(g1["Sc"] - 2 / 3) < 1e-9 and abs(g1["Cu"] - 1 / 3) < 1e-9
    assert json.loads(s.loc[1, "oktas_dist"]) == {"0": 1 / 3, "7": 1 / 3, "8": 1 / 3} and s.loc[2, "obscured_share"] == 1.0
    assert s.loc[2, "oktas_dist"] is None and json.loads(s.loc[2, "h_band_dist"]) == {"6": 1.0}
    c = ag.ceiling_table(ann, d).set_index("quantity")
    assert "étage set" in c.index and "genus Sc present" in c.index and c.loc["raw N (oktas 0-8)", "level"] == "interval"
    text = ag.report_markdown(c.reset_index(), pd.DataFrame([{"rater": "all", "units_etage": 1, "etage": 1.0, "units_n": 1, "n": 1.0}]),
                              2, 4, ["decided"])
    assert "| étage set |" in text and "- decided" in text
