import json

import pandas as pd

from stratia.labels import mapping as mp
from stratia.labels.ontology import load_ontology

ONT = load_ontology()


def test_class_mapping_handles_single_merged_clear_and_unknown_classes():
    assert mp.map_class(ONT, "ccsn", "Cu") == {"genus_set": ["Cu"], "genus_alternatives": False, "extra": None,
                                                "etage_set": ["low"], "cloud_present": True, "excluded_reason": None}
    ct = mp.map_class(ONT, "ccsn", "Ct")
    assert ct["genus_set"] == [] and ct["extra"] == "contrail" and ct["cloud_present"] is True
    merged = mp.map_class(ONT, "mgcd", "cirrus")
    assert merged["genus_set"] == ["Ci", "Cs"] and merged["genus_alternatives"] and merged["etage_set"] == ["high"]
    two_etages = mp.map_class(ONT, "mgcd", "stratocumulus")
    assert two_etages["etage_set"] is None                      # Sc/St are low, As is mid: alternatives span two étages
    clear = mp.map_class(ONT, "mgcd", "clearsky")
    assert clear["genus_set"] == [] and clear["extra"] == "clear" and clear["cloud_present"] is False
    mixed = mp.map_class(ONT, "mgcd", "mixed")
    assert mixed["genus_set"] is None and mixed["cloud_present"] is True and "not named" in mixed["excluded_reason"]
    patch = mp.map_class(ONT, "swimcat", "E-veil")
    assert patch["genus_set"] is None and patch["cloud_present"] is True


def test_montenegro_mapping_uses_altitude_classes_and_majority_codes():
    r = mp.map_montenegro(ONT, "Low clouds,High clouds", json.dumps({"8": 0.6, "5": 0.4}), json.dumps({"0": 1.0}),
                          json.dumps({"9": 0.7, "/": 0.3}))
    assert r["genus_set"] == ["Cc", "Cu", "Sc"] and r["etage_set"] == ["high", "low"] and r["cloud_present"] is True
    clear = mp.map_montenegro(ONT, "Clear", json.dumps({"0": 1.0}), json.dumps({"0": 1.0}), json.dumps({"0": 1.0}))
    assert clear["genus_set"] == [] and clear["extra"] == "clear" and clear["cloud_present"] is False
    unseen = mp.map_montenegro(ONT, "Low clouds", json.dumps({"/": 1.0}), None, None)
    assert unseen["genus_set"] is None and unseen["etage_set"] == ["low"] and "not visible" in unseen["excluded_reason"]


def test_mask_rows_and_full_mapping_with_a_conflict_merge():
    m = pd.DataFrame({
        "sample_id": ["ccsn:a", "ccsn:b", "ccsn:c", "mgcd:d", "almeria:e", "swimseg:f", "eye2sky:g", "montenegro:h"],
        "dataset": ["ccsn", "ccsn", "ccsn", "mgcd", "almeria", "swimseg", "eye2sky", "montenegro"],
        "image_file": ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.png", "g.jpg", "h.jpg"],
        "source_label": ["Cc", "Cs", "Cu", "cumulus", None, None, None, "Middle clouds"],
        "cl_dist": [None] * 7 + [json.dumps({"0": 1.0})], "cm_dist": [None] * 7 + [json.dumps({"3": 1.0})],
        "ch_dist": [None] * 7 + [json.dumps({"0": 1.0})]})
    frac = pd.DataFrame({"dataset": ["almeria", "swimseg"], "image_file": ["e.jpg", "f.png"], "cloud_fraction": [0.4, 0.0],
                         "low": [0.3, float("nan")], "mid": [0.0, float("nan")], "high": [0.1, float("nan")]})
    conflicts = pd.DataFrame({"i": [0, 1], "j": [1, 2], "kind": ["exact", "same_scene"], "conflict": [True, True],
                              "dataset_i": ["ccsn", "ccsn"]})
    labels = mp.apply_mapping(m, ONT, frac, conflicts).set_index("sample_id")
    assert json.loads(labels.loc["ccsn:a", "genus_set"]) == ["Cc", "Cs"] and labels.loc["ccsn:a", "label_conflict"]
    assert labels.loc["ccsn:b", "label_source"] == "conflict merge" and labels.loc["ccsn:b", "genus_alternatives"]
    assert json.loads(labels.loc["ccsn:c", "genus_set"]) == ["Cu"]
    assert not labels.loc["ccsn:c", "label_conflict"]                        # same scene only: untouched
    assert json.loads(labels.loc["almeria:e", "etage_set"]) == ["high", "low"] and labels.loc["almeria:e", "cloud_present"]
    assert labels.loc["swimseg:f", "extra"] == "clear" and json.loads(labels.loc["swimseg:f", "etage_set"]) == []
    assert labels.loc["eye2sky:g", "cloud_present"] is None or pd.isna(labels.loc["eye2sky:g", "cloud_present"])
    mon = labels.loc["montenegro:h"]
    assert json.loads(mon.genus_set) == ["Ac"] and json.loads(mon.etage_set) == ["mid"]
    s = mp.summary(labels.reset_index()).set_index("dataset")
    assert s.loc["ccsn", "conflicts_merged"] == 2 and s.loc["ccsn", "with_genus_set"] == 3
    assert s.loc["eye2sky", "no_label"] == 1
    reasons = labels.excluded_reason.dropna().value_counts()
    text = mp.report_markdown(mp.mapping_table(ONT), s.reset_index(), reasons, ["decided"])
    assert "| mgcd | mixed | cloud, genus unknown |" in text and "| ccsn | 3 |" in text and "- decided" in text
