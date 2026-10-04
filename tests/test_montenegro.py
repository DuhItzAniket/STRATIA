import json

import pandas as pd

from stratia.data.montenegro import VARIABLES, load_annotations, load_items, soft_labels

HEADER = ("annotation_item_id,user_id," + ",".join(VARIABLES)) + "\n"


def _write(tmp_path):
    rows = [
        "1,68,Low,7,2,5,8,7,0",
        "1,69,Low clouds,7,5,5,8,9,0",
        "1,70,High clouds,6,2,/,5,3,",     # '/' is a code; empty CH is missing
        "2,68,Clear,0,0,9,0,0,0",
    ]
    (tmp_path / "annotations.csv").write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "annotation_items.csv").write_text(
        "id,name,item_url,item_type,item_received\n"
        "1,snapshot_a.jpg,/images/,image,2025-10-03 16:07:01\n"
        "2,snapshot_b.jpg,/images/,image,2025-10-04 09:00:00\n", encoding="utf-8")
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "1_snapshot_a.jpg").write_bytes(b"x")     # archive style (id prefix)
    (tmp_path / "images" / "snapshot_b.jpg").write_bytes(b"x")       # README style


def test_codes_are_strings_and_low_is_merged(tmp_path):
    _write(tmp_path)
    ann = load_annotations(tmp_path)
    assert ann.loc[0, "N"] == "7" and isinstance(ann.loc[0, "h"], str)
    assert set(ann.altitude_class) == {"Low clouds", "High clouds", "Clear"}
    assert ann.loc[2, "h"] == "/" and pd.isna(ann.loc[2, "CH"])


def test_soft_labels(tmp_path):
    _write(tmp_path)
    soft = soft_labels(load_annotations(tmp_path)).set_index("item_id")
    r = soft.loc[1]
    assert r.n_raters == 3
    assert json.loads(r.altitude_class_dist) == {"High clouds": round(1 / 3, 6), "Low clouds": round(2 / 3, 6)}
    assert r.altitude_class_majority == "Low clouds" and abs(r.altitude_class_agreement - 2 / 3) < 1e-9
    assert r.CH_n == 2                                   # one missing answer excluded
    assert json.loads(r.h_dist) == {"/": round(1 / 3, 6), "5": round(2 / 3, 6)}
    assert soft.loc[2].N_majority == "0"


def test_ties_are_explicit(tmp_path):
    _write(tmp_path)
    soft = soft_labels(load_annotations(tmp_path)).set_index("item_id")
    assert soft.loc[1].CM_majority == "3,7,9"            # three-way tie kept explicit


def test_image_files_resolve_both_naming_styles(tmp_path):
    _write(tmp_path)
    items = load_items(tmp_path).set_index("item_id")
    assert items.loc[1, "image_file"] == "images/1_snapshot_a.jpg"
    assert items.loc[2, "image_file"] == "images/snapshot_b.jpg"
