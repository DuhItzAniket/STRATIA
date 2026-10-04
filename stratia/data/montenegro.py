"""Montenegro multi-annotator sky-camera dataset (P016).

Follows the dataset README (sections 4 and 7):
* every coded column is read as a string ('/' is a valid "cannot be determined" code);
* empty cells mean "not answered" and become missing (NaN), distinct from '/';
* altitude class: 'Low' and 'Low clouds' are the same option and are merged into 'Low clouds';
* timestamps are local time, stated as UTC+1 (CET); daylight saving in October 2025 is not confirmed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

VARIABLES = {
    "Cloud classification by altitude (dominant type recorded)": "altitude_class",
    "Total cloud cover in oktas": "N",
    "Amount of low cloud covering the sky": "Nh",
    "Height of base of lowest cloud": "h",
    "CL - Low cloud type": "CL",
    "CM - Middle cloud type": "CM",
    "CH - High cloud type": "CH",
}
UTC_OFFSET_HOURS_STATED = 1


def image_file(root: str | Path, item_id: int, name: str) -> str:
    """Relative image path. The distributed archive prefixes names with the item id
    (`17487_snapshot_...jpg`) although the CSVs list `snapshot_...jpg`; both forms are accepted."""
    images = Path(root) / "images"
    for candidate in (f"{item_id}_{name}", name):
        if (images / candidate).exists():
            return f"images/{candidate}"
    return f"images/{item_id}_{name}"


def load_items(root: str | Path) -> pd.DataFrame:
    items = pd.read_csv(Path(root) / "annotation_items.csv", dtype={"id": "int64", "name": str})
    items = items.rename(columns={"id": "item_id", "name": "file"})
    items["image_file"] = [image_file(root, i, n) for i, n in zip(items.item_id, items.file, strict=True)]
    items["local_time"] = pd.to_datetime(items["item_received"])
    items["day"] = items["local_time"].dt.date
    return items[["item_id", "file", "image_file", "local_time", "day"]]


def load_annotations(root: str | Path) -> pd.DataFrame:
    ann = pd.read_csv(Path(root) / "annotations.csv", dtype=str, keep_default_na=False)
    ann = ann.rename(columns={"annotation_item_id": "item_id", **VARIABLES})
    ann["item_id"] = ann["item_id"].astype("int64")
    ann["user_id"] = ann["user_id"].astype("int64")
    for col in VARIABLES.values():
        ann[col] = ann[col].str.strip().replace({"": pd.NA})
    ann["altitude_class"] = ann["altitude_class"].replace({"Low": "Low clouds"})
    return ann


def soft_labels(ann: pd.DataFrame) -> pd.DataFrame:
    """One row per image: number of raters, and per variable the code distribution, majority code and its share."""
    rows = []
    for item_id, g in ann.groupby("item_id", sort=True):
        rec = {"item_id": item_id, "n_raters": len(g), "raters": ",".join(map(str, sorted(g.user_id)))}
        for col in VARIABLES.values():
            vals = g[col].dropna()
            dist = vals.value_counts(normalize=True).sort_index()
            rec[f"{col}_n"] = int(vals.size)
            rec[f"{col}_dist"] = json.dumps({k: round(float(v), 6) for k, v in dist.items()})
            if vals.size:
                top = dist.max()
                rec[f"{col}_majority"] = ",".join(sorted(dist[dist == top].index))   # ties kept explicit
                rec[f"{col}_agreement"] = float(top)
            else:
                rec[f"{col}_majority"], rec[f"{col}_agreement"] = None, float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)
