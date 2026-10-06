"""Exact duplicates within and across datasets (P024).

Two hashes per image:
    file_sha256    of the file's bytes: the same file under two names
    pixel_sha256   of the decoded 8-bit RGB pixels: the same picture re-saved (metadata stripped, PNG re-encoded,
                   different JPEG container) - still an exact duplicate for a model, invisible to the file hash
Images that share a hash form a duplicate group. A group that spans datasets is leakage waiting to happen if its
members land in different splits; the policy (docs/data/duplicates_report.md) is that all members of a group share
one group_id in the manifest, so the split generator (P038) keeps them together.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

COLUMNS = ["sample_id", "dataset", "image_file", "file_sha256", "pixel_sha256", "error"]


def hash_image(path: str | Path) -> dict:
    """Both hashes of one image; `pixel_sha256` is None and `error` set if it cannot be decoded."""
    out: dict = {"file_sha256": None, "pixel_sha256": None, "error": None}
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["file_sha256"] = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(path) as img:
            rgb = np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))
        h = hashlib.sha256()
        h.update(np.array(rgb.shape, dtype=np.int64).tobytes())  # same bytes in another shape are not the same picture
        h.update(rgb.tobytes())
        out["pixel_sha256"] = h.hexdigest()
    except Exception as e:  # noqa: BLE001 - every decoder failure is recorded, none stops the run
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def duplicate_groups(table: pd.DataFrame) -> pd.DataFrame:
    """One row per image that has at least one duplicate.

    Columns: sample_id, dataset, image_file, dup_group (the group's id: the pixel hash, or the file hash when the
    picture could not be decoded), by_file (shares bytes with another image), by_pixels (shares pixels only),
    group_size, group_datasets (sorted, comma-separated), cross_dataset.
    """
    t = table.copy()
    t["key"] = t["pixel_sha256"].where(t["pixel_sha256"].notna(), "file:" + t["file_sha256"].fillna(""))
    t = t[t["key"].notna() & (t["key"] != "file:")]
    sizes = t.groupby("key")["sample_id"].transform("size")
    dup = t[sizes > 1].copy()
    if dup.empty:
        return pd.DataFrame(columns=["sample_id", "dataset", "image_file", "dup_group", "by_file", "by_pixels",
                                     "group_size", "group_datasets", "cross_dataset"])
    file_counts = dup.groupby("file_sha256")["sample_id"].transform("size")
    dup["by_file"] = file_counts > 1
    dup["by_pixels"] = ~dup["by_file"]
    dup["dup_group"] = dup["key"]
    dup["group_size"] = dup.groupby("dup_group")["sample_id"].transform("size")
    dup["group_datasets"] = dup.groupby("dup_group")["dataset"].transform(lambda s: ",".join(sorted(set(s))))
    dup["cross_dataset"] = dup["group_datasets"].str.contains(",")
    return dup[["sample_id", "dataset", "image_file", "dup_group", "by_file", "by_pixels", "group_size",
                "group_datasets", "cross_dataset"]].sort_values(["cross_dataset", "dup_group", "dataset", "image_file"],
                                                                 ascending=[False, True, True, True]).reset_index(drop=True)


def pair_summary(groups: pd.DataFrame) -> pd.DataFrame:
    """How many duplicate groups span each combination of datasets."""
    if groups.empty:
        return pd.DataFrame(columns=["datasets", "groups", "images"])
    g = groups.drop_duplicates("dup_group")
    s = g.groupby("group_datasets").agg(groups=("dup_group", "size"), images=("group_size", "sum")).reset_index()
    return s.rename(columns={"group_datasets": "datasets"}).sort_values("groups", ascending=False).reset_index(drop=True)


def report_markdown(table: pd.DataFrame, groups: pd.DataFrame, max_examples: int = 30) -> str:
    n_images = len(table)
    n_dup_images = len(groups)
    n_groups = groups["dup_group"].nunique() if not groups.empty else 0
    n_cross = groups.loc[groups["cross_dataset"], "dup_group"].nunique() if not groups.empty else 0
    by_file = int(groups["by_file"].sum()) if not groups.empty else 0
    lines = ["# Exact duplicates report (P024)", "",
             f"{n_images:,} images hashed (file bytes and decoded pixels); "
             f"{int(table['error'].notna().sum())} could not be decoded.", "",
             f"- Images with at least one exact duplicate: **{n_dup_images:,}** in **{n_groups:,}** groups.",
             f"- Of these, {by_file:,} images share identical file bytes with another; "
             f"{n_dup_images - by_file:,} share only the pixels (re-saved copies).",
             f"- Groups spanning more than one dataset: **{n_cross:,}**.", "",
             "## Duplicate groups by dataset combination", "", "| Datasets | Groups | Images |", "|---|---|---|"]
    for _, r in pair_summary(groups).iterrows():
        lines.append(f"| {r['datasets']} | {r['groups']:,} | {r['images']:,} |")
    lines += ["", "## Within-dataset duplicates", "", "| Dataset | Images in a duplicate group | Groups |", "|---|---|---|"]
    if not groups.empty:
        within = groups[~groups["cross_dataset"]]
        for dataset, part in within.groupby("dataset", sort=True):
            lines.append(f"| {dataset} | {len(part):,} | {part['dup_group'].nunique():,} |")
    lines += ["", f"## Examples (first {min(max_examples, n_groups)} groups, cross-dataset first)", "",
              "| Group | Dataset | File | Same bytes |", "|---|---|---|---|"]
    shown = 0
    if not groups.empty:
        for key, part in groups.groupby("dup_group", sort=False):
            if shown >= max_examples:
                break
            for _, r in part.iterrows():
                lines.append(f"| {key[:12]} | {r['dataset']} | `{r['image_file']}` | {'yes' if r['by_file'] else 'no'} |")
            shown += 1
    lines += ["", "## Policy", "",
              "1. Every member of a duplicate group gets the same `group_id` in the manifest (`data/duplicate_groups.parquet`,",
              "   applied when the splits are generated in P038), so no exact duplicate can sit on both sides of a split.",
              "2. Cross-dataset groups are kept, not deleted: they are evidence about how the public datasets were built and",
              "   are reported in the data card. Within-dataset byte-identical copies count once in dataset statistics.",
              "3. Duplicates with different labels are passed to the label-conflict audit (P026); nothing is auto-corrected.", ""]
    return "\n".join(lines)
