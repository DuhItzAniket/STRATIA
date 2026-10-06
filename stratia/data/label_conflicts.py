"""Label-conflict audit (P026): the same picture carrying different labels.

Three kinds of "same picture", from the two previous phases:
    exact        P024 duplicate groups (identical pixels)           labels must agree; masks must be identical
    copy         P025 pairs with dihedral pHash <= 2, and the exhaustive SWIM-family table
                                                                    labels must agree; masks must agree once the
                                                                    copy's flip or rotation is undone
    same_scene   P025 pairs with cosine >= 0.97 that are not copies (consecutive frames, re-posted crops)
                                                                    labels may differ legitimately, so the
                                                                    disagreement rate is a label-noise floor, not a list
                                                                    of errors
Labels compared: the dataset's native class (`source_label`: CCSN genus, MGCD and SWIMCAT sky type, Montenegro
étage set) and, for the segmentation datasets, the mask itself (pixel agreement after aligning the copy).
Nothing is corrected here. Every conflict is listed with both labels; the harmonisation phases (P033, P034) decide
what the training label of a conflicting sample is, and no conflicting image may sit in a test split.
"""

from __future__ import annotations

import itertools
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from stratia.data.near_duplicates import dihedral, popcount
from stratia.data.segmentation import IGNORE, load_masks

KINDS = ("exact", "copy", "same_scene")           # in order of strength: a pair keeps its strongest kind
PAIR_COLUMNS = ["i", "j", "kind"]

# ------------------------------------------------------------------------------------------------ pairs


def pairs_within_groups(positions: pd.Series, group_of: pd.Series) -> pd.DataFrame:
    """All pairs (i < j, manifest positions) inside each group. `positions` are manifest positions indexed like
    `group_of` (one row per image that has a duplicate); groups are small, so every pair is listed."""
    rows = []
    for _, members in pd.Series(positions.values).groupby(group_of.values):
        rows.extend(itertools.combinations(sorted(int(p) for p in members), 2))
    return pd.DataFrame(rows, columns=["i", "j"], dtype=np.int64)


def merge_pairs(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per unordered pair with its strongest kind (`KINDS` order); `tables` maps kind -> DataFrame[i, j]."""
    parts = []
    for kind in KINDS:
        t = tables.get(kind)
        if t is None or len(t) == 0:
            continue
        i, j = np.minimum(t.i.values, t.j.values), np.maximum(t.i.values, t.j.values)
        parts.append(pd.DataFrame({"i": i, "j": j, "kind": kind}))
    if not parts:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    pairs = pd.concat(parts, ignore_index=True)
    pairs = pairs[pairs.i != pairs.j]
    return pairs.drop_duplicates(["i", "j"], keep="first").reset_index(drop=True)   # first = strongest kind


# ------------------------------------------------------------------------------------------- class labels


def compare_labels(manifest: pd.DataFrame, pairs: pd.DataFrame, label_col: str = "source_label") -> pd.DataFrame:
    """The pairs with both images' dataset, file, official split and native label; `conflict` is True when both
    labels exist and differ (a multi-label set such as Montenegro's is a sorted string, so string inequality is
    set inequality)."""
    out = pairs.copy()
    for side in ("i", "j"):
        rows = manifest.iloc[out[side].values]
        out[f"dataset_{side}"] = rows.dataset.values
        out[f"image_file_{side}"] = rows.image_file.values
        out[f"official_split_{side}"] = rows.official_split.values
        out[f"label_{side}"] = rows[label_col].values
    out["same_dataset"] = out.dataset_i.values == out.dataset_j.values
    both = out.label_i.notna() & out.label_j.notna()
    out["compared"] = both
    out["conflict"] = both & (out.label_i != out.label_j)
    out["crosses_official_split"] = (out.same_dataset & (out.official_split_i != out.official_split_j)
                                     & ~out.official_split_i.isin(["none"]) & ~out.official_split_j.isin(["none"]))
    return out


def conflict_summary(compared: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and kind: pairs with labels on both sides, conflicting pairs, their share, images involved."""
    rows = []
    t = compared[compared.compared & compared.same_dataset]
    for (ds, kind), part in t.groupby(["dataset_i", "kind"], sort=True):
        c = part[part.conflict]
        rows.append({"dataset": ds, "kind": kind, "pairs": len(part), "conflicts": len(c),
                     "rate": len(c) / len(part) if len(part) else 0.0,
                     "images": int(pd.unique(np.concatenate([c.i.values, c.j.values])).size)})
    return pd.DataFrame(rows, columns=["dataset", "kind", "pairs", "conflicts", "rate", "images"])


def label_confusions(compared: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """Which label pairs conflict most, per dataset and kind (unordered label pairs)."""
    c = compared[compared.conflict & compared.same_dataset].copy()
    if len(c) == 0:
        return pd.DataFrame(columns=["dataset", "kind", "labels", "pairs"])
    a, b = c.label_i.astype(str), c.label_j.astype(str)
    c["labels"] = np.where(a <= b, a + " / " + b, b + " / " + a)
    c["dataset"] = c.dataset_i
    out = (c.groupby(["dataset", "kind", "labels"]).size().rename("pairs").reset_index()
            .sort_values(["dataset", "kind", "pairs"], ascending=[True, True, False]))
    return out.groupby(["dataset", "kind"], group_keys=False).head(top).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------- masks


def best_variant(variants_a: np.ndarray, plain_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row: the dihedral variant of a that is nearest to b's plain pHash, and that distance. Ties go to
    the lowest index, so an untransformed copy (or a symmetric image) is treated as untransformed."""
    d = popcount(variants_a ^ plain_b[:, None])
    k = d.argmin(axis=1)
    return k, d[np.arange(len(d)), k]


def align_mask(mask_a: np.ndarray, k: int, shape: tuple[int, int]) -> np.ndarray:
    """Apply dihedral variant `k` to a's mask (so it lies like b's picture) and bring it to b's mask shape by
    nearest-neighbour resampling (a label map must not be interpolated)."""
    aligned = np.ascontiguousarray(dihedral(mask_a)[int(k)])
    if aligned.shape != tuple(shape):
        aligned = cv2.resize(aligned, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)
    return aligned


def mask_agreement(mask_a: np.ndarray, mask_b: np.ndarray, k: int = 0, ignore: int = IGNORE) -> dict:
    """Pixel agreement and class IoUs between a's mask, aligned by `k`, and b's mask, over pixels labelled on both
    sides. `iou_cloud` is for the sky-parsing class 2; `iou_mean` averages the classes present on either side."""
    a = align_mask(mask_a, k, mask_b.shape).astype(np.int16)
    b = mask_b.astype(np.int16)
    valid = (a != ignore) & (b != ignore)
    n = int(valid.sum())
    if n == 0:
        return {"agreement": np.nan, "iou_cloud": np.nan, "iou_mean": np.nan, "n_valid": 0}
    a, b = a[valid], b[valid]
    ious = {}
    for cls in np.union1d(a, b):
        inter, union = int(((a == cls) & (b == cls)).sum()), int(((a == cls) | (b == cls)).sum())
        ious[int(cls)] = inter / union if union else np.nan
    return {"agreement": float((a == b).mean()), "iou_cloud": ious.get(2, np.nan),
            "iou_mean": float(np.nanmean(list(ious.values()))), "n_valid": n}


@lru_cache(maxsize=512)
def _cached_masks(dataset: str, path: str) -> tuple[np.ndarray, np.ndarray]:
    return load_masks(dataset, path)


def compare_masks(manifest: pd.DataFrame, pairs: pd.DataFrame, data_root: str | Path, k: np.ndarray) -> pd.DataFrame:
    """Mask agreement for every pair whose two images both have a mask (sky parsing; and the layer map where both
    sides have one). `k[n]` is the dihedral variant that lays image i of pair n like image j."""
    root = Path(data_root)
    rows = []
    for n, (i, j) in enumerate(zip(pairs.i.values, pairs.j.values, strict=True)):
        ri, rj = manifest.iloc[int(i)], manifest.iloc[int(j)]
        if pd.isna(ri.seg_file) or pd.isna(rj.seg_file):
            continue
        sky_i, layer_i = _cached_masks(ri.dataset, str(root / ri.seg_file))
        sky_j, layer_j = _cached_masks(rj.dataset, str(root / rj.seg_file))
        row = {"i": int(i), "j": int(j), "variant": int(k[n]), **mask_agreement(sky_i, sky_j, int(k[n]))}
        if bool(ri.has_layers) and bool(rj.has_layers):
            layers = mask_agreement(layer_i, layer_j, int(k[n]))
            row["layer_agreement"], row["layer_iou_mean"] = layers["agreement"], layers["iou_mean"]
        else:
            row["layer_agreement"], row["layer_iou_mean"] = np.nan, np.nan
        rows.append(row)
    cols = ["i", "j", "variant", "agreement", "iou_cloud", "iou_mean", "n_valid", "layer_agreement", "layer_iou_mean"]
    return pd.DataFrame(rows, columns=cols)


def mask_summary(masks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per dataset pair and kind: how many copies were compared, how well their masks agree, how many fall below
    `threshold` (the mask-conflict rule)."""
    rows = []
    for (pair, kind), part in masks.groupby(["datasets", "kind"], sort=True):
        rows.append({"datasets": pair, "kind": kind, "pairs": len(part),
                     "median_agreement": float(part.agreement.median()), "p10_agreement": float(part.agreement.quantile(0.1)),
                     "min_agreement": float(part.agreement.min()), "median_iou_cloud": float(part.iou_cloud.median()),
                     "below_threshold": int((part.agreement < threshold).sum())})
    return pd.DataFrame(rows, columns=["datasets", "kind", "pairs", "median_agreement", "p10_agreement", "min_agreement",
                                       "median_iou_cloud", "below_threshold"])


# ------------------------------------------------------------------------------------------------ raters


def rater_agreement(manifest: pd.DataFrame, columns: tuple[str, ...] = ("oktas_dist", "h_dist", "cl_dist", "cm_dist",
                                                                        "ch_dist")) -> pd.DataFrame:
    """Where several raters labelled an image (Montenegro: distributions stored as JSON), how often they agree: the
    majority share per image, summarised per quantity and dataset."""
    rows = []
    for ds, part in manifest.groupby("dataset"):
        for col in columns:
            if col not in part or part[col].notna().sum() == 0:
                continue
            majority = part[col].dropna().map(lambda s: max(json.loads(s).values()))
            rows.append({"dataset": ds, "quantity": col.removesuffix("_dist"), "images": len(majority),
                         "median_majority": float(majority.median()), "unanimous": float((majority >= 0.999).mean()),
                         "no_majority": float((majority <= 0.5).mean())})
    return pd.DataFrame(rows, columns=["dataset", "quantity", "images", "median_majority", "unanimous", "no_majority"])


# ------------------------------------------------------------------------------------------------ figures


def mask_sheet(pairs: pd.DataFrame, manifest: pd.DataFrame, data_root: str | Path, cache_root: str | Path,
               out: str | Path, tile: int = 150) -> Path:
    """One row per pair: image i laid like j, its mask aligned the same way, image j, its mask; agreement printed.
    Cloud is white, sky dark blue, ignored pixels grey."""
    from stratia.data.image_cache import cache_path, read_rgb

    root = Path(data_root)
    palette = np.array([[40, 40, 40], [30, 60, 160], [235, 235, 235], [250, 200, 40]], dtype=np.uint8)
    sheet = np.full((max(len(pairs), 1) * (tile + 22), 4 * (tile + 4), 3), 255, dtype=np.uint8)
    for n, (_, p) in enumerate(pairs.iterrows()):
        ri, rj = manifest.iloc[int(p["i"])], manifest.iloc[int(p["j"])]
        k = int(p["variant"])
        img_i = dihedral(read_rgb(cache_path(cache_root, ri.image_file)))[k]
        img_j = read_rgb(cache_path(cache_root, rj.image_file))
        sky_i = align_mask(_cached_masks(ri.dataset, str(root / ri.seg_file))[0], k, (tile, tile))
        sky_j = align_mask(_cached_masks(rj.dataset, str(root / rj.seg_file))[0], 0, (tile, tile))
        tiles = [cv2.resize(np.ascontiguousarray(img_i), (tile, tile), interpolation=cv2.INTER_AREA),
                 _colour(sky_i, palette), cv2.resize(img_j, (tile, tile), interpolation=cv2.INTER_AREA), _colour(sky_j, palette)]
        y = n * (tile + 22)
        for c, t in enumerate(tiles):
            sheet[y:y + tile, c * (tile + 4):c * (tile + 4) + tile] = t
        text = (f"{ri.dataset}/{Path(ri.image_file).name} vs {rj.dataset}/{Path(rj.image_file).name}  k={k}  "
                f"agree {p['agreement']:.3f}")
        cv2.putText(sheet, text, (2, y + tile + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise OSError(f"cannot encode {out}")
    enc.tofile(str(out))
    return out


def _colour(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    rgb = np.full(mask.shape + (3,), 128, dtype=np.uint8)
    known = mask < len(palette)
    rgb[known] = palette[mask[known]]
    return rgb


# ------------------------------------------------------------------------------------------------ report


def report_markdown(counts: dict, summary: pd.DataFrame, confusions: pd.DataFrame, crossing: pd.DataFrame,
                    masks: pd.DataFrame, mask_threshold: float, raters: pd.DataFrame, examples: pd.DataFrame,
                    examples_note: str = "") -> str:
    def pct(x: float) -> str:
        return f"{x:.1%}"

    lines = ["# Label-conflict audit (P026)", "",
             "Pairs of images that are the same picture (exact duplicates, P024; copies, P025) or the same scene "
             "(cosine >= 0.97, P025), compared on the labels each dataset ships with. Nothing is corrected; the "
             "policy at the end says what the later phases do with each kind of conflict.", "",
             "## Pairs compared", "",
             "| Kind | Pairs | Of which within one dataset | With a class label on both sides | With masks on both sides |",
             "|---|---|---|---|---|"]
    for kind in KINDS:
        c = counts.get(kind, {})
        lines.append(f"| {kind} | {c.get('pairs', 0):,} | {c.get('same_dataset', 0):,} | {c.get('labelled', 0):,} | "
                     f"{c.get('masked', 0):,} |")
    lines += ["", "## Class-label conflicts", "",
              "A conflict is a pair of same-dataset images with two different native labels. For exact duplicates and "
              "copies this is a labelling error or a deliberately ambiguous photograph; for same-scene pairs (frames "
              "seconds or minutes apart, re-posted crops) it is the rate at which the label changes while the picture "
              "barely does: a floor on label noise.", "",
              "| Dataset | Kind | Pairs | Conflicts | Rate | Images involved |", "|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.dataset} | {r.kind} | {r.pairs:,} | {r.conflicts:,} | {pct(r.rate)} | {r.images:,} |")
    lines += ["", "### Which labels conflict", "", "| Dataset | Kind | Labels | Pairs |", "|---|---|---|---|"]
    for _, r in confusions.iterrows():
        lines.append(f"| {r.dataset} | {r.kind} | {r.labels} | {r.pairs:,} |")
    lines += ["", "### Exact duplicates and copies with different labels", "",
              "| Kind | Dataset | Image i | Label i | Image j | Label j |", "|---|---|---|---|---|---|"]
    for _, r in examples.iterrows():
        lines.append(f"| {r.kind} | {r.dataset_i} | `{r.image_file_i}` | {r.label_i} | `{r.image_file_j}` | {r.label_j} |")
    if examples_note:
        lines += ["", examples_note]
    lines += ["", "## Pairs that cross a published split", "",
              "Same-dataset pairs whose two images sit in different official splits (train/val/test as released). Any "
              "such pair is leakage in the published protocol.", "",
              "| Dataset | Kind | Pairs crossing the official split |", "|---|---|---|"]
    for _, r in crossing.iterrows():
        lines.append(f"| {r.dataset} | {r.kind} | {r.pairs:,} |")
    lines += ["", "## Mask agreement between copies", "",
              f"For exact duplicates and copies of segmentation images, the mask of image i is flipped or rotated the same "
              f"way as its picture and compared with the mask of image j, pixel by pixel. A pair whose agreement is below "
              f"{mask_threshold:.2f} is a mask conflict.", "",
              "| Datasets | Kind | Pairs | Median agreement | 10th percentile | Minimum | Median cloud IoU | Below threshold |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in masks.iterrows():
        lines.append(f"| {r.datasets} | {r.kind} | {r.pairs:,} | {r.median_agreement:.3f} | {r.p10_agreement:.3f} | "
                     f"{r.min_agreement:.3f} | {r.median_iou_cloud:.3f} | {r.below_threshold:,} |")
    lines += ["", "## Rater agreement (where several raters labelled one image)", "",
              "Majority share = the largest fraction of raters giving the same answer for an image.", "",
              "| Dataset | Quantity | Images | Median majority share | Unanimous | No majority (<= 50 %) |",
              "|---|---|---|---|---|---|"]
    for _, r in raters.iterrows():
        lines.append(f"| {r.dataset} | {r.quantity} | {r.images:,} | {r.median_majority:.2f} | {pct(r.unanimous)} | "
                     f"{pct(r.no_majority)} |")
    lines += ["", "## Policy", "",
              "1. **Exact duplicates and copies with different class labels** are counted once in the data card and are "
              "excluded from every test split. Their training label is decided by the harmonisation phases (P033, "
              "P034): where the ontology holds a set (genus_set), the set is the union of the conflicting labels; where "
              "it holds one value and the conflicting labels disagree at that level, the sample carries no label for "
              "that task. No label is edited by hand.",
              "2. **Same-scene disagreements** are kept as they are: the images differ, and a sky can change between "
              "frames. The disagreement rate is reported in the data card as the label-consistency floor of each "
              "dataset, and the group-level splits (P038) keep such pairs on one side.",
              "3. **Mask conflicts between copies** are listed in `data/mask_agreement.parquet`; the SWIM family is "
              "one source for the leave-one-dataset-out protocol (P025), so which mask is \"right\" does not change a "
              "split; conflicting pairs are excluded from any test split and go to the segmentation harmonisation "
              "(P034) for review.",
              "4. **Rater disagreement** stays in the manifest as a distribution; training may use it as a soft label "
              "and evaluation reports agreement with the majority (P034 decides). Nothing is collapsed here."]
    return "\n".join(lines) + "\n"
