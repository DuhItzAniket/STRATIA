"""Label-conflict audit over the manifest (P026): the same picture with different labels -> list, report, sheets.

    python scripts/audit_label_conflicts.py [--mask-agree 0.95] [--sheet-pairs 16]

Inputs (all produced by P024 and P025): data/manifest.parquet, data/duplicate_groups.parquet,
data/near_duplicate_pairs.parquet, data/swinyseg_sources.parquet, data/perceptual_hashes.parquet.
Outputs: data/label_conflicts.parquet (every compared pair, both labels, flags), data/mask_agreement.parquet
(copies of segmentation images with their aligned-mask agreement), docs/data/label_conflicts_report.md and
contact sheets docs/data/figures/label_conflict_*.jpg, docs/data/figures/mask_*.jpg.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data import label_conflicts as lc  # noqa: E402
from stratia.data.near_duplicates import contact_sheet  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402

TIME_SERIES = ("eye2sky", "mgcd", "montenegro", "almeria")   # one camera, consecutive frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-agree", type=float, default=0.95, help="pixel agreement below which two copies' masks conflict")
    ap.add_argument("--sheet-pairs", type=int, default=16, help="pairs per contact sheet")
    a = ap.parse_args()
    t0 = time.perf_counter()
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    figures = Path("docs/data/figures")

    m = pd.read_parquet("data/manifest.parquet")
    pos_of_sample = pd.Series(np.arange(len(m)), index=m.sample_id)
    pos_of_file = pd.Series(np.arange(len(m)), index=m.image_file)

    # ---- the three kinds of "same picture" ------------------------------------------------------------------
    dup = pd.read_parquet("data/duplicate_groups.parquet")
    exact = lc.pairs_within_groups(pd.Series(pos_of_sample[dup.sample_id.values].values), dup.dup_group.reset_index(drop=True))
    nd_pairs = pd.read_parquet("data/near_duplicate_pairs.parquet")
    swim = pd.read_parquet("data/swinyseg_sources.parquet")
    copies = pd.concat([nd_pairs.loc[nd_pairs["copy"], ["i", "j"]],
                        pd.DataFrame({"i": pos_of_file[swim.swinyseg_file.values].values,
                                      "j": pos_of_file[swim.source_file.values].values})], ignore_index=True)
    same_scene = nd_pairs.loc[nd_pairs.same_scene & ~nd_pairs["copy"], ["i", "j"]]
    pairs = lc.merge_pairs({"exact": exact, "copy": copies, "same_scene": same_scene})
    # Inside a camera's own time series a pHash distance <= 2 is not evidence of a copy: the pHash of a smooth sky
    # carries little information (P025), so two frames of one camera that hash alike are treated as the same scene,
    # not as the same picture. Copies across datasets and within the constructed datasets (SWIM family) stand.
    ds = m.dataset.values
    series = np.isin(ds[pairs.i.values], TIME_SERIES) & np.isin(ds[pairs.j.values], TIME_SERIES)
    pairs.loc[series & (pairs.kind == "copy"), "kind"] = "same_scene"
    print(f"pairs: {pairs.kind.value_counts().to_dict()}", flush=True)

    # ---- which flip or rotation lays i like j (from the P025 hashes) ---------------------------------------------
    hashes = pd.read_parquet("data/perceptual_hashes.parquet")
    ph = hashes[[f"phash_{k}" for k in range(8)]].values.astype(np.uint64)
    variant, distance = lc.best_variant(ph[pairs.i.values], ph[pairs.j.values, 0])

    # ---- class labels -------------------------------------------------------------------------------------------
    compared = lc.compare_labels(m, pairs)
    compared["variant"], compared["phash_distance"] = variant, distance
    compared["masked"] = pd.notna(m.seg_file.values[compared.i.values]) & pd.notna(m.seg_file.values[compared.j.values])
    utc = m.utc.dt.tz_convert(None).values
    ui, uj = utc[compared.i.values], utc[compared.j.values]
    gap = np.abs((ui - uj).astype("timedelta64[s]").astype(float))
    gap[np.isnat(ui) | np.isnat(uj)] = np.nan                                  # no timestamp, no gap
    compared["gap_s"] = gap
    compared.to_parquet("data/label_conflicts.parquet", index=False)
    for ds_name, part in compared[compared.conflict & compared.same_dataset].groupby("dataset_i"):
        strong = part[part.kind.isin(["exact", "copy"])]
        print(f"{ds_name}: {len(part)} conflicting pairs touch {pd.unique(np.concatenate([part.i, part.j])).size} images; "
              f"exact+copy conflicts touch {pd.unique(np.concatenate([strong.i, strong.j])).size}", flush=True)
        if part.gap_s.notna().any():
            print(f"    time gap median {part.gap_s.median():.0f} s, <= 1 h: {(part.gap_s <= 3600).mean():.1%}, "
                  f"> 1 day: {(part.gap_s > 86400).mean():.1%}", flush=True)
    counts = {kind: {"pairs": int((compared.kind == kind).sum()),
                     "same_dataset": int((compared.same_dataset & (compared.kind == kind)).sum()),
                     "labelled": int((compared.compared & (compared.kind == kind)).sum()),
                     "masked": int((compared.masked & (compared.kind == kind)).sum())} for kind in lc.KINDS}
    summary = lc.conflict_summary(compared)
    confusions = lc.label_confusions(compared)
    crossing = (compared[compared.crosses_official_split].groupby(["dataset_i", "kind"]).size().rename("pairs")
                .reset_index().rename(columns={"dataset_i": "dataset"}))
    hard = compared[compared.conflict & compared.kind.isin(["exact", "copy"])].sort_values(["kind", "dataset_i", "image_file_i"])
    examples = pd.concat([hard[hard.kind == "exact"], hard[hard.kind == "copy"].head(60)])
    note = (f"{len(hard) - len(examples):,} further copy conflicts are in `data/label_conflicts.parquet`."
            if len(hard) > len(examples) else "")
    print(summary.to_string(index=False), flush=True)

    # ---- masks of copies ----------------------------------------------------------------------------------------
    sel = compared.masked & compared.kind.isin(["exact", "copy"])
    masks = lc.compare_masks(m, compared.loc[sel, ["i", "j"]], data_root, compared.loc[sel, "variant"].values)
    masks = masks.merge(compared.loc[sel, ["i", "j", "kind"]], on=["i", "j"], how="left")
    ds_i, ds_j = m.dataset.values[masks.i.values], m.dataset.values[masks.j.values]
    masks["datasets"] = [" / ".join(sorted({x, y})) for x, y in zip(ds_i, ds_j, strict=True)]
    masks["conflict"] = masks.agreement < a.mask_agree
    masks.to_parquet("data/mask_agreement.parquet", index=False)
    mask_summary = lc.mask_summary(masks, a.mask_agree)
    print(mask_summary.to_string(index=False), flush=True)

    raters = lc.rater_agreement(m)

    # ---- figures ------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    conflicts = compared[compared.conflict & compared.same_dataset].copy()
    conflicts["caption"] = conflicts.label_i.astype(str) + " | " + conflicts.label_j.astype(str)
    sheets = []
    for kind in lc.KINDS:
        for ds, part in conflicts[conflicts.kind == kind].groupby("dataset_i"):
            take = part if len(part) <= a.sheet_pairs else part.iloc[rng.choice(len(part), a.sheet_pairs, replace=False)]
            out = figures / f"label_conflict_{kind}_{ds}.jpg"
            contact_sheet(take, m.image_file.tolist(), cache_root, out, caption_cols=("caption",))
            sheets.append(out)
    if len(masks):
        worst = masks.sort_values("agreement").drop_duplicates("i").drop_duplicates("j").head(10)   # one row per picture
        lc.mask_sheet(worst, m, data_root, cache_root, figures / "mask_conflicts_worst.jpg")
        typical = masks.iloc[rng.choice(len(masks), min(6, len(masks)), replace=False)]
        lc.mask_sheet(typical, m, data_root, cache_root, figures / "mask_agreement_typical.jpg")
        sheets += [figures / "mask_conflicts_worst.jpg", figures / "mask_agreement_typical.jpg"]

    report = lc.report_markdown(counts, summary, confusions, crossing, mask_summary, a.mask_agree, raters, examples, note)
    report += "\n## Contact sheets\n\n" + "\n".join(f"- `{s.as_posix()}`" for s in sheets) + "\n"
    Path("docs/data/label_conflicts_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: data/label_conflicts.parquet, data/mask_agreement.parquet, "
          f"docs/data/label_conflicts_report.md, {len(sheets)} sheets", flush=True)


if __name__ == "__main__":
    main()
