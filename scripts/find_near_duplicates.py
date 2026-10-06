"""Near-duplicate search over the manifest (P025): DINOv3 embeddings + perceptual hashes -> pairs, groups, report.

    python scripts/find_near_duplicates.py [--model vits16] [--k 10] [--min-sim 0.90] [--cos 0.97] [--phash 10]

Steps (each cached, so a re-run with other thresholds is cheap):
  1. embeddings  cache/features/dinov3_<model>_224_cls.npy  (ViT CLS at 224 px, L2-normalised; also used by P029)
  2. hashes      data/perceptual_hashes.parquet              (dHash, pHash of all 8 flips/rotations)
  3. pairs       data/near_duplicate_pairs.parquet           (k nearest neighbours with cosine >= --min-sim)
  4. groups      data/near_duplicate_groups.parquet          (cosine >= --cos or pHash distance <= --phash)
  5. report      docs/data/near_duplicates_report.md, contact sheets in docs/data/figures/
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data import near_duplicates as nd  # noqa: E402
from stratia.data.image_cache import CachedImageDataset  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402

MODELS = {"vits16": "facebook/dinov3-vits16-pretrain-lvd1689m", "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m"}


def embed(files: list[str], cache_root: Path, model_name: str, out: Path, batch: int = 128) -> np.ndarray:
    if out.exists():
        emb = np.load(out)
        if emb.shape[0] == len(files):
            print(f"embeddings from {out}", flush=True)
            return emb
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODELS[model_name]).cuda().eval().half()
    loader = DataLoader(CachedImageDataset(files, cache_root, size=224), batch_size=batch, num_workers=8,
                        pin_memory=True, persistent_workers=True)
    mean = torch.tensor(nd.IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1).half()
    std = torch.tensor(nd.IMAGENET_STD, device="cuda").view(1, 3, 1, 1).half()
    chunks = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for k, (x, _) in enumerate(loader, 1):
            x = (x.cuda(non_blocking=True).half() / 255.0 - mean) / std
            feats = model(pixel_values=x).pooler_output.float()
            chunks.append(torch.nn.functional.normalize(feats, dim=1).cpu().numpy())
            if k % 50 == 0:
                print(f"  {k * batch:,}/{len(files):,} ({k * batch / (time.perf_counter() - t0):.0f} img/s)", flush=True)
    emb = np.concatenate(chunks).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    print(f"embedded {len(files):,} images in {time.perf_counter() - t0:.0f} s -> {out}", flush=True)
    return emb


def hashes(files: list[str], root: Path, out: Path, workers: int) -> pd.DataFrame:
    if out.exists():
        h = pd.read_parquet(out)
        if len(h) == len(files):
            print(f"hashes from {out}", flush=True)
            return h
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(nd.hashes_of, [str(root / f) for f in files], chunksize=64))
    # One uint64 column per variant (Python ints above 2^63 do not survive the trip into parquet otherwise);
    # an undecodable image gets all-zero hashes and an error.
    h = pd.DataFrame({"image_file": files, "error": [r["error"] for r in results]})
    for kind in ("dhash", "phash"):
        for k in range(8):
            h[f"{kind}_{k}"] = np.array([r[kind][k] or 0 for r in results], dtype=np.uint64)
    h.to_parquet(out, index=False)
    print(f"hashed {len(files):,} images in {time.perf_counter() - t0:.0f} s; errors {h['error'].notna().sum()}", flush=True)
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vits16", choices=sorted(MODELS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-sim", type=float, default=0.90, help="keep neighbour pairs with cosine at least this")
    ap.add_argument("--cos", type=float, default=0.97, help="cosine at or above which a pair shows the same scene")
    ap.add_argument("--phash", type=int, default=2, help="pHash distance at or below which a pair is a copy")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    paths = load_paths()
    root, cache = Path(paths["data_root"]), Path(paths["cache_root"])
    m = pd.read_parquet("data/manifest.parquet", columns=["sample_id", "dataset", "image_file", "camera_id", "utc"])
    files = list(m.image_file)

    emb = embed(files, cache, a.model, cache / "features" / f"dinov3_{a.model}_224_cls.npy")
    h = hashes(files, root, Path("data/perceptual_hashes.parquet"), a.workers)

    t0 = time.perf_counter()
    idx, sim = nd.nearest_neighbours(emb, k=a.k)
    pairs = nd.candidate_pairs(idx, sim, a.min_sim)
    print(f"{len(pairs):,} candidate pairs with cosine >= {a.min_sim} in {time.perf_counter() - t0:.0f} s", flush=True)
    ph = h[[f"phash_{k}" for k in range(8)]].to_numpy(dtype=np.uint64)
    dh = h[[f"dhash_{k}" for k in range(8)]].to_numpy(dtype=np.uint64)
    undecoded = h["error"].notna().to_numpy()
    pairs["phash"] = nd.pair_min_hamming(ph, pairs.i.to_numpy(), pairs.j.to_numpy())
    pairs["dhash"] = nd.pair_min_hamming(dh, pairs.i.to_numpy(), pairs.j.to_numpy())
    bad = undecoded[pairs.i.to_numpy()] | undecoded[pairs.j.to_numpy()]
    pairs.loc[bad, ["phash", "dhash"]] = 64
    for col in ("dataset", "image_file", "camera_id", "utc"):
        pairs[f"{col}_i"] = m[col].values[pairs.i.values]
        pairs[f"{col}_j"] = m[col].values[pairs.j.values]
    pairs["same_dataset"] = pairs.dataset_i == pairs.dataset_j
    pairs["same_camera"] = pairs.camera_id_i == pairs.camera_id_j
    both_timed = pairs.utc_i.notna() & pairs.utc_j.notna()
    pairs["gap_s"] = np.where(both_timed, (pairs.utc_i - pairs.utc_j).abs().dt.total_seconds(), np.nan)
    # A same-station Eye2Sky pair is the camera's own time series, not a copy: temporal blocking handles it (P027).
    pairs["temporal"] = (pairs.dataset_i == "eye2sky") & (pairs.dataset_j == "eye2sky") & pairs.same_camera
    # Two tiers: a copy (the same source picture, possibly flipped, rotated, resized or re-encoded: the dihedral
    # pHash sees it, the embedding often does not) and the same scene (consecutive frames of one camera: the
    # embedding sees it, the pHash of a smooth sky is unreliable). Both leak across a random split.
    pairs["copy"] = (pairs.phash <= a.phash) & ~pairs.temporal
    pairs["same_scene"] = (pairs.cosine >= a.cos) & ~pairs.temporal
    pairs["near_duplicate"] = pairs["copy"] | pairs.same_scene
    pairs.to_parquet("data/near_duplicate_pairs.parquet", index=False)

    # The SWIM family is small enough to try every cross-dataset pair by pHash alone (no embedding needed):
    # is SWINySEG built from flipped and rotated SWIMSEG / SWINSEG patches?
    swim = {ds: np.flatnonzero(m.dataset.values == ds) for ds in ("swinyseg", "swimseg", "swinseg", "shwimseg")}
    exhaustive = []
    for src in ("swimseg", "swinseg", "shwimseg"):
        hits = nd.exhaustive_matches(ph[swim["swinyseg"]], ph[swim[src]][:, 0], a.phash)
        hits["swinyseg_file"] = m.image_file.values[swim["swinyseg"][hits.a.values]]
        hits["source_file"] = m.image_file.values[swim[src][hits.b.values]]
        hits["source"] = src
        exhaustive.append(hits)
    exhaustive = pd.concat(exhaustive, ignore_index=True)
    exhaustive.to_parquet("data/swinyseg_sources.parquet", index=False)
    swinyseg_copies = exhaustive.swinyseg_file.nunique()
    per_source = exhaustive.groupby("source").swinyseg_file.nunique().to_dict()

    edges = pairs[pairs.near_duplicate]
    labels = nd.connected_components(len(m), edges.i.values, edges.j.values)
    groups = m[["sample_id", "dataset", "image_file"]].copy()
    groups["near_dup_group"] = labels
    groups = groups[groups.near_dup_group >= 0]
    groups["group_size"] = groups.groupby("near_dup_group")["sample_id"].transform("size")
    groups["group_datasets"] = groups.groupby("near_dup_group")["dataset"].transform(
        lambda s: ",".join(sorted(set(s))))
    groups.to_parquet("data/near_duplicate_groups.parquet", index=False)

    # Report and contact sheets.
    figures = Path("docs/data/figures")
    sheets = []
    rng = np.random.default_rng(0)
    for name, part in [("cross_dataset", pairs[~pairs.same_dataset].sort_values("cosine", ascending=False)),
                       ("band_0.97_1.00", pairs[(pairs.cosine >= 0.97) & ~pairs.temporal & pairs.same_dataset]),
                       ("band_0.94_0.97", pairs[(pairs.cosine >= 0.94) & (pairs.cosine < 0.97) & ~pairs.temporal]),
                       ("band_0.90_0.94", pairs[(pairs.cosine < 0.94) & ~pairs.temporal]),
                       ("copies", pairs[pairs["copy"] & ~pairs.same_dataset]),
                       ("phash_3_to_6", pairs[(pairs.phash > a.phash) & (pairs.phash <= 6) & (pairs.cosine < a.cos)
                                              & ~pairs.temporal])]:
        if len(part):
            sample = part if len(part) <= 16 else part.iloc[sorted(rng.choice(len(part), 16, replace=False))]
            sheets.append((name, len(part), nd.contact_sheet(sample, files, cache, figures / f"near_dup_{name}.jpg")))
    lines = ["# Near-duplicates report (P025)", "",
             f"DINOv3 {a.model} CLS embeddings at 224 px, {a.k} nearest neighbours per image, pairs kept with "
             f"cosine >= {a.min_sim}: **{len(pairs):,} candidate pairs** among {len(m):,} images; "
             "pHash/dHash over all flips and rotations.", "",
             f"Rule (chosen from the contact sheets): a pair is a **copy** if its pHash distance over all flips and "
             f"rotations is <= {a.phash}, and shows the **same scene** if its cosine is >= {a.cos}; either makes it a "
             "near-duplicate. Same-station Eye2Sky pairs are excluded (the camera's own time series; handled by "
             "temporal blocking, P027/P037).", "",
             f"- Candidate pairs that are same-station Eye2Sky time series: {int(pairs.temporal.sum()):,}",
             f"- Copies: {int(pairs['copy'].sum()):,} pairs; same scene: {int(pairs.same_scene.sum()):,} pairs; "
             f"near-duplicate pairs in all: **{int(pairs.near_duplicate.sum()):,}**, of which cross-dataset "
             f"{int((pairs.near_duplicate & ~pairs.same_dataset).sum()):,}",
             f"- Exhaustive pHash check of SWINySEG against SWIMSEG, SWINSEG and SHWIMSEG (every pair, no embedding): "
             f"**{swinyseg_copies:,} of {len(swim['swinyseg']):,} SWINySEG images "
             f"({swinyseg_copies / len(swim['swinyseg']):.1%}) are flipped, rotated or re-encoded copies of a "
             f"source-dataset image** (by source: {per_source}); table `data/swinyseg_sources.parquet`",
             f"- Images in a near-duplicate group: **{len(groups):,}** in **{groups.near_dup_group.nunique():,}** groups "
             f"(largest {int(groups.group_size.max()) if len(groups) else 0})", "",
             "## Cosine similarity of candidate pairs by dataset pair", "",
             "| Datasets | Pairs | >= 0.99 | 0.97-0.99 | 0.94-0.97 | 0.90-0.94 | pHash <= " + str(a.phash) + " |",
             "|---|---|---|---|---|---|---|"]
    key = pairs.apply(lambda r: ",".join(sorted([r.dataset_i, r.dataset_j])), axis=1)
    for ds, part in pairs.groupby(key):
        lines.append(f"| {ds} | {len(part):,} | {int((part.cosine >= 0.99).sum()):,} | "
                     f"{int(((part.cosine >= 0.97) & (part.cosine < 0.99)).sum()):,} | "
                     f"{int(((part.cosine >= 0.94) & (part.cosine < 0.97)).sum()):,} | "
                     f"{int((part.cosine < 0.94).sum()):,} | {int((part.phash <= a.phash).sum()):,} |")
    lines += ["", "## Near-duplicate groups by dataset", "",
              "| Dataset | Images in a group | Share of dataset | Groups | Largest group |", "|---|---|---|---|---|"]
    counts = m.dataset.value_counts()
    for ds, part in groups.groupby("dataset"):
        lines.append(f"| {ds} | {len(part):,} | {len(part) / counts[ds]:.1%} | {part.near_dup_group.nunique():,} | "
                     f"{int(part.groupby('near_dup_group').size().max())} |")
    lines += ["", "## Contact sheets (16 pairs each, left and right of a pair side by side)", ""]
    for name, n, path in sheets:
        lines.append(f"- `{path.as_posix()}`: {name} ({n:,} pairs)")
    lines += ["", "## Groups that span datasets", "", "| Group | Datasets | Images |", "|---|---|---|"]
    cross = groups[groups.group_datasets.str.contains(",")].drop_duplicates("near_dup_group")
    for _, g in cross.head(40).iterrows():
        lines.append(f"| {g.near_dup_group} | {g.group_datasets} | {g.group_size} |")
    Path("docs/data/near_duplicates_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{int(pairs.near_duplicate.sum()):,} near-duplicate pairs, {len(groups):,} images in "
          f"{groups.near_dup_group.nunique():,} groups; {len(cross):,} groups span datasets "
          "-> docs/data/near_duplicates_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
