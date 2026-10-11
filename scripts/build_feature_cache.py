"""Build frozen-feature caches (P051).

    python scripts/build_feature_cache.py --backbones dinov3_vits16 --size 512 [--patches seg] [--cbh] [--batch 64]

Rows = every manifest image (P021) plus, with --cbh, the 10-minute-cadence daytime pairs of P047 (their 768 px
cache entries come from scripts/cache_cbh_frames.py). --patches seg stores the patch-token grid for the dense
subsets (segmentation index + Almería); --patches cbh for the cloud-base-height frames; --patches all for everything
(check the disk budget first: ViT-S at 512 is 0.79 MB per frame). Appends to docs/data/feature_cache.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.registry import load_paths  # noqa: E402
from stratia.features.backbones import BACKBONES, FrozenBackbone  # noqa: E402
from stratia.features.cache import FeatureCache, list_caches  # noqa: E402

DOC = Path("docs/data/feature_cache.md")


def cbh_subset(cadence_min: int = 10) -> pd.DataFrame:
    p = Path("data/cbh_pairs.parquet")
    if not p.exists():
        return pd.DataFrame(columns=["sample_id", "image_file"])
    d = pd.read_parquet(p, columns=["image_file", "utc", "paired", "daytime", "zenith_inside_mask"])
    d = d[d.paired & d.daytime & d.zenith_inside_mask & (d.utc.dt.minute % cadence_min == 0)]
    return pd.DataFrame({"sample_id": "eye2sky:" + d.image_file, "image_file": d.image_file}).reset_index(drop=True)


def rows_for(manifest: pd.DataFrame, with_cbh: bool, patches: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all rows, rows that get patch tokens)."""
    rows = manifest[["sample_id", "image_file", "dataset"]].copy()
    if with_cbh:
        extra = cbh_subset()
        extra["dataset"] = "eye2sky_cbh"
        rows = pd.concat([rows, extra[~extra.sample_id.isin(rows.sample_id)]], ignore_index=True)
    if patches == "all":
        dense = rows
    elif patches == "seg":
        seg = pd.read_parquet("data/segmentation_index.parquet", columns=["image_file"])
        dense = rows[rows.image_file.isin(seg.image_file)]
    elif patches == "cbh":
        dense = rows[rows.dataset == "eye2sky_cbh"]
    else:
        dense = rows.iloc[0:0]
    return rows, dense


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["dinov3_vits16"], choices=sorted(BACKBONES))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--patches", choices=["none", "seg", "cbh", "all"], default="none")
    ap.add_argument("--cbh", action="store_true", help="include the 10-min cloud-base-height frames (P047)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.perf_counter()
    paths = load_paths()
    cache_root = Path(paths["cache_root"])
    manifest = pd.read_parquet("data/manifest.parquet", columns=["sample_id", "image_file", "dataset"])
    rows, dense = rows_for(manifest, a.cbh, a.patches)
    n_cbh = int((rows.dataset == "eye2sky_cbh").sum())
    print(f"{len(rows):,} rows ({n_cbh:,} CBH frames), {len(dense):,} with patch tokens", flush=True)
    for key in a.backbones:
        spec = BACKBONES[key]
        size = spec.fixed_size or a.size
        if spec.fixed_size and a.size != spec.fixed_size:
            print(f"{key}: fixed input {spec.fixed_size}, ignoring --size {a.size}")
        backbone = FrozenBackbone(key)
        pooled = FeatureCache(cache_root, key, size)
        pooled.build(rows.sample_id.tolist(), rows.image_file.tolist(), backbone, cache_root, patches=False, batch=a.batch,
                     workers=a.workers)
        if len(dense):
            grid = FeatureCache(cache_root, key, size, transform=f"ffr1-patches-{a.patches}")
            grid.build(dense.sample_id.tolist(), dense.image_file.tolist(), backbone, cache_root, patches=True,
                       batch=max(8, a.batch // 4), workers=a.workers)
        del backbone
        import torch

        torch.cuda.empty_cache()
    table = list_caches(cache_root)
    lines = ["# Frozen-feature caches (P051)", "",
             "Under `<cache_root>/features/`; built by `scripts/build_feature_cache.py`. Transform `ffr1` = full-frame "
             "resize of the P022 cached image (no crop, no mask zeroing). A fingerprint ties every cache to its row list, "
             "model and preprocessing.", "",
             "| Cache | Backbone | Size | Rows | Dim | Patch tokens | Disk | Build | Fingerprint |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, m in table.iterrows():
        tokens = int(m.patch_tokens) if pd.notna(m.patch_tokens) else "-"
        lines.append(f"| `{m.key}` | {m.hf_id} | {m['size']} | {int(m.n):,} | {int(m.dim)} | {tokens} | "
                     f"{m.disk_bytes / 2**30:.2f} GB | {m.seconds:.0f} s | {m.fingerprint} |")
    total = table.disk_bytes.sum() / 2**30 if len(table) else 0.0
    lines += ["", f"Total on disk: {total:.2f} GB."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s; {len(table)} caches, {total:.2f} GB -> {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
