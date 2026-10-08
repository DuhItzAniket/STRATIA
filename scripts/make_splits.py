"""Generate the splits (P038) from configs/splits.yaml and the Stage C/D tables.

    python scripts/make_splits.py

Inputs: data/manifest.parquet, data/duplicate_groups.parquet, data/near_duplicate_groups.parquet, data/labels.parquet,
data/mask_cloud_fraction.parquet. Outputs: data/splits/{in_domain,lodo,held_out_station}.parquet,
data/splits/hashes.json, docs/data/splits_report.md. Exit code 1 if a guarantee fails (nothing is written then).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data import near_duplicates as nd  # noqa: E402
from stratia.data import splits as sp  # noqa: E402
from stratia.data.imbalance import cloud_fraction_bin  # noqa: E402


def main() -> int:
    t0 = time.perf_counter()
    config = sp.load_split_config()
    m = pd.read_parquet("data/manifest.parquet")
    labels = pd.read_parquet("data/labels.parquet").set_index("sample_id")
    dup = pd.read_parquet("data/duplicate_groups.parquet").set_index("sample_id").dup_group
    near = pd.read_parquet("data/near_duplicate_groups.parquet").set_index("sample_id").near_dup_group
    frac = pd.read_parquet("data/mask_cloud_fraction.parquet").set_index("image_file").cloud_fraction

    # ---- units ------------------------------------------------------------------------------------------------
    # Exact duplicates always share a unit. Near-duplicate links (dihedral-pHash copies, same-scene pairs at cosine
    # >= 0.97) do so only where no temporal block exists: inside a time series the block is the unit (P027), a pHash
    # match between smooth-sky frames days apart or at the other station is a coincidence, not a copy (P026), and a
    # same-scene link across days or stations is a weather look-alike (P025, P027); applied, they fuse whole days or
    # weeks into one unit (eight of nine Eye2Sky days; three units for Montenegro's ten weeks).
    g_exact = ("exact:" + m.sample_id.map(dup).astype(str)).where(m.sample_id.isin(dup.index), None)
    pairs = pd.read_parquet("data/near_duplicate_pairs.parquet", columns=["i", "j", "copy"])
    copy_pairs = pairs[pairs["copy"]]
    copy_labels = nd.connected_components(len(m), copy_pairs.i.values, copy_pairs.j.values)
    copy_keys = np.where(copy_labels >= 0, "copy:" + pd.Series(copy_labels).astype(str), None)
    g_copy = pd.Series(copy_keys, index=m.index, dtype=object)
    blocks = sp.block_keys(m, config)
    g_near = ("near:" + m.sample_id.map(near).astype(str)).where(m.sample_id.isin(near.index), None)
    if config.get("near_duplicate_links_only_without_blocks", True):
        g_copy = g_copy.where(blocks.isna(), None)
        g_near = g_near.where(blocks.isna(), None)
    units = sp.unit_ids(len(m), [g_exact, g_copy, g_near, blocks])

    # ---- strata and exclusions --------------------------------------------------------------------------------
    source = sp.source_of(m.dataset, config)
    stratum = pd.Series("all", index=m.index, dtype=object)
    for src, col in config["stratify"].items():
        rows = source == src
        if col == "primary_class":
            stratum[rows] = m.loc[rows, "source_label"].str.split(",").str[0]
        elif col == "cloud_fraction_bin":
            stratum[rows] = cloud_fraction_bin(m.loc[rows, "image_file"].map(frac).values).astype(str).values
        else:
            stratum[rows] = m.loc[rows, col].astype(str)
    stratum = stratum.fillna("unknown")
    flags = config["exclude_from_test"]["flags"]
    test_ok = ~labels.reindex(m.sample_id)[flags].fillna(False).astype(bool).any(axis=1).values
    table = pd.DataFrame({"sample_id": m.sample_id, "dataset": m.dataset, "source": source, "unit": units, "stratum": stratum,
                          "test_ok": test_ok})

    # ---- protocols --------------------------------------------------------------------------------------------
    in_domain = sp.in_domain_split(table, config)
    lodo = sp.lodo_folds(in_domain, config)
    station = sp.held_out_station(m, in_domain, config)
    problems = sp.verify(in_domain, lodo, station, m, [g_exact, g_copy, g_near], blocks)
    counts, strata = sp.summary_tables(in_domain)
    print(counts.to_string(index=False), flush=True)
    for p in problems:
        print("FAILED:", p, flush=True)
    if problems:
        return 1
    hashes = sp.split_hashes(in_domain, lodo, station)
    out = Path("data/splits")
    out.mkdir(parents=True, exist_ok=True)
    in_domain.to_parquet(out / "in_domain.parquet", index=False)
    lodo.to_parquet(out / "lodo.parquet", index=False)
    station.to_parquet(out / "held_out_station.parquet", index=False)
    (out / "hashes.json").write_text(json.dumps({"config_version": config["version"], "seed": config["seed"], "files": hashes},
                                                indent=2), encoding="utf-8")
    notes = [
        "The few-shot B0268 protocol (`few_shot_target`) is not instantiated: B0268 has one day of frames; it is generated "
        "when P041 delivers the labelled days.",
        "Eye2Sky has no image-level labels; its in-domain split (by shared dates) serves self-supervised and CBH work (P047).",
        f"Units: {in_domain.unit.nunique():,} over {len(in_domain):,} images; the largest unit has "
        f"{int(in_domain.groupby('unit').size().max()):,} images.",
    ]
    report = sp.report_markdown(counts, strata, lodo, station, problems, hashes, notes)
    Path("docs/data/splits_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: data/splits/*.parquet, data/splits/hashes.json, docs/data/splits_report.md",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
