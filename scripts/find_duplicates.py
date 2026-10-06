"""Hash every manifest image (file bytes and decoded pixels) and report exact duplicates (P024).

    python scripts/find_duplicates.py [--workers 12]

Writes data/image_hashes.parquet, data/duplicate_groups.parquet and docs/data/duplicates_report.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.duplicates import COLUMNS, duplicate_groups, hash_image, report_markdown  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    root = Path(load_paths()["data_root"])
    m = pd.read_parquet("data/manifest.parquet", columns=["sample_id", "dataset", "image_file"])
    paths = [str(root / f) for f in m.image_file]
    print(f"{len(paths):,} images to hash", flush=True)
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(hash_image, paths, chunksize=64), 1):
            results.append(r)
            if k % 5000 == 0:
                print(f"  {k:,}/{len(paths):,} ({k / (time.perf_counter() - t0):.0f} img/s)", flush=True)
    table = pd.concat([m.reset_index(drop=True), pd.DataFrame(results)], axis=1)[COLUMNS]
    table.to_parquet("data/image_hashes.parquet", index=False)
    groups = duplicate_groups(table)
    groups.to_parquet("data/duplicate_groups.parquet", index=False)
    Path("docs/data/duplicates_report.md").write_text(report_markdown(table, groups), encoding="utf-8")
    n_groups = groups["dup_group"].nunique() if len(groups) else 0
    print(f"done in {time.perf_counter() - t0:.0f} s; {len(groups):,} images in {n_groups:,} duplicate groups "
          "-> data/duplicate_groups.parquet, docs/data/duplicates_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
