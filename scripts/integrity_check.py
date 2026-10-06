"""Decode every manifest image and write data/integrity.parquet and docs/data/integrity_report.md (P023).

    python scripts/integrity_check.py [--workers 12] [--datasets ccsn mgcd ...]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.integrity import COLUMNS, add_size_outliers, inspect_image, report_markdown  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402


def _job(args: tuple[str, int, int]) -> dict:
    return inspect_image(*args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--out", default="data/integrity.parquet")
    ap.add_argument("--report", default="docs/data/integrity_report.md")
    a = ap.parse_args()
    root = Path(load_paths()["data_root"])
    m = pd.read_parquet("data/manifest.parquet", columns=["sample_id", "dataset", "image_file", "width", "height"])
    if a.datasets:
        m = m[m.dataset.isin(a.datasets)]
    jobs = [(str(root / f), int(w), int(h)) for f, w, h in zip(m.image_file, m.width, m.height, strict=True)]
    print(f"{len(jobs):,} images to decode", flush=True)
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(_job, jobs, chunksize=64), 1):
            results.append(r)
            if k % 5000 == 0:
                print(f"  {k:,}/{len(jobs):,} ({k / (time.perf_counter() - t0):.0f} img/s)", flush=True)
    table = pd.concat([m[["sample_id", "dataset", "image_file"]].reset_index(drop=True), pd.DataFrame(results)], axis=1)
    table = add_size_outliers(table)[COLUMNS]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(a.out, index=False)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(report_markdown(table), encoding="utf-8")
    flagged = int((table["flags"] != "").sum())
    print(f"done in {time.perf_counter() - t0:.0f} s; {flagged:,} flagged of {len(table):,} -> {a.out}, {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
