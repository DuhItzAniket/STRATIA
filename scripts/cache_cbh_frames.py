"""Add the cloud-base-height frames to the 768 px image cache (P051 helper for P058/P075).

    python scripts/cache_cbh_frames.py [--cadence-min 10] [--workers 10]

Selects the daytime, paired, zenith-in-mask frames of data/cbh_pairs.parquet on a `cadence-min` grid and writes
their P022 cache entries (<cache_root>/img768/...), skipping those already present. Full-resolution decoding is
CPU-bound, so this uses processes.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.image_cache import cache_path, write_cached  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402


def _job(src: str, dst: str) -> str:
    write_cached(src, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cadence-min", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    t0 = time.perf_counter()
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    d = pd.read_parquet("data/cbh_pairs.parquet", columns=["image_file", "utc", "paired", "daytime", "zenith_inside_mask"])
    d = d[d.paired & d.daytime & d.zenith_inside_mask & (d.utc.dt.minute % a.cadence_min == 0)]
    jobs = [(str(data_root / f), str(cache_path(cache_root, f))) for f in d.image_file if not cache_path(cache_root, f).exists()]
    print(f"{len(d):,} frames on the {a.cadence_min}-min cadence, {len(jobs):,} to cache", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(_job, src, dst) for src, dst in jobs]
        for _ in as_completed(futures):
            done += 1
            if done % 2000 == 0:
                print(f"  {done:,}/{len(jobs):,} ({done / (time.perf_counter() - t0):.0f} frames/s)", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f} s: {done:,} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
