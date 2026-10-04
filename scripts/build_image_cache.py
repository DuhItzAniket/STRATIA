"""Build the resized image cache for every manifest row (P022).

    python scripts/build_image_cache.py [--workers 12] [--datasets eye2sky b0268 ...]

Skips files already cached. Uses processes (Windows: spawn) because decoding is CPU-bound.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.image_cache import cache_path, write_cached  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402


def _job(args: tuple[str, str]) -> str | None:
    src, dst = args
    try:
        write_cached(src, dst)
        return None
    except Exception as e:  # noqa: BLE001 - report and continue
        return f"{src}: {type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--datasets", nargs="*")
    a = ap.parse_args()
    paths = load_paths()
    root, cache = Path(paths["data_root"]), Path(paths["cache_root"])
    m = pd.read_parquet("data/manifest.parquet", columns=["dataset", "image_file"])
    if a.datasets:
        m = m[m.dataset.isin(a.datasets)]
    jobs = [(str(root / f), str(cache_path(cache, f))) for f in m.image_file if not cache_path(cache, f).exists()]
    print(f"{len(m):,} manifest images, {len(jobs):,} to cache -> {cache}", flush=True)
    t0 = time.perf_counter()
    errors = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, err in enumerate(ex.map(_job, jobs, chunksize=64), 1):
            if err:
                errors.append(err)
            if k % 5000 == 0:
                print(f"  {k:,}/{len(jobs):,} ({k / (time.perf_counter() - t0):.0f} img/s)", flush=True)
    dt = time.perf_counter() - t0
    size_gb = sum(p.stat().st_size for p in (cache / "img768").rglob("*.jpg")) / 1e9
    print(f"done in {dt:.0f} s ({len(jobs) / max(dt, 1e-9):.0f} img/s); errors: {len(errors)}; cache size {size_gb:.2f} GB")
    for e in errors[:20]:
        print("  ", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
