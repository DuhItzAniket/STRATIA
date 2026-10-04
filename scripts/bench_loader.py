"""Data-loading benchmark (P022): images/s delivered to the GPU as 512 x 512 uint8 batches.

    python scripts/bench_loader.py [--n 3000] [--workers 0 4 8]

Compares the resized cache with the original files (Eye2Sky 2112 x 2048 JPEGs) on Windows DataLoader workers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.image_cache import CachedImageDataset, read_rgb  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402


class OriginalImageDataset(Dataset):
    def __init__(self, files: list[str], root: str, size: int = 512):
        self.files, self.root, self.size = files, Path(root), size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        img = cv2.resize(read_rgb(self.root / self.files[i]), (self.size, self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), i


def bench(ds: Dataset, workers: int, batch: int = 32) -> float:
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True,
                    persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    it = iter(dl)
    x, _ = next(it)                                   # warm-up (worker start-up excluded)
    x.cuda(non_blocking=True)
    torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    for x, _ in it:
        x = x.cuda(non_blocking=True).float().div_(255)
        n += x.shape[0]
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n-original", type=int, default=600)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8])
    a = ap.parse_args()
    paths = load_paths()
    m = pd.read_parquet("data/manifest.parquet", columns=["dataset", "image_file"])
    files = m[m.dataset == "eye2sky"].image_file.sample(a.n, random_state=0).tolist()
    res = {"cache": {}, "original": {}}
    for w in a.workers:
        res["cache"][w] = round(bench(CachedImageDataset(files, paths["cache_root"]), w), 1)
        res["original"][w] = round(bench(OriginalImageDataset(files[: a.n_original], paths["data_root"]), w), 1)
        print(f"workers={w}: cache {res['cache'][w]} img/s | original {res['original'][w]} img/s", flush=True)
    Path("docs/data/loader_benchmark.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
