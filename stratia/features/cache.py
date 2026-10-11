"""Versioned frozen-feature cache (P051).

One cache per (backbone, input size, transform version) under ``<cache_root>/features/<backbone>_<size>_<transform>/``:

    meta.json        backbone id, size, transform, dims, n, fingerprint, disk bytes, timings, git sha
    index.parquet    row -> sample_id, image_file
    cls.npy          float16 (N, D)   the backbone's pooled embedding
    mean.npy         float16 (N, D)   mean of the patch tokens
    patches.npy      float16 (N, L, D) memmap, only when built with patches=True (dense subsets)
    progress.json    rows completed, so an interrupted build resumes

The fingerprint hashes the ordered sample ids, the backbone's Hugging Face id, the size and the transform version:
a cache built for a different list, model or preprocessing has a different fingerprint and is never reused by
mistake. Transform ``ffr1`` = full-frame resize of the P022 cached image to size x size (INTER_AREA), no crop, no
mask zeroing (the contract's mask zeroing is a property of the final model's input, not of these baselines).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.image_cache import CachedImageDataset
from ..utils.run import _git

TRANSFORM = "ffr1"
FEATURE_SUBDIR = "features"


def fingerprint(sample_ids, hf_id: str, size: int, transform: str = TRANSFORM) -> str:
    h = hashlib.sha256()
    h.update(f"{hf_id}|{size}|{transform}\n".encode())
    for s in sample_ids:
        h.update(str(s).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


@dataclass
class CacheMeta:
    key: str
    backbone: str
    hf_id: str
    size: int
    transform: str
    n: int
    dim: int
    patch_tokens: int | None
    fingerprint: str
    created_utc: str
    git_sha: str
    seconds: float
    disk_bytes: int
    mean_dim: int | None = None      # absent in caches built before P051's CLIP fix

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


class FeatureCache:
    def __init__(self, cache_root: str | Path, backbone_key: str, size: int, transform: str = TRANSFORM):
        self.key = f"{backbone_key}_{size}_{transform}"
        self.dir = Path(cache_root) / FEATURE_SUBDIR / self.key
        self.backbone_key, self.size, self.transform = backbone_key, size, transform

    # -- state ----------------------------------------------------------------------------------------------------
    @property
    def complete(self) -> bool:
        return (self.dir / "meta.json").exists()

    def meta(self) -> dict:
        return json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))

    def index(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "index.parquet")

    def load(self, kind: str = "cls") -> np.ndarray:
        """Memory-mapped float16 array: 'cls' / 'mean' (N, D) or 'patches' (N, L, D)."""
        path = self.dir / f"{kind}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{self.key} has no {kind} (built without patches?)")
        return np.load(path, mmap_mode="r")

    def rows_for(self, sample_ids) -> np.ndarray:
        idx = self.index()
        pos = pd.Series(np.arange(len(idx)), index=idx.sample_id)
        rows = pos.reindex(list(sample_ids))
        if rows.isna().any():
            missing = rows[rows.isna()].index[:3].tolist()
            raise KeyError(f"{self.key}: {int(rows.isna().sum())} sample ids not cached, e.g. {missing}")
        return rows.to_numpy().astype(int)

    def features(self, sample_ids, kind: str = "cls") -> np.ndarray:
        """float32 (n, D) for the given sample ids, in that order."""
        rows = self.rows_for(sample_ids)
        arr = self.load(kind)
        order = np.argsort(rows)
        out = np.asarray(arr[rows[order]], dtype=np.float32)           # memmap reads are fastest in row order
        back = np.empty_like(out)
        back[order] = out
        return back

    # -- build ----------------------------------------------------------------------------------------------------
    def build(self, sample_ids: list[str], image_files: list[str], backbone, images_root: str | Path, patches: bool = False,
              batch: int = 64, workers: int = 6, resume: bool = True, log=print, pool: int = 1) -> CacheMeta:
        """Extract features for `image_files` (P022 cached images) with `backbone` (a FrozenBackbone) and store them.
        Resumes an interrupted build of the same fingerprint; refuses to overwrite a different one. `pool` > 1 stores
        the patch grid average-pooled by pool x pool (a 32 x 32 grid with pool=4 becomes 8 x 8; P067)."""
        fp = fingerprint(sample_ids, backbone.spec.hf_id, self.size, self.transform + (f"-pool{pool}" if pool > 1 else ""))
        self.dir.mkdir(parents=True, exist_ok=True)
        n, dim = len(sample_ids), backbone.dim
        mean_dim = int(getattr(backbone, "mean_dim", dim))
        side = self.size // backbone.spec.patch
        if patches and side % pool:
            raise ValueError(f"grid side {side} is not divisible by pool {pool}")
        side_out = side // pool
        n_tokens = side_out * side_out if patches else None
        progress_path = self.dir / "progress.json"
        done = 0
        if progress_path.exists():
            prog = json.loads(progress_path.read_text(encoding="utf-8"))
            if prog.get("fingerprint") != fp:
                raise RuntimeError(f"{self.dir} holds a build of another fingerprint ({prog.get('fingerprint')}); "
                                   "delete it first")
            done = int(prog.get("rows_done", 0)) if resume else 0
        if self.complete and self.meta().get("fingerprint") == fp and (not patches or (self.dir / "patches.npy").exists()):
            log(f"{self.key}: complete ({n:,} rows), nothing to do")
            return CacheMeta(**self.meta())
        mode = "r+" if done else "w+"
        cls = np.lib.format.open_memmap(self.dir / "cls.npy", mode=mode, dtype=np.float16, shape=(n, dim))
        mean = np.lib.format.open_memmap(self.dir / "mean.npy", mode=mode, dtype=np.float16, shape=(n, mean_dim))
        patch_arr = None
        if patches:
            probe = backbone.forward(torch.zeros(1, 3, self.size, self.size, dtype=torch.uint8), patches=True)
            pdim = int(probe["patches"].shape[-1])
            patch_arr = np.lib.format.open_memmap(self.dir / "patches.npy", mode=mode, dtype=np.float16,
                                                  shape=(n, n_tokens, pdim))
        pd.DataFrame({"sample_id": sample_ids, "image_file": image_files}).to_parquet(self.dir / "index.parquet", index=False)
        t0 = time.perf_counter()
        if done < n:
            ds = CachedImageDataset(image_files[done:], images_root, self.size)
            loader = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=backbone.device == "cuda",
                                persistent_workers=workers > 0)
            row = done
            last_log = time.perf_counter()
            for x, _ in loader:
                out = backbone.forward(x, patches=patches)
                k = x.shape[0]
                if not torch.isfinite(out["cls"]).all():
                    raise RuntimeError(f"{self.key}: non-finite features at rows {row}-{row + k} (fp16 overflow? use a "
                                       "bfloat16 compute dtype for this backbone)")
                cls[row:row + k] = out["cls"].cpu().numpy().astype(np.float16)
                mean[row:row + k] = out["mean"].cpu().numpy().astype(np.float16)
                if patches:
                    grid = out["patches"].float()
                    if pool > 1:
                        g = grid.transpose(1, 2).reshape(k, -1, side, side)
                        grid = torch.nn.functional.avg_pool2d(g, pool).flatten(2).transpose(1, 2)
                    patch_arr[row:row + k] = grid.cpu().numpy().astype(np.float16)
                row += k
                if time.perf_counter() - last_log > 30:
                    progress_path.write_text(json.dumps({"fingerprint": fp, "rows_done": row}), encoding="utf-8")
                    log(f"{self.key}: {row:,}/{n:,} rows, {row / (time.perf_counter() - t0):.0f} img/s")
                    last_log = time.perf_counter()
            cls.flush()
            mean.flush()
            if patches:
                patch_arr.flush()
        del cls, mean, patch_arr
        seconds = time.perf_counter() - t0
        disk = sum(p.stat().st_size for p in self.dir.glob("*.npy"))
        meta = CacheMeta(key=self.key, backbone=self.backbone_key, hf_id=backbone.spec.hf_id, size=self.size,
                         transform=self.transform, n=n, dim=dim, patch_tokens=n_tokens, mean_dim=mean_dim, fingerprint=fp,
                         created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         git_sha=_git("rev-parse", "HEAD")[:12], seconds=round(seconds, 1), disk_bytes=disk)
        (self.dir / "meta.json").write_text(meta.to_json(), encoding="utf-8")
        progress_path.unlink(missing_ok=True)
        tokens_txt = f", {n_tokens} patch tokens" if patches else ""
        log(f"{self.key}: {n:,} rows, dim {dim}{tokens_txt}, {disk / 2**30:.2f} GB, {seconds:.0f} s")
        return meta


def list_caches(cache_root: str | Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted((Path(cache_root) / FEATURE_SUBDIR).glob("*/meta.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append(m)
    return pd.DataFrame(rows)
