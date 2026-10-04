"""Resized image cache and a manifest-backed image dataset (P022).

Cache: <cache_root>/img768/<image_file with .jpg suffix>, longest side 768 px (aspect preserved, JPEG quality 95).
768 rather than 512 keeps headroom for crop/scale augmentation (P048) before the final 512 x 512 model input.
OpenCV reads via np.fromfile/imdecode so Windows paths with spaces or non-ASCII characters work.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CACHE_SUBDIR = "img768"
CACHE_MAX_SIDE = 768
JPEG_QUALITY = 95


def cache_path(cache_root: str | Path, image_file: str) -> Path:
    return Path(cache_root) / CACHE_SUBDIR / Path(image_file).with_suffix(".jpg")


def read_rgb(path: str | Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"cannot decode {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_cached(src: str | Path, dst: str | Path, max_side: int = CACHE_MAX_SIDE) -> tuple[int, int]:
    """Resize `src` so its longest side is at most `max_side` and write it as JPEG to `dst`. Returns (w, h)."""
    img = read_rgb(src)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise OSError(f"cannot encode {dst}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    enc.tofile(str(dst))
    return img.shape[1], img.shape[0]


class CachedImageDataset(Dataset):
    """Images from the cache, resized to size x size (full frame, no crop: stratia-contract v1), as uint8 CHW tensors."""

    def __init__(self, image_files: list[str], cache_root: str | Path, size: int = 512):
        self.files = list(image_files)
        self.cache_root = Path(cache_root)
        self.size = size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img = read_rgb(cache_path(self.cache_root, self.files[i]))
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), i
