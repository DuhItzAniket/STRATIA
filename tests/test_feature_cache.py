"""P051: the feature cache is keyed, fingerprinted, resumable and reads back in the requested order."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from stratia.features import cache as FC
from stratia.features.backbones import BACKBONES, grid_side, to_grid


@dataclass(frozen=True)
class _Spec:
    key: str = "fake"
    hf_id: str = "fake/backbone"
    patch: int = 16


class FakeBackbone:
    """Deterministic stand-in: cls = projected mean colour, patches = per-patch mean colour projected."""

    def __init__(self, dim=8):
        self.spec = _Spec()
        self.device = "cpu"
        self.dim = dim
        self.mean_dim = dim
        g = torch.Generator().manual_seed(0)
        self.proj = torch.randn(3, dim, generator=g)

    def forward(self, x_uint8, patches=True):
        x = x_uint8.float() / 255.0
        n, _, h, w = x.shape
        side = h // 16
        grid = x.reshape(n, 3, side, 16, side, 16).mean(dim=(3, 5)).permute(0, 2, 3, 1).reshape(n, side * side, 3)
        tokens = grid @ self.proj
        cls = x.mean(dim=(2, 3)) @ self.proj
        return {"cls": cls, "mean": tokens.mean(1), "patches": tokens if patches else None}


def _images(tmp_path: Path, n=12):
    """n synthetic cached images under <tmp>/img768/<file>.jpg with distinct mean colours."""
    files, ids = [], []
    for i in range(n):
        rel = f"ds/img_{i:03d}.jpg"
        p = tmp_path / "img768" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.full((96, 128, 3), (20 * i, 255 - 20 * i, 128), dtype=np.uint8)
        cv2.imwrite(str(p), img)
        files.append(rel)
        ids.append(f"fake:{rel}")
    return ids, files


def test_fingerprint_depends_on_rows_model_and_size():
    a = FC.fingerprint(["x", "y"], "m", 224)
    assert a == FC.fingerprint(["x", "y"], "m", 224)
    assert a != FC.fingerprint(["y", "x"], "m", 224) != FC.fingerprint(["x", "y"], "m2", 224)
    assert a != FC.fingerprint(["x", "y"], "m", 512) and a != FC.fingerprint(["x", "y"], "m", 224, "other")


def test_build_load_and_order(tmp_path):
    ids, files = _images(tmp_path)
    bb = FakeBackbone()
    cache = FC.FeatureCache(tmp_path, "fake", 64)
    meta = cache.build(ids, files, bb, tmp_path, patches=True, batch=5, workers=0, log=lambda *a: None)
    assert cache.complete and meta.n == 12 and meta.dim == 8 and meta.patch_tokens == 16
    assert set(p.name for p in cache.dir.glob("*")) >= {"meta.json", "index.parquet", "cls.npy", "mean.npy", "patches.npy"}
    cls = cache.load("cls")
    assert cls.shape == (12, 8) and cls.dtype == np.float16
    assert cache.load("patches").shape == (12, 16, 8)
    # features come back in the requested order, not cache order
    sub = cache.features([ids[7], ids[2], ids[9]], "cls")
    assert sub.dtype == np.float32 and np.allclose(sub[0], np.asarray(cls[7], dtype=np.float32))
    assert np.allclose(sub[1], np.asarray(cls[2], dtype=np.float32))
    with pytest.raises(KeyError):
        cache.features(["fake:missing.jpg"])
    # the values are the backbone's: recompute one image
    img = cv2.imread(str(tmp_path / "img768" / files[3]))[..., ::-1]
    x = torch.from_numpy(np.ascontiguousarray(cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)))[None]
    direct = bb.forward(x)["cls"][0].numpy()
    assert np.allclose(np.asarray(cls[3], dtype=np.float32), direct, atol=1e-2)
    # rebuilding the same fingerprint is a no-op; a different row list is refused
    again = cache.build(ids, files, bb, tmp_path, patches=True, batch=5, workers=0, log=lambda *a: None)
    assert again.fingerprint == meta.fingerprint
    with pytest.raises(RuntimeError):
        (cache.dir / "progress.json").write_text(json.dumps({"fingerprint": "other", "rows_done": 3}))
        cache.build(ids, files, bb, tmp_path, workers=0, log=lambda *a: None)


def test_resume_continues_from_progress(tmp_path):
    ids, files = _images(tmp_path, n=10)
    bb = FakeBackbone()
    cache = FC.FeatureCache(tmp_path, "fake", 64)
    fp = FC.fingerprint(ids, bb.spec.hf_id, 64)
    # simulate an interrupted build: first 4 rows written, progress recorded
    cache.dir.mkdir(parents=True)
    arr = np.lib.format.open_memmap(cache.dir / "cls.npy", mode="w+", dtype=np.float16, shape=(10, 8))
    mean = np.lib.format.open_memmap(cache.dir / "mean.npy", mode="w+", dtype=np.float16, shape=(10, 8))
    arr[:4] = 7.0
    mean[:4] = 7.0
    del arr, mean
    (cache.dir / "progress.json").write_text(json.dumps({"fingerprint": fp, "rows_done": 4}))
    cache.build(ids, files, bb, tmp_path, batch=3, workers=0, log=lambda *a: None)
    cls = cache.load("cls")
    assert np.all(np.asarray(cls[:4]) == 7.0) and not np.all(np.asarray(cls[4:]) == 7.0)   # kept + continued
    assert not (cache.dir / "progress.json").exists()
    table = FC.list_caches(tmp_path)
    assert len(table) == 1 and table.iloc[0].key == "fake_64_ffr1"


def test_backbone_registry_and_grid_helpers():
    assert set(BACKBONES) >= {"dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16", "dinov2_small", "clip_vitb16", "siglip2_b16"}
    assert BACKBONES["clip_vitb16"].fixed_size == 224 and BACKBONES["dinov3_vits16"].fixed_size is None
    assert grid_side(512, 16) == 32 and grid_side(224, 14) == 16
    g = to_grid(np.zeros((2, 1024, 5)), 512, 16)
    assert g.shape == (2, 32, 32, 5)
