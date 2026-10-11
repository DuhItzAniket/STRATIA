"""Dense-head training data on cached token grids (P065, P066, P074, P076): token + mask dataset, evaluation per
dataset with per-image confusion matrices, bootstrap intervals, and the predicted-class-fraction monitors."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..baselines import segmentation as S
from ..data.image_cache import cache_path, read_rgb
from ..data.segmentation import IGNORE
from ..model.dense import DenseHeads

SIZE = 512
N_LAYER = 3
CLASS_NAMES = ("invalid", "sky", "cloud", "sun_glare")


class TokenMaskDataset(Dataset):
    def __init__(self, rows: np.ndarray, patches, sky: list[np.ndarray], layer: list[np.ndarray], train: bool, seed: int = 0):
        self.rows, self.patches, self.sky, self.layer, self.train = rows, patches, sky, layer, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        tok = np.asarray(self.patches[self.rows[i]], dtype=np.float32)          # (1024, 384)
        sky = cv2.resize(self.sky[i], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        lay = cv2.resize(self.layer[i], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        if self.train and self.rng.uniform() < 0.5:
            side = int(np.sqrt(tok.shape[0]))
            tok = tok.reshape(side, side, -1)[:, ::-1].reshape(side * side, -1)
            sky, lay = sky[:, ::-1], lay[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(tok)), torch.from_numpy(np.ascontiguousarray(sky).astype(np.int64)),
                torch.from_numpy(np.ascontiguousarray(lay).astype(np.int64)))


@torch.no_grad()
def evaluate(model: DenseHeads, dl: DataLoader, datasets: list[str], device: str, layers: bool) -> dict:
    """Per-dataset confusion matrices for sky parsing (and layers on cloud pixels), plus per-image sky mIoU."""
    model.eval()
    cm_sky = {d: np.zeros((S.N_CLASSES, S.N_CLASSES), dtype=np.int64) for d in set(datasets)}
    cm_lay = {d: np.zeros((N_LAYER, N_LAYER), dtype=np.int64) for d in set(datasets)}
    per_image = []
    k = 0
    for tok, sky, lay in dl:
        out = model(tok.to(device))
        logits = torch.nn.functional.interpolate(out["sky_parse_logits"].float(), size=(SIZE, SIZE), mode="bilinear",
                                                 align_corners=False)
        pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
        if layers:
            llog = torch.nn.functional.interpolate(out["layer_logits"].float(), size=(SIZE, SIZE), mode="bilinear",
                                                   align_corners=False)
            lpred = llog.argmax(1).cpu().numpy().astype(np.uint8)
        for j in range(len(tok)):
            d = datasets[k]
            truth = sky[j].numpy().astype(np.uint8)
            cm = S.confusion(pred[j], truth)
            cm_sky[d] += cm
            per_image.append((d, cm))
            if layers:
                lt = lay[j].numpy().astype(np.uint8)
                keep = lt != IGNORE
                if keep.any():
                    cm_lay[d] += np.bincount(N_LAYER * lt[keep].astype(int) + lpred[j][keep].astype(int),
                                             minlength=N_LAYER * N_LAYER).reshape(N_LAYER, N_LAYER)
            k += 1
    return {"sky": cm_sky, "layer": cm_lay, "per_image": per_image}


def layer_scores(cm: np.ndarray) -> dict:
    out = {}
    for c, name in enumerate(("low", "mid", "high")):
        if cm[c].sum() == 0:
            continue
        tp = cm[c, c]
        out[name] = float(tp / max(cm[c].sum() + cm[:, c].sum() - tp, 1))
    return out


def bootstrap_miou(per_image: list, dataset: str | None, n: int = 1000, seed: int = 0, scorer=None) -> tuple[float, float]:
    cms = [cm for d, cm in per_image if dataset is None or d == dataset]
    if not cms:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.stack(cms)
    vals = []
    for _ in range(n):
        pick = rng.integers(len(arr), size=len(arr))
        vals.append(scorer(arr[pick].sum(0)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


@torch.no_grad()
def class_fractions(model: DenseHeads, files: list[str], masks: list[np.ndarray] | None, cache_root: Path, device: str,
                    backbone=None) -> dict[str, float]:
    """Mean predicted-class fraction (inside the camera mask when given) over the frames: the collapse monitor."""
    from stratia.features.backbones import FrozenBackbone

    bb = backbone or FrozenBackbone("dinov3_vits16")
    fracs = []
    model.eval()
    for i, f in enumerate(files):
        img = cv2.resize(read_rgb(cache_path(cache_root, f)), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.transpose(2, 0, 1))[None]
        tokens = bb.forward(x, patches=True)["patches"].float().to(device)
        logits = torch.nn.functional.interpolate(model(tokens)["sky_parse_logits"].float(), size=(SIZE, SIZE),
                                                 mode="bilinear", align_corners=False)
        pred = logits.argmax(1)[0].cpu().numpy()
        if masks is not None:
            pred = pred[masks[i]]
        fracs.append([float((pred == c).mean()) for c in range(len(CLASS_NAMES))])
    if backbone is None:
        del bb
        torch.cuda.empty_cache()
    if not fracs:
        return {c: float("nan") for c in CLASS_NAMES}
    return dict(zip(CLASS_NAMES, np.mean(fracs, axis=0).tolist(), strict=True))


def b0268_files() -> list[str]:
    man = pd.read_parquet("data/manifest.parquet", columns=["dataset", "image_file"])
    return man[man.dataset == "b0268"].image_file.tolist()


@torch.no_grad()
def _layer_cms(model: DenseHeads, dl: DataLoader, datasets: list[str], device: str) -> list[np.ndarray]:
    """Per-image layer confusion matrices (for the bootstrap)."""
    model.eval()
    out = []
    for tok, _, lay in dl:
        llog = torch.nn.functional.interpolate(model(tok.to(device))["layer_logits"].float(), size=(SIZE, SIZE),
                                               mode="bilinear", align_corners=False)
        lpred = llog.argmax(1).cpu().numpy().astype(np.uint8)
        for j in range(len(tok)):
            lt = lay[j].numpy().astype(np.uint8)
            keep = lt != IGNORE
            cm = np.zeros((N_LAYER, N_LAYER), dtype=np.int64)
            if keep.any():
                cm = np.bincount(N_LAYER * lt[keep].astype(int) + lpred[j][keep].astype(int),
                                 minlength=N_LAYER * N_LAYER).reshape(N_LAYER, N_LAYER)
            out.append(cm)
    return out
