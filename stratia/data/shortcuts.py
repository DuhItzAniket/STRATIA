"""Shortcut audit (P029): what a model could learn instead of clouds.

Instruments
    camera statistics   per-camera mean and standard-deviation images at 224 px; pixels that never change (std far
                        below the camera's median) are *fixed structure*: fisheye corners, horizon objects, burned-in
                        text, logos, borders
    image variants      the same images re-embedded with DINOv3 after a controlled change: `lowres` (down to a few
                        dozen pixels and back: resolution and sharpness removed), `skyonly` (fixed structure painted
                        grey), `fixedonly` (everything but the fixed structure painted grey)
    linear probes       logistic regression on frozen CLS features, train/test split by day or by near-duplicate
                        group (never by image), predicting what a model should *not* need: the dataset, the camera,
                        the hour of day; and, for comparison, the labels themselves
A probe that reads the camera or the hour from features of the sky alone names a shortcut the training recipe must
blunt (masking, resolution normalisation, explicit sun input, camera-balanced sampling, out-of-camera evaluation).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from stratia.data.image_cache import CachedImageDataset
from stratia.data.near_duplicates import IMAGENET_MEAN, IMAGENET_STD

MODELS = {"vits16": "facebook/dinov3-vits16-pretrain-lvd1689m", "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m"}
GREY = 128

# ------------------------------------------------------------------------------------------ camera statistics


def camera_statistics(files: list[str], cache_root: str | Path, size: int = 224, batch: int = 256,
                      workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard-deviation images (float32, CHW, 0-255 scale) of a camera's frames at size x size."""
    loader = DataLoader(CachedImageDataset(files, cache_root, size), batch_size=batch, num_workers=workers)
    s = torch.zeros(3, size, size, dtype=torch.float64)
    s2 = torch.zeros_like(s)
    n = 0
    for x, _ in loader:
        xf = x.double()
        s += xf.sum(0)
        s2 += (xf * xf).sum(0)
        n += len(x)
    mean = s / max(n, 1)
    std = (s2 / max(n, 1) - mean * mean).clamp_min(0).sqrt()
    return mean.numpy().astype(np.float32), std.numpy().astype(np.float32)


def fixed_mask(std: np.ndarray, rel: float = 0.5) -> np.ndarray:
    """Pixels whose grey standard deviation is below `rel` times the camera's median (the parts of the frame that
    never change), plus everything outside the camera's main field of view: a changing speck inside the fixed
    surround, such as the digits of a burned-in timestamp in a fisheye corner, is not sky (HxW bool, True = fixed)."""
    grey = std.mean(axis=0)
    fixed = grey < rel * float(np.median(grey))
    n, labels = cv2.connectedComponents((~fixed).astype(np.uint8), connectivity=4)
    if n > 2:                                    # several changing regions: keep the largest as the field of view
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0                             # label 0 is the fixed set itself
        fixed = labels != int(sizes.argmax())
    return fixed


# ------------------------------------------------------------------------------------------ image variants


class VariantDataset(Dataset):
    """Cached images after a controlled change, as uint8 CHW tensors with their index.
    variant: 'original' | 'lowres' (down to `low` px and back up) | 'skyonly' (fixed pixels grey) |
    'fixedonly' (all but the fixed pixels grey); `masks` maps a camera name to its fixed mask (HxW bool at `size`),
    `cameras` gives the camera name of each file."""

    def __init__(self, files: list[str], cache_root: str | Path, variant: str = "original", size: int = 224,
                 low: int = 96, masks: dict[str, np.ndarray] | None = None, cameras: list[str] | None = None):
        self.base = CachedImageDataset(files, cache_root, size)
        self.variant, self.size, self.low = variant, size, low
        self.masks = masks or {}
        self.cameras = list(cameras) if cameras is not None else [""] * len(files)
        if variant not in {"original", "lowres", "skyonly", "fixedonly"}:
            raise ValueError(f"unknown variant {variant!r}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        x, _ = self.base[i]
        return torch.from_numpy(np.ascontiguousarray(self.apply(x.numpy(), self.cameras[i]))), i

    def apply(self, chw: np.ndarray, camera: str) -> np.ndarray:
        if self.variant == "original":
            return chw
        hwc = np.ascontiguousarray(chw.transpose(1, 2, 0))
        if self.variant == "lowres":
            small = cv2.resize(hwc, (self.low, self.low), interpolation=cv2.INTER_AREA)
            hwc = cv2.resize(small, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        else:
            mask = self.masks[camera]
            hwc = hwc.copy()
            hwc[mask if self.variant == "skyonly" else ~mask] = GREY
        return hwc.transpose(2, 0, 1)


def embed(dataset: Dataset, out: str | Path | None, model_name: str = "vits16", batch: int = 128, workers: int = 8,
          device: str = "cuda") -> np.ndarray:
    """L2-normalised DINOv3 CLS features (fp16 on the GPU, float32 out) of every item of `dataset`; cached at `out`."""
    out = Path(out) if out else None
    if out and out.exists():
        emb = np.load(out)
        if emb.shape[0] == len(dataset):
            return emb
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODELS[model_name]).to(device).eval().half()
    loader = DataLoader(dataset, batch_size=batch, num_workers=workers, pin_memory=device == "cuda",
                        persistent_workers=workers > 0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1).half()
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1).half()
    chunks = []
    with torch.no_grad():
        for x, _ in loader:
            x = (x.to(device, non_blocking=True).half() / 255.0 - mean) / std
            feats = model(pixel_values=x).pooler_output.float()
            chunks.append(torch.nn.functional.normalize(feats, dim=1).cpu().numpy())
    emb = np.concatenate(chunks).astype(np.float32)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, emb)
    return emb


# ------------------------------------------------------------------------------------------ splits and probes


def holdout_last_blocks(keys: pd.Series, frac: float = 0.3) -> np.ndarray:
    """Test mask holding out the last `frac` of the distinct, sorted block keys (days, weeks): contiguous in time."""
    distinct = np.sort(pd.unique(keys.dropna()))
    n_test = max(1, int(np.ceil(frac * len(distinct)))) if len(distinct) > 1 else 0
    test_keys = set(distinct[len(distinct) - n_test:]) if n_test else set()
    return keys.isin(test_keys).to_numpy()


def holdout_random_groups(keys: pd.Series, frac: float = 0.3, seed: int = 0) -> np.ndarray:
    """Test mask holding out a random `frac` of the distinct group keys, so a group never straddles the split."""
    distinct = pd.unique(keys.astype(str))
    rng = np.random.default_rng(seed)
    rng.shuffle(distinct)
    n_test = max(1, int(round(frac * len(distinct)))) if len(distinct) > 1 else 0
    return keys.astype(str).isin(set(distinct[:n_test])).to_numpy()


def balanced_subsample(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Indices with at most `per_class` of each label, drawn at random."""
    take = []
    for cls in pd.unique(labels):
        idx = np.flatnonzero(labels == cls)
        take.append(idx if len(idx) <= per_class else rng.choice(idx, per_class, replace=False))
    return np.sort(np.concatenate(take))


def linear_probe(X: np.ndarray, y: np.ndarray, test: np.ndarray, max_iter: int = 2000, seed: int = 0) -> dict:
    """Multinomial logistic regression on standardised features; accuracy and balanced accuracy on the test rows,
    with the chance level (1 / classes) and the majority-class share of the test rows for comparison."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    train = ~test
    scaler = StandardScaler().fit(X[train])
    clf = LogisticRegression(max_iter=max_iter, C=1.0, random_state=seed).fit(scaler.transform(X[train]), y[train])
    pred = clf.predict(scaler.transform(X[test]))
    truth = y[test]
    classes = np.unique(y)
    recalls = [float((pred[truth == c] == c).mean()) for c in classes if (truth == c).any()]
    counts = pd.Series(truth).value_counts()
    return {"classes": int(len(classes)), "n_train": int(train.sum()), "n_test": int(test.sum()),
            "accuracy": float((pred == truth).mean()), "balanced_accuracy": float(np.mean(recalls)),
            "chance": 1.0 / len(classes), "majority": float(counts.iloc[0] / counts.sum())}


# ------------------------------------------------------------------------------------------ figure and report


def statistics_figure(stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], out: str | Path, tile: int = 160) -> Path:
    """One column per camera: mean image, standard deviation (scaled to its maximum), fixed mask (white = fixed)."""
    names = list(stats)
    sheet = np.full((3 * (tile + 4) + 20, len(names) * (tile + 4), 3), 255, dtype=np.uint8)
    for c, name in enumerate(names):
        mean, std, mask = stats[name]
        tiles = [np.clip(mean, 0, 255).astype(np.uint8).transpose(1, 2, 0),
                 np.clip(std.mean(0) / max(float(std.max()), 1e-6) * 255, 0, 255).astype(np.uint8)[..., None].repeat(3, -1),
                 (mask.astype(np.uint8) * 255)[..., None].repeat(3, -1)]
        for r, t in enumerate(tiles):
            t = cv2.resize(t, (tile, tile), interpolation=cv2.INTER_AREA)
            y, x = 20 + r * (tile + 4), c * (tile + 4)
            sheet[y:y + tile, x:x + tile] = t
        cv2.putText(sheet, name[:22], (c * (tile + 4) + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise OSError(f"cannot encode {out}")
    enc.tofile(str(out))
    return out


def report_markdown(probes: pd.DataFrame, fixed_share: pd.DataFrame, shortcuts: list[dict], figure: str) -> str:
    lines = ["# Shortcut audit report (P029)", "",
             "Linear probes (logistic regression) on frozen DINOv3 ViT-S/16 CLS features, trained and tested on disjoint "
             "days or near-duplicate groups, predicting what a cloud model should not need (dataset, camera, hour of "
             "day) and, for scale, the labels themselves. Variants re-embed the same images after a controlled change: "
             "`lowres` removes resolution and sharpness, `skyonly` paints the camera's fixed structure grey, `fixedonly` "
             "paints everything else grey.", "",
             f"![camera statistics]({figure})", "",
             "## Fixed structure per camera", "",
             "Share of the 224 x 224 frame whose standard deviation over the camera's frames is below half the median "
             "(fisheye corners, horizon objects, text, logos, borders).", "",
             "| Camera | Frames | Fixed share |", "|---|---|---|"]
    for _, r in fixed_share.iterrows():
        lines.append(f"| {r.camera} | {int(r.frames):,} | {r.fixed_share:.1%} |")
    lines += ["", "## Probes", "",
              "| Task | Variant | Classes | Train | Test | Accuracy | Balanced accuracy | Chance | Majority |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in probes.iterrows():
        lines.append(f"| {r.task} | {r.variant} | {int(r.classes)} | {int(r.n_train):,} | {int(r.n_test):,} | {r.accuracy:.1%} | "
                     f"{r.balanced_accuracy:.1%} | {r.chance:.1%} | {r.majority:.1%} |")
    lines += ["", "## Shortcut list and mitigations", "",
              "| Shortcut | Evidence | Mitigation (where) |", "|---|---|---|"]
    for s in shortcuts:
        lines.append(f"| {s['shortcut']} | {s['evidence']} | {s['mitigation']} |")
    return "\n".join(lines) + "\n"
