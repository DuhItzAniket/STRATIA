"""Segmentation baselines (P057): a small U-Net from scratch and a linear probe on frozen DINOv3 patch tokens.

Targets are the unified sky-parsing maps of P018 (0 invalid, 1 sky, 2 cloud; 255 ignored; sun-glare has no
labels in these datasets). Both models are scored with per-class IoU and the unweighted mean on the grouped val
split of each dataset, plus the collapse check of the plan: the predicted-cloud fraction on clear-sky Eye2Sky
frames (ceilometer label "none", P047 pairs), which the legacy prototype got wrong (85-99 % cloud everywhere).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ..data.image_cache import cache_path, read_rgb
from ..data.segmentation import CLOUD, IGNORE, SKY, SKY_INVALID

N_CLASSES = 3            # invalid, sky, cloud (the contract's fourth class, sun glare, has no labels here)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------------------------- data
class SegDataset(Dataset):
    """Cached image resized to size x size; mask (loaded by the caller into `masks`) resized with nearest neighbour."""

    def __init__(self, files: list[str], masks: list[np.ndarray], cache_root, size: int, train: bool, seed: int = 0):
        self.files, self.masks, self.cache_root, self.size, self.train = list(files), masks, cache_root, size, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        img = read_rgb(cache_path(self.cache_root, self.files[i]))
        m = self.masks[i]
        if self.train and self.rng.uniform() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        img = cv2.resize(np.ascontiguousarray(img), (self.size, self.size), interpolation=cv2.INTER_AREA)
        m = cv2.resize(np.ascontiguousarray(m), (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), torch.from_numpy(m.astype(np.int64))


# ----------------------------------------------------------------------------------------------- U-Net
class _Block(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    """Four-level U-Net, base width 32 (1.9 M parameters): the 'legacy U-Net' slot of the plan, trained from scratch."""

    def __init__(self, n_classes: int = N_CLASSES, base: int = 32):
        super().__init__()
        w = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([_Block(3, w[0]), _Block(w[0], w[1]), _Block(w[1], w[2]), _Block(w[2], w[3])])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[3], w[2], 2, 2), nn.ConvTranspose2d(w[2], w[1], 2, 2),
                                 nn.ConvTranspose2d(w[1], w[0], 2, 2)])
        self.dec = nn.ModuleList([_Block(w[3], w[2]), _Block(w[2], w[1]), _Block(w[1], w[0])])
        self.head = nn.Conv2d(w[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


# ----------------------------------------------------------------------------------------------- metrics
@dataclass
class SegScores:
    iou: dict[str, float]
    miou: float
    cloud_fraction: float


def confusion(pred: np.ndarray, true: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    keep = true != IGNORE
    return np.bincount(n * true[keep].astype(int) + pred[keep].astype(int), minlength=n * n).reshape(n, n)


def scores_from_confusion(cm: np.ndarray) -> SegScores:
    """Per-class IoU and their mean over the classes present in the truth (a class with no true pixels is left
    out whatever is predicted: the SWIMSEG family has no invalid pixels, and a handful of predicted ones must not
    halve a dataset's mIoU); the predicted-cloud fraction is the collapse number."""
    names = {SKY_INVALID: "invalid", SKY: "sky", CLOUD: "cloud"}
    iou = {}
    for c, name in names.items():
        tp = cm[c, c]
        denom = cm[c, :].sum() + cm[:, c].sum() - tp
        if cm[c, :].sum() == 0:
            continue
        iou[name] = float(tp / denom) if denom else 0.0
    cloud_frac = float(cm[:, CLOUD].sum() / max(cm.sum(), 1))
    return SegScores(iou, float(np.mean(list(iou.values()))) if iou else 0.0, cloud_frac)


# ----------------------------------------------------------------------------------------------- patch probe
class TokenProbe(nn.Module):
    """Linear classifier on frozen patch tokens (one logit set per token)."""

    def __init__(self, dim: int, n_classes: int = N_CLASSES):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, tokens):                      # (N, L, D) -> (N, L, C)
        return self.fc(self.norm(tokens))


def mask_to_tokens(mask: np.ndarray, side: int) -> np.ndarray:
    """Majority label per patch cell (IGNORE where no labelled pixel): (H, W) -> (side, side)."""
    h, w = mask.shape
    out = np.full((side, side), IGNORE, dtype=np.uint8)
    ys = (np.arange(side + 1) * h) // side
    xs = (np.arange(side + 1) * w) // side
    for i in range(side):
        for j in range(side):
            cell = mask[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            cell = cell[cell != IGNORE]
            if cell.size:
                out[i, j] = np.bincount(cell, minlength=N_CLASSES)[:N_CLASSES].argmax()
    return out


def train_token_probe(tokens: np.ndarray, labels: np.ndarray, dim: int, device: str = "cpu", epochs: int = 30, lr: float = 1e-3,
                      batch: int = 8192, seed: int = 0) -> TokenProbe:
    """`tokens` (M, D) float16/32 with per-token `labels` (M,) in 0..2 (IGNORE rows must be removed beforehand)."""
    torch.manual_seed(seed)
    model = TokenProbe(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x = torch.from_numpy(np.asarray(tokens, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    n = len(x)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            xb, yb = x[idx].to(device), y[idx].to(device)
            loss = nn.functional.cross_entropy(model(xb[:, None, :])[:, 0, :], yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def probe_predict(model: TokenProbe, tokens: np.ndarray, side: int, out_size: int, device: str = "cpu") -> np.ndarray:
    """(N, L, D) tokens -> (N, out_size, out_size) class maps (bilinear upsampling of the logits)."""
    x = torch.from_numpy(np.asarray(tokens, dtype=np.float32)).to(device)
    logits = model(x).permute(0, 2, 1).reshape(len(x), N_CLASSES, side, side)
    up = nn.functional.interpolate(logits, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return up.argmax(1).cpu().numpy().astype(np.uint8)
