"""Supervised CNN baselines (P052 legacy MobileNetV3-L, P053 ResNet-50 / ConvNeXt-T).

One generic image classifier trainer on the P022 image cache: ImageNet-pretrained timm model, 224 px, AdamW with a
cosine schedule (P053) or the legacy recipe (P052: AdamW 1e-3, ReduceLROnPlateau, early stopping), label-smoothed
cross-entropy, train-time RandomResizedCrop + horizontal flip (the legacy recipe's augmentation), full-frame resize
at validation. Every epoch logs train/val loss and accuracy and the val macro-F1 (the fit report of PLAN §4); the
best val macro-F1 epoch is kept. Test splits are never loaded. The allocator is capped (compute.md §2) so an
oversized batch fails instead of spilling into system memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..data.image_cache import cache_path, read_rgb
from . import tasks as T

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
ARCHS = {"mobilenetv3_large": "mobilenetv3_large_100", "resnet50": "resnet50", "convnext_tiny": "convnext_tiny"}


@dataclass
class TrainConfig:
    arch: str = "resnet50"
    size: int = 224
    epochs: int = 12
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    schedule: str = "cosine"            # cosine | plateau (legacy)
    patience: int = 5                   # early stopping on val macro-F1 (plateau schedule)
    seed: int = 0
    workers: int = 6
    amp: bool = True
    crop_scale: tuple[float, float] = (0.7, 1.0)
    memory_fraction: float = 0.8
    pretrained: bool = True
    extra: dict = field(default_factory=dict)


class ImageClassDataset(Dataset):
    def __init__(self, files: list[str], labels: np.ndarray, cache_root: str | Path, size: int, train: bool,
                 crop_scale=(0.7, 1.0), seed: int = 0):
        self.files, self.labels = list(files), np.asarray(labels)
        self.cache_root, self.size, self.train, self.crop_scale = Path(cache_root), size, train, crop_scale
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        img = read_rgb(cache_path(self.cache_root, self.files[i]))
        h, w = img.shape[:2]
        if self.train:
            area = h * w * self.rng.uniform(*self.crop_scale)
            ratio = float(np.exp(self.rng.uniform(np.log(3 / 4), np.log(4 / 3))))
            cw, ch = int(min(w, round(np.sqrt(area * ratio)))), int(min(h, round(np.sqrt(area / ratio))))
            x0, y0 = int(self.rng.integers(0, w - cw + 1)), int(self.rng.integers(0, h - ch + 1))
            img = img[y0:y0 + ch, x0:x0 + cw]
            if self.rng.uniform() < 0.5:
                img = img[:, ::-1]
        img = cv2.resize(np.ascontiguousarray(img), (self.size, self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), int(self.labels[i])


def build_model(arch: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    import timm

    return timm.create_model(ARCHS.get(arch, arch), pretrained=pretrained, num_classes=n_classes)


def _normalise(x: torch.Tensor, device) -> torch.Tensor:
    return (x.to(device, non_blocking=True).float() / 255.0 - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, classes: np.ndarray, amp: bool) -> dict:
    model.eval()
    preds, truth, loss_sum, n = [], [], 0.0, 0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        y = y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
            logits = model(_normalise(x, device))
        loss_sum += float(crit(logits.float(), y)) * len(y)
        n += len(y)
        preds.append(logits.argmax(1).cpu().numpy())
        truth.append(y.cpu().numpy())
    p, t = np.concatenate(preds), np.concatenate(truth)
    return {"loss": loss_sum / max(n, 1), "acc": T.accuracy(t, p), "macro_f1": T.macro_f1(classes[t], classes[p]),
            "pred": classes[p], "true": classes[t]}


def train_classifier(train_files, train_labels, val_files, val_labels, cache_root, cfg: TrainConfig, run=None, log=print) -> dict:
    """Train on (train_files, labels) and select on val; returns the best epoch's val metrics, the curves and the model."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg.memory_fraction)
    classes = np.array(sorted(set(train_labels)))
    to_idx = {c: i for i, c in enumerate(classes)}
    yt = np.array([to_idx[c] for c in train_labels])
    yv = np.array([to_idx.get(c, -1) for c in val_labels])
    keep = yv >= 0                                                     # val classes unseen in train are skipped
    val_files = [f for f, k in zip(val_files, keep, strict=True) if k]
    yv = yv[keep]
    ds_tr = ImageClassDataset(train_files, yt, cache_root, cfg.size, True, cfg.crop_scale, cfg.seed)
    ds_va = ImageClassDataset(val_files, yv, cache_root, cfg.size, False)
    pin = device == "cuda"
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=pin, drop_last=True,
                       persistent_workers=cfg.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.workers, pin_memory=pin,
                       persistent_workers=cfg.workers > 0)
    model = build_model(cfg.arch, len(classes), cfg.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.schedule == "cosine":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(dl_tr), pct_start=0.1)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    best, best_state, curves, bad_epochs = None, None, [], 0
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in dl_tr:
            y = y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.amp and device == "cuda"):
                logits = model(_normalise(x, device))
                loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if cfg.schedule == "cosine":
                sched.step()
            loss_sum += float(loss) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            n += len(y)
        val = evaluate(model, dl_va, device, classes, cfg.amp)
        if cfg.schedule != "cosine":
            sched.step(val["macro_f1"])
        rec = {"epoch": epoch + 1, "train_loss": loss_sum / max(n, 1), "train_acc": correct / max(n, 1), "val_loss": val["loss"],
               "val_acc": val["acc"], "val_macro_f1": val["macro_f1"], "lr": opt.param_groups[0]["lr"],
               "seconds": time.perf_counter() - t0}
        curves.append(rec)
        if run is not None:
            run.log(epoch + 1, **{k: v for k, v in rec.items() if k != "epoch"})
        log(f"  epoch {epoch + 1:2d}/{cfg.epochs}: train loss {rec['train_loss']:.3f} acc {rec['train_acc']:.3f} | "
            f"val loss {rec['val_loss']:.3f} acc {rec['val_acc']:.3f} macro-F1 {rec['val_macro_f1']:.3f} | "
            f"{rec['seconds']:.0f} s")
        if best is None or val["macro_f1"] > best["macro_f1"]:
            best = {**val, "epoch": epoch + 1}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if cfg.schedule != "cosine" and bad_epochs >= cfg.patience:
                log(f"  early stop after epoch {epoch + 1} (no val gain for {cfg.patience} epochs)")
                break
    model.load_state_dict(best_state)
    gap = curves[best["epoch"] - 1]["train_acc"] - best["acc"]
    return {"best": best, "curves": curves, "classes": classes, "model": model, "train_val_gap": gap,
            "epochs_run": len(curves), "seconds": time.perf_counter() - t0}


def fit_verdict(curves: list[dict], best_epoch: int) -> str:
    """The PLAN §4 reading of the curves: overfit (train up, val flat/down, gap > 15 pts), underfit (both low), or ok."""
    if not curves:
        return "no epochs"
    last, best = curves[-1], curves[best_epoch - 1]
    gap = best["train_acc"] - best["val_acc"]
    if gap > 0.15 and last["val_macro_f1"] <= best["val_macro_f1"] - 0.01:
        return f"overfit (train-val gap {gap:.2f}, val fell after epoch {best_epoch})"
    if gap > 0.15:
        return f"overfit-leaning (train-val gap {gap:.2f})"
    if best["train_acc"] < 0.6:
        return f"underfit (train acc {best['train_acc']:.2f})"
    return f"ok (gap {gap:.2f})"
