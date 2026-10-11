"""Segmentation baselines (P057): small U-Net from scratch, DINOv3 patch-token linear probe, collapse check.

    python scripts/baseline_segmentation.py [--epochs 10] [--seeds 0] [--size 256] [--skip-unet] [--skip-probe]

Data: the P018 segmentation index (SWIMSEG family + Almería) on the grouped in-domain splits (train -> val; test
locked). U-Net: 256 px, cross-entropy with 255 ignored, AdamW + one-cycle, per-epoch val mIoU. Probe: linear
classifier on the frozen DINOv3 ViT-S/16 patch tokens at 512 px (P051 cache `dinov3_vits16_512_ffr1-patches-seg`,
32 x 32 tokens), token labels by patch majority, logits upsampled to 512 px for scoring. Collapse check: the
predicted-cloud fraction on clear-sky Eye2Sky frames (P047 pairs with label "none", confidence >= 0.9), inside the
camera mask. SAM 3 is deferred (separate Python 3.12 / torch >= 2.7 environment).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.baselines import segmentation as S  # noqa: E402
from stratia.baselines.segmentation import camera_masks, clear_sky_frames, load_sky_masks, seg_tables  # noqa: E402
from stratia.data.image_cache import cache_path, read_rgb  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.data.segmentation import IGNORE  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/baseline_segmentation.md")
OUT = Path("data/baseline_segmentation.parquet")
SEG_DATASETS = ("swimseg", "swinseg", "swinyseg", "shwimseg", "almeria")


# ----------------------------------------------------------------------------------------------- U-Net
def train_unet(tab: pd.DataFrame, masks: list[np.ndarray], cache_root: Path, size: int, epochs: int, seed: int, workers: int,
               run: Run, device: str) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    tr = tab.split == "train"
    train_masks = [m for m, k in zip(masks, tr, strict=True) if k]
    ds_tr = S.SegDataset(tab[tr].image_file.tolist(), train_masks, cache_root, size, True, seed)
    dl = DataLoader(ds_tr, batch_size=16, shuffle=True, num_workers=workers, pin_memory=device == "cuda", drop_last=True,
                    persistent_workers=workers > 0)
    model = S.SmallUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    mean, std = S.IMAGENET_MEAN.to(device), S.IMAGENET_STD.to(device)
    best, best_state = None, None
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in dl:
            x = (x.to(device).float() / 255.0 - mean) / std
            y = y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = nn.functional.cross_entropy(model(x).float(), y, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss) * len(y)
            n += len(y)
        val = score_unet(model, tab, masks, cache_root, size, workers, device)
        rec = {"train_loss": total / max(n, 1), "val_miou": val["all"].miou, "seconds": time.perf_counter() - t0}
        run.log(epoch + 1, **rec)
        print(f"  unet epoch {epoch + 1}/{epochs}: loss {rec['train_loss']:.3f} val mIoU {rec['val_miou']:.3f}", flush=True)
        if best is None or val["all"].miou > best["all"].miou:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), best


@torch.no_grad()
def score_unet(model, tab, masks, cache_root, size, workers, device) -> dict[str, S.SegScores]:
    va = tab.split == "val"
    rows = tab[va].reset_index(drop=True)
    ds = S.SegDataset(rows.image_file.tolist(), [m for m, k in zip(masks, va, strict=True) if k], cache_root, size, False)
    dl = DataLoader(ds, batch_size=32, num_workers=workers)
    mean, std = S.IMAGENET_MEAN.to(device), S.IMAGENET_STD.to(device)
    cms = {d: np.zeros((S.N_CLASSES, S.N_CLASSES), dtype=np.int64) for d in rows.dataset.unique()}
    i = 0
    model.eval()
    for x, y in dl:
        pred = model((x.to(device).float() / 255.0 - mean) / std).argmax(1).cpu().numpy()
        for p, t in zip(pred, y.numpy(), strict=True):
            cms[rows.dataset.iloc[i]] += S.confusion(p, t)
            i += 1
    out = {d: S.scores_from_confusion(cm) for d, cm in cms.items()}
    out["all"] = S.scores_from_confusion(sum(cms.values()))
    return out


@torch.no_grad()
def unet_cloud_fraction(model, frames: pd.DataFrame, cam_masks, cache_root, size, device) -> float:
    mean, std = S.IMAGENET_MEAN.to(device), S.IMAGENET_STD.to(device)
    fracs = []
    for f, m in zip(frames.image_file, cam_masks, strict=True):
        img = cv2.resize(read_rgb(cache_path(cache_root, f)), (size, size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(device).float() / 255.0
        pred = model((x - mean) / std).argmax(1)[0].cpu().numpy()
        fracs.append(float((pred[m] == S.CLOUD).mean()))
    return float(np.mean(fracs)) if fracs else float("nan")


# ----------------------------------------------------------------------------------------------- probe
def run_probe(tab, masks, cache_root, device, seed, tokens_per_image=256) -> tuple[S.TokenProbe, dict, int]:
    cache = FeatureCache(cache_root, "dinov3_vits16", 512, transform="ffr1-patches-seg")
    side = 32
    rows = cache.rows_for(tab.sample_id)
    patches = cache.load("patches")
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for r, m, split in zip(rows, masks, tab.split, strict=True):
        if split != "train":
            continue
        lab = S.mask_to_tokens(m, side).ravel()
        keep = np.flatnonzero(lab != IGNORE)
        pick = rng.choice(keep, min(tokens_per_image, len(keep)), replace=False) if len(keep) else keep
        xs.append(np.asarray(patches[r][pick]))
        ys.append(lab[pick])
    x, y = np.concatenate(xs), np.concatenate(ys)
    model = S.train_token_probe(x, y, dim=x.shape[1], device=device, seed=seed)
    cms = {d: np.zeros((S.N_CLASSES, S.N_CLASSES), dtype=np.int64) for d in tab.dataset.unique()}
    va = tab[tab.split == "val"]
    val_rows = rows[(tab.split == "val").to_numpy()]
    val_masks = [mm for mm, k in zip(masks, tab.split == "val", strict=True) if k]
    for (_, row), r, m in zip(va.iterrows(), val_rows, val_masks, strict=True):
        pred = S.probe_predict(model, np.asarray(patches[r])[None], side, 512, device)[0]
        truth = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST)
        cms[row.dataset] += S.confusion(pred, truth)
    out = {d: S.scores_from_confusion(cm) for d, cm in cms.items()}
    out["all"] = S.scores_from_confusion(sum(cms.values()))
    return model, out, len(x)


@torch.no_grad()
def probe_cloud_fraction(model, frames, cam_masks, cache_root, device) -> float:
    from stratia.features.backbones import FrozenBackbone

    bb = FrozenBackbone("dinov3_vits16")
    fracs = []
    for f, m in zip(frames.image_file, cam_masks, strict=True):
        img = cv2.resize(read_rgb(cache_path(cache_root, f)), (512, 512), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.transpose(2, 0, 1))[None]
        tokens = bb.forward(x, patches=True)["patches"].float().cpu().numpy()
        pred = S.probe_predict(model, tokens, 32, 512, device)[0]
        fracs.append(float((pred[m] == S.CLOUD).mean()))
    del bb
    torch.cuda.empty_cache()
    return float(np.mean(fracs)) if fracs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--clear-frames", type=int, default=200)
    ap.add_argument("--skip-unet", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--phase", default="P057")
    a = ap.parse_args()
    t0 = time.perf_counter()
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.8)
    tab = seg_tables(data_root)
    masks = load_sky_masks(tab, data_root)
    clear = clear_sky_frames(a.clear_frames)
    cam_masks = camera_masks(clear, data_root, a.size) if len(clear) else []
    cam_masks_512 = camera_masks(clear, data_root, 512) if len(clear) else []
    print(f"{len(tab):,} labelled images ({(tab.split == 'train').sum():,} train / {(tab.split == 'val').sum():,} val), "
          f"{len(clear)} clear-sky frames for the collapse check", flush=True)
    results = []
    if not a.skip_unet:
        for seed in a.seeds:
            cfg = OmegaConf.create({"model": "small_unet", "size": a.size, "epochs": a.epochs, "seed": seed, "batch": 16,
                                    "lr": 1e-3})
            with Run(cfg, name=f"seg_unet_s{seed}", phase=a.phase) as run:
                model, best = train_unet(tab, masks, cache_root, a.size, a.epochs, seed, a.workers, run, device)
                frac = unet_cloud_fraction(model, clear, cam_masks, cache_root, a.size, device) if len(clear) else float("nan")
                for d, sc in best.items():
                    results.append({"model": "small_unet", "seed": seed, "dataset": d, "miou": sc.miou,
                                    **{f"iou_{k}": v for k, v in sc.iou.items()},
                                    "clear_sky_cloud_fraction": frac, "run_id": run.run_id})
                run.finish(val_miou=best["all"].miou, clear_sky_cloud_fraction=frac)
            print(f"unet seed {seed}: val mIoU {best['all'].miou:.3f}; clear-sky cloud fraction {frac:.3f}", flush=True)
    if not a.skip_probe:
        for seed in a.seeds:
            cfg = OmegaConf.create({"model": "dinov3_vits16_patch_probe", "size": 512, "seed": seed, "tokens_per_image": 256})
            with Run(cfg, name=f"seg_probe_s{seed}", phase=a.phase) as run:
                model, sc_all, n_tok = run_probe(tab, masks, cache_root, device, seed)
                frac = probe_cloud_fraction(model, clear, cam_masks_512, cache_root, device) if len(clear) else float("nan")
                for d, sc in sc_all.items():
                    results.append({"model": "dinov3_vits16_patch_probe", "seed": seed, "dataset": d, "miou": sc.miou,
                                    **{f"iou_{k}": v for k, v in sc.iou.items()},
                                    "clear_sky_cloud_fraction": frac, "run_id": run.run_id})
                run.finish(val_miou=sc_all["all"].miou, clear_sky_cloud_fraction=frac, train_tokens=n_tok)
            print(f"probe seed {seed}: val mIoU {sc_all['all'].miou:.3f}; clear-sky cloud fraction {frac:.3f}", flush=True)
    df = pd.DataFrame(results)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        key = ["model", "seed", "dataset"]
        df = pd.concat([old[~old.set_index(key).index.isin(df.set_index(key).index)], df], ignore_index=True)
    df.to_parquet(OUT, index=False)
    write_doc(df)
    print(f"done in {time.perf_counter() - t0:.0f} s -> {DOC}")
    return 0


def write_doc(df: pd.DataFrame) -> None:
    lines = ["# Segmentation baselines (P057)", "",
             "Sky parsing (invalid / sky / cloud) on the grouped val splits of the SWIMSEG family and Almería. U-Net: 1.9 M "
             "parameters from scratch at 256 px. Probe: linear classifier on frozen DINOv3 ViT-S/16 patch tokens at 512 px, "
             "logits upsampled. "
             "Collapse check: mean predicted-cloud fraction inside the camera mask on clear-sky Eye2Sky frames (ceilometer "
             "label none, confidence >= 0.9); the legacy prototype predicted 85-99 % cloud on such frames. SAM 3 deferred.", "",
             "| Model | Seed | Dataset | mIoU | IoU invalid | IoU sky | IoU cloud | Clear-sky cloud fraction |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values(["model", "seed", "dataset"]).iterrows():
        cells = " | ".join(f"{r[c]:.3f}" if c in r and pd.notna(r[c]) else "-" for c in ("iou_invalid", "iou_sky", "iou_cloud"))
        lines.append(f"| {r.model} | {r.seed} | {r.dataset} | {r.miou:.3f} | {cells} | {r.clear_sky_cloud_fraction:.3f} |")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
