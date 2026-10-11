"""Train the dense heads on cached DINOv3 tokens (P065 sky parsing; P066 + layers).

    python scripts/train_dense_heads.py --phase P065 --epochs 15                # sky parsing only
    python scripts/train_dense_heads.py --phase P066 --epochs 15 --layers       # + layer head on Almería

Data: the P018 segmentation index on the grouped splits (train -> val; test locked), tokens from the P051 cache
`dinov3_vits16_512_ffr1-patches-seg` (32 x 32 x 384 per image), masks resized to 512 (nearest). Model: the P065
DenseDecoder + SkyParseHead (+ LayerHead), losses dense CE with ignore 255 (ADR-006 §5), logits upsampled from
128 to 512 for the loss and the scores. Augmentation on cached tokens: horizontal flip of the token grid and the
mask (p = 0.5). Scores: per-dataset mIoU over the classes present (as P057), bootstrap 95 % intervals over val
images, the clear-sky collapse check on 200 Eye2Sky frames (backbone run live), and the comparison with the P057
token probe. Results: data/dense_heads.parquet, docs/data/dense_heads.md, one registry run per invocation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.baselines import segmentation as S  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.model import losses as L  # noqa: E402
from stratia.model.balance import UncertaintyWeighting  # noqa: E402
from stratia.model.dense import DenseHeads  # noqa: E402
from stratia.train.dense_data import (  # noqa: E402
    SIZE,
    TokenMaskDataset,
    _layer_cms,  # noqa: E402
    b0268_files,
    bootstrap_miou,
    class_fractions,
    evaluate,
    layer_scores,
)
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/dense_heads.md")
OUT = Path("data/dense_heads.parquet")














def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="P065")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)   # >0 pickles the mask lists to spawned workers (fails on Windows)
    ap.add_argument("--layers", action="store_true", help="train the layer head too (P066)")
    ap.add_argument("--layer-weight", type=float, default=1.0)
    ap.add_argument("--clear-frames", type=int, default=200)
    ap.add_argument("--kendall", action="store_true", help="P070 uncertainty weighting of the sky / layer losses")
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.8)
    tab = S.seg_tables(data_root)
    sky_masks, layer_masks = S.load_both_masks(tab, data_root)
    cache = FeatureCache(cache_root, "dinov3_vits16", 512, transform="ffr1-patches-seg")
    rows = cache.rows_for(tab.sample_id)
    patches = cache.load("patches")
    tr = (tab.split == "train").to_numpy()
    va = ~tr
    ds_tr = TokenMaskDataset(rows[tr], patches, [m for m, k in zip(sky_masks, tr, strict=True) if k],
                             [m for m, k in zip(layer_masks, tr, strict=True) if k], True, a.seed)
    ds_va = TokenMaskDataset(rows[va], patches, [m for m, k in zip(sky_masks, va, strict=True) if k],
                             [m for m, k in zip(layer_masks, va, strict=True) if k], False)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                       pin_memory=device == "cuda")
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=a.workers)
    val_datasets = tab.dataset[va].tolist()
    print(f"{tr.sum():,} train / {va.sum():,} val images; layers={'on' if a.layers else 'off'}", flush=True)

    model = DenseHeads(d_model=patches.shape[-1]).to(device)
    weighting = UncertaintyWeighting(["sky", "layer"]).to(device) if (a.kendall and a.layers) else None
    n_params = sum(p.numel() for p in model.parameters())
    params = list(model.parameters()) + (list(weighting.parameters()) if weighting is not None else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    cfg = OmegaConf.create({"model": "dense_heads", "layers": a.layers, "epochs": a.epochs, "batch": a.batch, "lr": a.lr,
                            "seed": a.seed, "layer_weight": a.layer_weight, "kendall": a.kendall, "params": n_params,
                            "cache_fingerprint": cache.meta()["fingerprint"]})
    best, best_state, curves = None, None, []
    with Run(cfg, name=f"dense_{'sky_layer' if a.layers else 'sky'}_s{a.seed}", phase=a.phase) as run:
        for epoch in range(a.epochs):
            model.train()
            total, n = 0.0, 0
            for tok, sky, lay in dl_tr:
                tok, sky, lay = tok.to(device, non_blocking=True), sky.to(device), lay.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                    out = model(tok)
                sky_loss = L.dense_ce(out["sky_parse_logits"].float(), sky)
                if a.layers and weighting is not None:
                    loss, _ = weighting({"sky": sky_loss, "layer": L.dense_ce(out["layer_logits"].float(), lay)})
                elif a.layers:
                    loss = sky_loss + a.layer_weight * L.dense_ce(out["layer_logits"].float(), lay)
                else:
                    loss = sky_loss
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += float(loss) * len(tok)
                n += len(tok)
            ev = evaluate(model, dl_va, val_datasets, device, a.layers)
            sc_all = S.scores_from_confusion(sum(ev["sky"].values()))
            rec = {"epoch": epoch + 1, "train_loss": total / max(n, 1), "val_miou": sc_all.miou,
                   "val_iou_cloud": sc_all.iou.get("cloud", float("nan")), "seconds": time.perf_counter() - t0}
            if a.layers:
                lsc = layer_scores(sum(ev["layer"].values()))
                rec["val_layer_miou"] = float(np.mean(list(lsc.values()))) if lsc else float("nan")
            curves.append(rec)
            run.log(epoch + 1, **{k: v for k, v in rec.items() if k != "epoch"})
            print(f"  epoch {epoch + 1:2d}/{a.epochs}: loss {rec['train_loss']:.3f} | val mIoU {sc_all.miou:.3f}"
                  + (f" | layer mIoU {rec['val_layer_miou']:.3f}" if a.layers else "") + f" | {rec['seconds']:.0f} s", flush=True)
            if best is None or sc_all.miou > best["val_miou"]:
                best = dict(rec, ev=ev)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        ev = best["ev"]
        clear = S.clear_sky_frames(a.clear_frames)
        cam_masks = S.camera_masks(clear, data_root, SIZE) if len(clear) else []
        from stratia.features.backbones import FrozenBackbone
        bb = FrozenBackbone("dinov3_vits16")
        clear_frac = class_fractions(model, clear.image_file.tolist(), cam_masks, cache_root, device, bb) if len(clear) else {}
        b0268_frac = class_fractions(model, b0268_files(), None, cache_root, device, bb)
        del bb
        torch.cuda.empty_cache()
        frac = clear_frac.get("cloud", float("nan"))
        if weighting is not None:
            print(f"  kendall weights: {weighting.weights()}", flush=True)
        rows_out = []
        for d, cm in [("all", sum(ev["sky"].values()))] + sorted(ev["sky"].items()):
            sc = S.scores_from_confusion(cm)
            lo, hi = bootstrap_miou(ev["per_image"], None if d == "all" else d,
                                    scorer=lambda c: S.scores_from_confusion(c).miou)
            tag = "dense_heads" + ("+layers" if a.layers else "") + ("+kendall" if weighting is not None else "")
            row = {"phase": a.phase, "model": tag, "seed": a.seed, "dataset": d,
                   "miou": sc.miou, "miou_lo": lo, "miou_hi": hi, **{f"iou_{k}": v for k, v in sc.iou.items()},
                   "clear_sky_cloud_fraction": frac, "best_epoch": best["epoch"], "run_id": run.run_id,
                   **{f"clear_frac_{k}": v for k, v in clear_frac.items()},
                   **{f"b0268_frac_{k}": v for k, v in b0268_frac.items()}}
            if a.layers:
                lcm = sum(ev["layer"].values()) if d == "all" else ev["layer"][d]
                ls = layer_scores(lcm)
                row.update({f"layer_iou_{k}": v for k, v in ls.items()})
                row["layer_miou"] = float(np.mean(list(ls.values()))) if ls else float("nan")
                if ls:
                    lcms = _layer_cms(model, dl_va, val_datasets, device)
                    per_l = [(dd, c) for dd, c in zip([p[0] for p in ev["per_image"]], lcms, strict=True)]
                    llo, lhi = bootstrap_miou(per_l, None if d == "all" else d,
                                              scorer=lambda c: float(np.mean(list(layer_scores(c).values()) or [np.nan])))
                    row["layer_miou_lo"], row["layer_miou_hi"] = llo, lhi
            rows_out.append(row)
        summary = {"val_miou": best["val_miou"], "best_epoch": best["epoch"], "clear_sky_cloud_fraction": frac,
                   "b0268_cloud_fraction": b0268_frac.get("cloud", float("nan"))}
        if a.layers:
            summary["val_layer_miou"] = best.get("val_layer_miou", float("nan"))
        run.finish(**summary)
    df = pd.DataFrame(rows_out)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        key = ["phase", "model", "seed", "dataset"]
        df = pd.concat([old[~old.set_index(key).index.isin(df.set_index(key).index)], df], ignore_index=True)
    df.to_parquet(OUT, index=False)
    write_doc(df)
    print(f"done in {time.perf_counter() - t0:.0f} s: val mIoU {best['val_miou']:.3f} (epoch {best['epoch']}), "
          f"clear-sky cloud fraction {frac:.3f} -> {DOC}", flush=True)
    return 0




def write_doc(df: pd.DataFrame) -> None:
    probe_path = Path("data/baseline_segmentation.parquet")
    probe = pd.read_parquet(probe_path) if probe_path.exists() else pd.DataFrame()
    lines = ["# Dense heads on cached DINOv3 tokens (P065 / P066)", "",
             "DPT-lite decoder (ADR-005 §2) on the frozen DINOv3 ViT-S/16 32 x 32 token grid at 512 px, grouped train -> val "
             "(test locked), masks at 512 px, mIoU over the classes present in each dataset's truth (as P057), 95 % bootstrap "
             "intervals over val images, and the clear-sky collapse check (predicted-cloud fraction inside the camera mask on "
             "200 clear-sky Eye2Sky frames). The P057 token probe is the reference the decoder has to beat.", "",
             "| Phase | Model | Dataset | mIoU [95 % CI] | IoU sky | IoU cloud | IoU invalid | Layer mIoU [CI] | "
             "low / mid / high IoU | Clear-sky cloud fraction | B0268 cloud fraction | Probe mIoU |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values(["phase", "model", "dataset"]).iterrows():
        pr = probe[(probe.model == "dinov3_vits16_patch_probe") & (probe.dataset == r.dataset)] if len(probe) else pd.DataFrame()
        pm = f"{pr.miou.mean():.3f}" if len(pr) else "-"
        lm = (f"{r.layer_miou:.3f} [{r.layer_miou_lo:.3f}, {r.layer_miou_hi:.3f}]"
              if "layer_miou" in r and pd.notna(r.get("layer_miou")) and pd.notna(r.get("layer_miou_lo")) else "-")
        lih = " / ".join(f"{r[k]:.3f}" if k in r and pd.notna(r[k]) else "-"
                         for k in ("layer_iou_low", "layer_iou_mid", "layer_iou_high"))
        inv = f"{r.iou_invalid:.3f}" if "iou_invalid" in r and pd.notna(r.iou_invalid) else "-"
        b0 = f"{r.b0268_frac_cloud:.3f} " if "b0268_frac_cloud" in r and pd.notna(r.get("b0268_frac_cloud")) else "- "
        lines.append(f"| {r.phase} | {r.model} | {r.dataset} | {r.miou:.3f} [{r.miou_lo:.3f}, {r.miou_hi:.3f}] | "
                     f"{r.iou_sky:.3f} | "
                     f"{r.iou_cloud:.3f} | {inv} | {lm} | {lih} | {r.clear_sky_cloud_fraction:.3f} | {b0}| {pm} |")
    lines += ["", "Rows in `data/dense_heads.parquet`; curves in each run's `metrics.jsonl` (phase P065 / P066)."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
