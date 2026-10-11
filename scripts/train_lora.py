"""Stage-5 partial unfreeze: LoRA on the last blocks of the backbone with the heads in the graph (P077).

    python scripts/train_lora.py --init runs/<joint run>/checkpoints/best.pt --epochs 4 [--rank 8 --last-n 4]

The backbone (DINOv3 ViT-S/16 through transformers) gets LoRA adapters (peft) on the q / k / v projections of its
last `--last-n` blocks, gradient checkpointing, and a layer-wise learning-rate decay; the heads start from the
P076 joint checkpoint (or fresh). Images come from the P022 cache with the P048 augmentation policy (geometric
chain + photometric), masks warped with the images; every row's ray map is the null state (the uncalibrated
sources; geometry-aware fine-tuning on Eye2Sky is outside this phase). Two batch groups (global 16 images, dense 8
images with masks) from the P069 sampler; bfloat16 autocast; the P072 engine.

Evaluation on the same val images at the full-frame 512 px resize, before training with the adapters disabled
(= the frozen reference of this pipeline) and after every epoch: the P073 global metrics per dataset and the
P065 / P066 dense scores. The gain over frozen and the forgetting (per-dataset drops) are the phase's numbers
(docs/data/stage5_lora.md).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.augment.policy import AugmentationPolicy  # noqa: E402
from stratia.baselines import segmentation as S  # noqa: E402
from stratia.data.image_cache import cache_path, read_rgb  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.data.sampler import SamplerConfig, TaskBatchSampler  # noqa: E402
from stratia.data.segmentation import IGNORE  # noqa: E402
from stratia.features.backbones import BACKBONES  # noqa: E402
from stratia.model import losses as L  # noqa: E402
from stratia.model.balance import UncertaintyWeighting  # noqa: E402
from stratia.model.stratia import STRATIA  # noqa: E402
from stratia.train import data as D  # noqa: E402
from stratia.train import dense_data as DD  # noqa: E402
from stratia.train.engine import Engine, EngineConfig  # noqa: E402
from stratia.train.global_eval import losses_for, score_dataset  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/stage5_lora.md")
OUT = Path("data/stage5_lora.parquet")
SIZE = 512
TASKS = ("genus", "etage", "oktas", "sky", "layer")


# ----------------------------------------------------------------------------------------------- backbone
def _block_index(name: str) -> int:
    m = re.search(r"\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


class LoraBackbone(nn.Module):
    def __init__(self, key: str, rank: int, alpha: int, last_n: int, dropout: float = 0.05, checkpointing: bool = True):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        self.spec = BACKBONES[key]
        base = AutoModel.from_pretrained(self.spec.hf_id)
        self.n_layers = int(base.config.num_hidden_layers)
        self.n_skip = 1 + int(getattr(base.config, "num_register_tokens", 0))
        first = self.n_layers - last_n
        targets = [n for n, m in base.named_modules()
                   if isinstance(m, nn.Linear) and n.split(".")[-1] in ("q_proj", "k_proj", "v_proj")
                   and _block_index(n) >= first]
        if not targets:
            raise RuntimeError("no q/k/v projections found for LoRA")
        if checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        self.model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets, lora_dropout=dropout))
        self.targets = targets
        self.register_buffer("mean", torch.tensor(self.spec.mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.spec.std).view(1, 3, 1, 1))

    def forward(self, x_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x_uint8.float() / 255.0 - self.mean) / self.std
        out = self.model(pixel_values=x)
        return out.last_hidden_state[:, self.n_skip:, :].float(), out.pooler_output.float()

    def lora_param_groups(self, lr: float, decay: float) -> list[dict]:
        groups: dict[int, list] = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                groups.setdefault(_block_index(n), []).append(p)
        return [{"params": ps, "lr": lr * (decay ** (self.n_layers - 1 - b))} for b, ps in sorted(groups.items())]


class LoraModel(nn.Module):
    def __init__(self, backbone: LoraBackbone, kendall: bool):
        super().__init__()
        self.backbone = backbone
        self.net = STRATIA(None, d_model=384, dense=True, cbh=True, ood=False)
        self.weighting = UncertaintyWeighting(TASKS) if kendall else None

    def forward(self, image, ray_map, meta):
        tokens, cls = self.backbone(image)
        return self.net.forward_tokens(tokens, cls, ray_map, meta)


# ----------------------------------------------------------------------------------------------- data
def encode_masks(sky: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """One uint8 label for the geometric chain: sky code (0-3, 4 = ignore) * 10 + layer code (0-2, 3 = ignore)."""
    s = np.where(sky == IGNORE, 4, sky).astype(np.uint8)
    lay = np.where(layer == IGNORE, 3, layer).astype(np.uint8)
    return s * 10 + lay


def decode_masks(code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ignore = code == 255                                   # padding written by the warp
    s = np.where(ignore, 4, code // 10)
    lay = np.where(ignore, 3, code % 10)
    sky = np.where(s == 4, IGNORE, s).astype(np.uint8)
    layer = np.where(lay == 3, IGNORE, lay).astype(np.uint8)
    return sky, layer


class ImageGlobalDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, camera_types: dict, cache_root: Path, policy: AugmentationPolicy | None,
                 seed: int = 0):
        self.rows = rows.reset_index(drop=True)
        self.cam = camera_types
        self.cache_root, self.policy = cache_root, policy
        self.rng = np.random.default_rng(seed)
        self.genus = np.stack(self.rows.genus_set.to_numpy())
        self.neg = np.stack(self.rows.genus_neg.to_numpy())
        self.etage = np.stack(self.rows.etage.to_numpy())
        self.etage_w = np.stack(self.rows.etage_w.to_numpy())
        self.oktas = np.stack(self.rows.oktas.to_numpy())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = read_rgb(cache_path(self.cache_root, r.image_file))
        if self.policy is not None:
            img = self.policy(img, self.cam.get(r.dataset, "consumer_photo"), self.rng).image
        img = cv2.resize(np.ascontiguousarray(img), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        return {"image": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                "genus_set": torch.from_numpy(self.genus[i]), "genus_alt": bool(r.genus_alt),
                "genus_neg": torch.from_numpy(self.neg[i]), "mask_genus": bool(r.mask_genus),
                "etage": torch.from_numpy(self.etage[i]), "etage_w": torch.from_numpy(self.etage_w[i]),
                "mask_etage": bool(r.mask_etage), "oktas": torch.from_numpy(self.oktas[i]), "mask_oktas": bool(r.mask_oktas),
                "index": i}


class ImageDenseDataset(Dataset):
    def __init__(self, files: list[str], datasets: list[str], sky: list[np.ndarray], layer: list[np.ndarray], camera_types: dict,
                 cache_root: Path, policy: AugmentationPolicy | None, seed: int = 0):
        self.files, self.datasets, self.sky, self.layer = files, datasets, sky, layer
        self.cam, self.cache_root, self.policy = camera_types, cache_root, policy
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = read_rgb(cache_path(self.cache_root, self.files[i]))
        sky, layer = self.sky[i], self.layer[i]
        if sky.shape[:2] != img.shape[:2]:
            sky = cv2.resize(sky, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            layer = cv2.resize(layer, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.policy is not None:
            aug = self.policy(img, self.cam.get(self.datasets[i], "wsi_crop"), self.rng, label=encode_masks(sky, layer))
            img = aug.image
            sky, layer = decode_masks(aug.label)
        img = cv2.resize(np.ascontiguousarray(img), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        sky = cv2.resize(np.ascontiguousarray(sky), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        layer = cv2.resize(np.ascontiguousarray(layer), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), torch.from_numpy(sky.astype(np.int64)),
                torch.from_numpy(layer.astype(np.int64)))


# ----------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_global(model: LoraModel, ds: ImageGlobalDataset, device: str, batch: int = 32) -> dict[str, float]:
    model.eval()
    dl = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=D.collate, num_workers=0)
    gen, eta, okt = [], [], []
    for b in dl:
        n = len(b["image"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(b["image"].to(device), torch.zeros(n, 4, 32, 32, device=device), torch.zeros(n, 3, device=device))
        gen.append(torch.sigmoid(out["genus_logits"].float()).cpu().numpy())
        eta.append(torch.sigmoid(out["etage_logits"].float()).cpu().numpy())
        okt.append(out["oktas_probs"].float().cpu().numpy())
    pred = {"genus": np.concatenate(gen), "etage": np.concatenate(eta), "oktas": np.concatenate(okt)}
    metrics, e_vals = {}, []
    for ds_name in ds.rows.dataset.unique():
        sel = (ds.rows.dataset == ds_name).to_numpy()
        sc = score_dataset(ds.rows[sel].reset_index(drop=True), {k: v[sel] for k, v in pred.items()})
        for k, v in sc.items():
            metrics[f"global/{ds_name}/{k}"] = float(v)
        if "etage_f1" in sc:
            e_vals.append(sc["etage_f1"])
    metrics["global/mean_etage_f1"] = float(np.mean(e_vals)) if e_vals else 0.0
    return metrics


@torch.no_grad()
def eval_dense(model: LoraModel, ds: ImageDenseDataset, device: str, batch: int = 8) -> dict[str, float]:
    model.eval()
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    cm_sky = {d: np.zeros((S.N_CLASSES, S.N_CLASSES), dtype=np.int64) for d in set(ds.datasets)}
    cm_lay = np.zeros((3, 3), dtype=np.int64)
    k = 0
    for img, sky, lay in dl:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            tokens, _ = model.backbone(img.to(device))
            out = model.net.dense_heads(tokens.float())
        logits = torch.nn.functional.interpolate(out["sky_parse_logits"].float(), size=(SIZE, SIZE), mode="bilinear",
                                                 align_corners=False)
        pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
        llog = torch.nn.functional.interpolate(out["layer_logits"].float(), size=(SIZE, SIZE), mode="bilinear",
                                               align_corners=False)
        lpred = llog.argmax(1).cpu().numpy().astype(np.uint8)
        for j in range(len(img)):
            cm_sky[ds.datasets[k]] += S.confusion(pred[j], sky[j].numpy().astype(np.uint8))
            lt = lay[j].numpy().astype(np.uint8)
            keep = lt != IGNORE
            if keep.any():
                cm_lay += np.bincount(3 * lt[keep].astype(int) + lpred[j][keep].astype(int), minlength=9).reshape(3, 3)
            k += 1
    metrics = {"dense/all/miou": S.scores_from_confusion(sum(cm_sky.values())).miou}
    for d, cm in cm_sky.items():
        metrics[f"dense/{d}/miou"] = S.scores_from_confusion(cm).miou
    ls = DD.layer_scores(cm_lay)
    metrics["dense/almeria/layer_miou"] = float(np.mean(list(ls.values()))) if ls else float("nan")
    return metrics


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="P077")
    ap.add_argument("--name", default="lora")
    ap.add_argument("--init", default=None, help="P076 joint checkpoint (best.pt) for the heads")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--last-n", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batches-per-epoch", type=int, default=500)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-heads", type=float, default=3e-4)
    ap.add_argument("--layer-decay", type=float, default=0.8)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--aug-strength", type=float, default=1.0)
    ap.add_argument("--kendall", action="store_true")
    ap.add_argument("--lambda-h", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.manual_seed(a.seed)
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9)
    man = pd.read_parquet("data/manifest.parquet", columns=["dataset", "camera_type"])
    camera_types = man.groupby("dataset").camera_type.first().to_dict()
    policy = AugmentationPolicy(strength=a.aug_strength)

    g_rows = D.global_rows(eye2sky_etage=False)
    g_tr = g_rows[g_rows.split == "train"]
    g_va = g_rows[g_rows.split == "val"]
    ds_g_tr = ImageGlobalDataset(g_tr, camera_types, cache_root, policy, a.seed)
    ds_g_va = ImageGlobalDataset(g_va, camera_types, cache_root, None)
    d_tab = S.seg_tables(data_root)
    sky_masks, layer_masks = S.load_both_masks(d_tab, data_root)
    d_tr = (d_tab.split == "train").to_numpy()
    pick = lambda lst, m: [x for x, k in zip(lst, m, strict=True) if k]  # noqa: E731
    ds_d_tr = ImageDenseDataset(pick(d_tab.image_file.tolist(), d_tr), pick(d_tab.dataset.tolist(), d_tr), pick(sky_masks, d_tr),
                                pick(layer_masks, d_tr), camera_types, cache_root, policy, a.seed)
    ds_d_va = ImageDenseDataset(pick(d_tab.image_file.tolist(), ~d_tr), pick(d_tab.dataset.tolist(), ~d_tr),
                                pick(sky_masks, ~d_tr), pick(layer_masks, ~d_tr), camera_types, cache_root, None)
    print(f"global {len(ds_g_tr):,}/{len(ds_g_va):,}; dense {len(ds_d_tr):,}/{len(ds_d_va):,} images (train/val)", flush=True)

    backbone = LoraBackbone("dinov3_vits16", a.rank, a.alpha, a.last_n)
    model = LoraModel(backbone, a.kendall).to(device)
    n_lora = sum(p.numel() for p in backbone.model.parameters() if p.requires_grad)
    if a.init:
        state = torch.load(a.init, map_location=device, weights_only=False)["model"]
        heads = {k: v for k, v in state.items() if k.startswith("net.")}
        missing, unexpected = model.load_state_dict(heads, strict=False)
        print(f"heads initialised from {a.init}: {len(heads)} tensors", flush=True)
    print(f"LoRA on {len(backbone.targets)} projections of the last {a.last_n} blocks: {n_lora:,} trainable backbone "
          "parameters", flush=True)
    groups = backbone.lora_param_groups(a.lr_lora, a.layer_decay)
    groups.append({"params": [p for n, p in model.named_parameters() if n.startswith("net.") or n.startswith("weighting.")],
                   "lr": a.lr_heads})
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * a.batches_per_epoch)

    table = pd.concat([pd.DataFrame({"dataset": g_tr.dataset.to_numpy(), "group_mask": [{"global"}] * len(g_tr),
                                     "pos": np.arange(len(g_tr))}),
                       pd.DataFrame({"dataset": ds_d_tr.datasets, "group_mask": [{"dense"}] * len(ds_d_tr),
                                     "pos": np.arange(len(ds_d_tr))})], ignore_index=True)
    cfg_s = SamplerConfig(tau=0.5, group_probs={"global": 0.6, "dense": 0.4}, batch_sizes={"global": 16, "dense": 8},
                          batches_per_epoch=a.batches_per_epoch)
    sampler = TaskBatchSampler(table, cfg_s, a.seed)

    def to_dev(b):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

    def make_batch(group, idx):
        if group == "global":
            b = D.collate([ds_g_tr[int(table.pos.iloc[i])] for i in idx])
            return "global", to_dev(b)
        items = [ds_d_tr[int(table.pos.iloc[i])] for i in idx]
        b = {"image": torch.stack([it[0] for it in items]), "sky": torch.stack([it[1] for it in items]),
             "layer": torch.stack([it[2] for it in items])}
        return "dense", to_dev(b)

    def step_fn(m, batch, step):
        group, b = batch
        n = len(b["image"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = m(b["image"], torch.zeros(n, 4, 32, 32, device=device), torch.zeros(n, 3, device=device))
        if group == "global":
            parts = losses_for({k: v.float() for k, v in out.items()}, b, a.lambda_h)
            losses = {"genus": parts["genus"], "etage": parts["etage"], "oktas": parts["oktas"]}
            extra = parts["hier"]
        else:
            losses = {"sky": L.dense_ce(out["sky_parse_logits"].float(), b["sky"]),
                      "layer": L.dense_ce(out["layer_logits"].float(), b["layer"])}
            extra = out["sky_parse_logits"].sum() * 0.0
        total = (m.weighting(losses)[0] if m.weighting is not None else sum(losses.values())) + extra
        return total, {f"loss/{k}": v.detach() for k, v in losses.items()}

    def full_eval(m):
        return {**eval_global(m, ds_g_va, device), **eval_dense(m, ds_d_va, device)}

    cfg = OmegaConf.create({"name": a.name, "init": a.init, "rank": a.rank, "alpha": a.alpha, "last_n": a.last_n,
                            "epochs": a.epochs,
                            "batches_per_epoch": a.batches_per_epoch, "lr_lora": a.lr_lora, "lr_heads": a.lr_heads,
                            "layer_decay": a.layer_decay, "weight_decay": a.weight_decay, "aug_strength": a.aug_strength,
                            "kendall": a.kendall, "seed": a.seed, "lora_params": n_lora, "targets": len(backbone.targets)})
    curves = []
    with Run(cfg, name=f"lora_{a.name}_s{a.seed}", phase=a.phase) as run:
        with backbone.model.disable_adapter():
            before = full_eval(model)
        print(f"  frozen reference: étage F1 {before['global/mean_etage_f1']:.3f} | dense mIoU {before['dense/all/miou']:.3f} "
              f"layer {before['dense/almeria/layer_miou']:.3f} | {time.perf_counter() - t0:.0f} s", flush=True)
        run.log(0, **{f"before/{k}": v for k, v in before.items()})
        ecfg = EngineConfig(epochs=a.epochs, clip_norm=1.0, amp=False, ema_decay=None, monitor="val/headline", mode="max",
                            patience=a.patience, log_every=0, device=device)
        engine = Engine(model, opt, ecfg, scheduler=sched, log=run.log, checkpoint_dir=run.dir / "checkpoints")

        def loader_fn(epoch):
            sampler.set_epoch(epoch)
            for group, idx in sampler:
                yield make_batch(group, idx)

        def eval_fn(m):
            metrics = full_eval(m)
            metrics["val/headline"] = float(np.mean([metrics["global/mean_etage_f1"], metrics["dense/all/miou"]]))
            curves.append({"epoch": engine.epoch + 1, **metrics})
            print(f"  epoch {engine.epoch + 1}: étage F1 {metrics['global/mean_etage_f1']:.3f} | "
                  f"dense mIoU {metrics['dense/all/miou']:.3f} "
                  f"layer {metrics['dense/almeria/layer_miou']:.3f} | headline {metrics['val/headline']:.3f} | "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
            return metrics

        res = engine.fit(loader_fn, step_fn, eval_fn)
        best = next((c for c in curves if c["epoch"] == res["best_step"] // a.batches_per_epoch), curves[-1])
        (run.dir / "curves.json").write_text(json.dumps({"before": before, "epochs": curves}, indent=1), encoding="utf-8")
        run.finish(best=res["best"], best_epoch=best["epoch"], before_headline=float(np.mean([before["global/mean_etage_f1"],
                                                                                               before["dense/all/miou"]])))
    rows = []
    for k, v in best.items():
        if k.count("/") == 2 or k.startswith("global/mean") or k == "dense/all/miou":
            rows.append({"phase": a.phase, "name": a.name, "seed": a.seed, "metric": k, "after": float(v),
                         "before": float(before.get(k, np.nan)),
                         "best_epoch": best["epoch"], "run_id": run.run_id})
    df = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        key = ["name", "seed", "metric"]
        df = pd.concat([old[~old.set_index(key).index.isin(df.set_index(key).index)], df], ignore_index=True)
    df.to_parquet(OUT, index=False)
    write_doc(df)
    frozen = float(np.mean([before["global/mean_etage_f1"], before["dense/all/miou"]]))
    print(f"done in {time.perf_counter() - t0:.0f} s: headline {res['best']:.3f} (frozen {frozen:.3f}) -> {DOC}", flush=True)
    return 0


def write_doc(df: pd.DataFrame) -> None:
    lines = ["# Stage-5 partial unfreeze: LoRA on the backbone (P077)", "",
             "DINOv3 ViT-S/16 with LoRA adapters on the q / k / v projections of the last blocks, heads from the P076 joint "
             "checkpoint, P048 augmentation, bfloat16, gradient checkpointing. 'Before' = the same heads on the frozen backbone "
             "evaluated through this image pipeline (adapters disabled); 'after' = the best epoch. A negative Δ on a dataset the "
             "run did not improve is forgetting.", "",
             "| Run | Metric | Before (frozen) | After (LoRA) | Δ |", "|---|---|---|---|---|"]
    for name, part in df.groupby("name"):
        for r in part.sort_values("metric").itertuples():
            if np.isnan(r.before):
                continue
            lines.append(f"| {name} (epoch {int(r.best_epoch)}) | {r.metric} | {r.before:.3f} | {r.after:.3f} | "
                         f"{r.after - r.before:+.3f} |")
    lines += ["", "Rows in `data/stage5_lora.parquet`; per-epoch curves in the run folder."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
