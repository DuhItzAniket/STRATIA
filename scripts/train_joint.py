"""Stage-4 joint multi-task training of all heads on cached features (P076).

    python scripts/train_joint.py --epochs 15 [--kendall] [--batches-per-epoch 400]

Three batch groups from the P069 sampler, one model (STRATIA without a backbone, P071): `global` batches (pooled
8 x 8 grid + cls of the labelled recognition images: genus / étage / oktas losses + hierarchy term), `dense` batches
(32 x 32 grid of the segmentation images: sky-parsing + layer losses), `cbh` batches (8 x 8 grid of the P047 frames
with ray maps: CBH loss + the CBH-étage consistency term through the global heads). Kendall weighting over the six
task losses (P070) and the task-gradient cosine monitor on the shared parameters, logged every `--log-every` steps.
Evaluation per epoch: the P073 per-dataset global metrics, the P065/P066 dense scores and the P067 CBH scores on
the same val sets, so the joint model is compared head by head with the single-task runs
(docs/data/stage4_joint.md). Early stopping on the mean of the three headline numbers (étage macro-F1, dense mIoU,
CBH skill).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.augment.dropout import GeometryDropout  # noqa: E402
from stratia.baselines import cbh as C  # noqa: E402
from stratia.baselines import segmentation as S  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.data.sampler import SamplerConfig, TaskBatchSampler  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.model import losses as L  # noqa: E402
from stratia.model.balance import GradConflictMonitor, UncertaintyWeighting  # noqa: E402
from stratia.model.stratia import STRATIA  # noqa: E402
from stratia.train import cbh_data as CD  # noqa: E402
from stratia.train import data as D  # noqa: E402
from stratia.train import dense_data as DD  # noqa: E402
from stratia.train.engine import Engine, EngineConfig  # noqa: E402
from stratia.train.global_eval import losses_for, score_dataset  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/stage4_joint.md")
OUT = Path("data/stage4_joint.parquet")
TASKS = ("genus", "etage", "oktas", "sky", "layer", "cbh")


class JointModel(torch.nn.Module):
    def __init__(self, d_model: int, kendall: bool):
        super().__init__()
        self.net = STRATIA(None, d_model=d_model, dense=True, cbh=True, ood=False)
        self.weighting = UncertaintyWeighting(TASKS) if kendall else None

    def forward(self, tokens, cls, ray_map, meta):
        return self.net.forward_tokens(tokens, cls, ray_map, meta)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="P076")
    ap.add_argument("--name", default="joint")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batches-per-epoch", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--kendall", action="store_true")
    ap.add_argument("--lambda-h", type=float, default=0.1)
    ap.add_argument("--lambda-layer", type=float, default=0.1)
    ap.add_argument("--p-drop", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--group-probs", nargs=3, type=float, default=[0.5, 0.3, 0.2])
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- global group
    g_rows = D.global_rows(eye2sky_etage=True)
    grid8 = FeatureCache(cache_root, "dinov3_vits16", 512, transform="ffr1-patches-all-pool4")
    cls_cache = FeatureCache(cache_root, "dinov3_vits16", 512)
    g_rows = g_rows[g_rows.sample_id.isin(set(grid8.index().sample_id))].reset_index(drop=True)
    g_geo = D.GeometryCache.build(g_rows, data_root, cache_root / "geometry" / f"pool4_joint_{len(g_rows)}.npz")
    g_ds = D.GlobalDataset(g_rows, grid8.rows_for(g_rows.sample_id), cls_cache.rows_for(g_rows.sample_id), grid8.load("patches"),
                           cls_cache.load("cls"), g_geo)
    g_tr = np.flatnonzero(g_rows.split == "train")
    g_va = np.flatnonzero(g_rows.split == "val")
    # ---- dense group
    d_tab = S.seg_tables(data_root)
    sky_masks, layer_masks = S.load_both_masks(d_tab, data_root)
    grid32 = FeatureCache(cache_root, "dinov3_vits16", 512, transform="ffr1-patches-seg")
    d_rows_all = grid32.rows_for(d_tab.sample_id)
    d_patches = grid32.load("patches")
    d_tr = (d_tab.split == "train").to_numpy()
    d_ds_tr = DD.TokenMaskDataset(d_rows_all[d_tr], d_patches, [m for m, k in zip(sky_masks, d_tr, strict=True) if k],
                                  [m for m, k in zip(layer_masks, d_tr, strict=True) if k], True, a.seed)
    d_ds_va = DD.TokenMaskDataset(d_rows_all[~d_tr], d_patches, [m for m, k in zip(sky_masks, ~d_tr, strict=True) if k],
                                  [m for m, k in zip(layer_masks, ~d_tr, strict=True) if k], False)
    d_cls_rows = cls_cache.rows_for(d_tab.sample_id)
    d_cls = cls_cache.load("cls")
    d_val_datasets = d_tab.dataset[~d_tr].tolist()
    # ---- cbh group
    pairs = CD.load_pairs(grid8)
    rms, metas = CD.ray_maps_for(pairs, data_root)
    c_tr, c_va = (pairs.split == "train").to_numpy(), (pairs.split == "val").to_numpy()
    c_heights = pairs.height_m.to_numpy().astype(np.float32)
    c_grid_rows, c_cls_rows = grid8.rows_for(pairs.sample_id), cls_cache.rows_for(pairs.sample_id)
    c_ds_tr = CD.PairDataset(c_grid_rows[c_tr], c_cls_rows[c_tr], grid8.load("patches"), cls_cache.load("cls"), rms[c_tr],
                             metas[c_tr], c_heights[c_tr])
    c_ds_va = CD.PairDataset(c_grid_rows[c_va], c_cls_rows[c_va], grid8.load("patches"), cls_cache.load("cls"), rms[c_va],
                             metas[c_va], c_heights[c_va])
    clim = C.climatology(pairs[c_tr])
    true_va, cloud_va = c_heights[c_va], pairs.cloud.to_numpy()[c_va]
    mae_clim = C.score_heights(np.full(int((~np.isnan(true_va)).sum()), clim), true_va[~np.isnan(true_va)]).mae
    print(f"global {len(g_tr):,}/{len(g_va):,}; dense {int(d_tr.sum()):,}/{int((~d_tr).sum()):,}; "
          f"cbh {int(c_tr.sum()):,}/{int(c_va.sum()):,} (train/val rows)", flush=True)

    # ---- sampler over the union
    table = pd.concat([pd.DataFrame({"dataset": g_rows.dataset.iloc[g_tr].to_numpy(), "group_mask": [{"global"}] * len(g_tr),
                                     "pos": g_tr, "group": "global"}),
                       pd.DataFrame({"dataset": d_tab.dataset[d_tr].to_numpy(), "group_mask": [{"dense"}] * int(d_tr.sum()),
                                     "pos": np.arange(int(d_tr.sum())), "group": "dense"}),
                       pd.DataFrame({"dataset": "eye2sky_cbh", "group_mask": [{"cbh"}] * int(c_tr.sum()),
                                     "pos": np.arange(int(c_tr.sum())), "group": "cbh"})], ignore_index=True)
    gp = dict(zip(("global", "dense", "cbh"), a.group_probs, strict=True))
    cfg_s = SamplerConfig(tau=a.tau, group_probs=gp, batch_sizes={"global": 64, "dense": 16, "cbh": 64},
                          batches_per_epoch=a.batches_per_epoch)
    sampler = TaskBatchSampler(table, cfg_s, seed=a.seed)
    model = JointModel(384, a.kendall).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * a.batches_per_epoch)
    drop = GeometryDropout(a.p_drop, a.p_drop)
    shared = list(model.net.geo.parameters()) + list(model.net.global_heads.parameters())
    monitor = GradConflictMonitor(shared)
    last_batch: dict[str, dict] = {}

    def to_dev(b):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    def make_batch(group, idx):
        if group == "global":
            b = D.collate([g_ds[int(table.pos.iloc[i])] for i in idx])
            rm, mt, _, _ = drop.batch(b["ray_map"].numpy(), b["meta"].numpy(), rng)
            b["ray_map"], b["meta"] = torch.from_numpy(rm), torch.from_numpy(mt)
            return "global", to_dev(b)
        if group == "dense":
            items = [d_ds_tr[int(table.pos.iloc[i])] for i in idx]
            tok = torch.stack([t for t, _, _ in items])
            sky = torch.stack([s_ for _, s_, _ in items])
            lay = torch.stack([l_ for _, _, l_ in items])
            n = tok.shape[0]
            rows = d_rows_all[d_tr][[int(table.pos.iloc[i]) for i in idx]]
            cls_ = torch.from_numpy(np.asarray(d_cls[d_cls_rows[d_tr][[int(table.pos.iloc[i]) for i in idx]]], dtype=np.float32))
            del rows
            side = int(np.sqrt(tok.shape[1]))
            b = {"tokens": tok, "cls": cls_, "ray_map": torch.zeros(n, 4, side, side), "meta": torch.zeros(n, 3),
                 "sky": sky, "layer": lay}
            return "dense", to_dev(b)
        items = [c_ds_tr[int(table.pos.iloc[i])] for i in idx]
        tok, cls_, rm, mt, h, _ = (torch.stack([it[k] for it in items]) for k in range(6))
        rm_np, mt_np, _, _ = drop.batch(rm.numpy(), mt.numpy(), rng)
        b = {"tokens": tok, "cls": cls_, "ray_map": torch.from_numpy(rm_np), "meta": torch.from_numpy(mt_np), "height_m": h}
        return "cbh", to_dev(b)

    def group_losses(m, group, b):
        out = m(b["tokens"], b["cls"], b["ray_map"], b["meta"])
        if group == "global":
            parts = losses_for(out, b, a.lambda_h)
            return {"genus": parts["genus"], "etage": parts["etage"], "oktas": parts["oktas"]}, parts["hier"]
        if group == "dense":
            return {"sky": L.dense_ce(out["sky_parse_logits"].float(), b["sky"]),
                    "layer": L.dense_ce(out["layer_logits"].float(), b["layer"])}, out["sky_parse_logits"].sum() * 0.0
        mask = torch.ones(len(b["height_m"]), dtype=torch.bool, device=device)
        cbh, _ = L.cbh_loss(out["cbh_logits"], b["height_m"], mask)
        cons = L.cbh_etage_consistency(out["cbh_probs"], out["etage_logits"], mask)
        return {"cbh": cbh}, a.lambda_layer * cons

    cfg = OmegaConf.create({"name": a.name, "epochs": a.epochs, "batches_per_epoch": a.batches_per_epoch, "lr": a.lr,
                            "weight_decay": a.weight_decay, "kendall": a.kendall, "lambda_h": a.lambda_h,
                            "lambda_layer": a.lambda_layer,
                            "p_drop": a.p_drop, "tau": a.tau, "group_probs": list(a.group_probs), "seed": a.seed,
                            "params": sum(p.numel() for p in model.parameters())})
    curves, cos_log = [], []
    with Run(cfg, name=f"joint_{a.name}_s{a.seed}", phase=a.phase) as run:
        ecfg = EngineConfig(epochs=a.epochs, clip_norm=1.0, amp=False, ema_decay=0.99, monitor="val/headline", mode="max",
                            patience=a.patience, log_every=0, device=device)
        engine = Engine(model, opt, ecfg, scheduler=sched, log=run.log, checkpoint_dir=run.dir / "checkpoints")

        def loader_fn(epoch):
            sampler.set_epoch(epoch)
            for group, idx in sampler:
                yield make_batch(group, idx)

        def step_fn(m, batch, step):
            group, b = batch
            parts, extra = group_losses(m, group, b)
            last_batch[group] = b
            if m.weighting is not None:
                total, _ = m.weighting(parts)
            else:
                total = sum(parts.values())
            total = total + extra
            metrics = {f"loss/{k}": v.detach() for k, v in parts.items()}
            if a.log_every and step % a.log_every == 0 and len(last_batch) == 3:
                all_losses = {}
                for g, bb in last_batch.items():
                    p, _ = group_losses(m, g, bb)
                    all_losses.update({k: v for k, v in p.items() if v.requires_grad and float(v) != 0.0})
                cos = monitor.cosines(all_losses)
                cos_log.append({"step": step, **cos})
                run.log(step, **{f"grad_cos/{k}": v for k, v in cos.items() if "|" in k},
                        **({f"weight/{k}": v for k, v in m.weighting.weights().items()} if m.weighting is not None else {}))
            return total, metrics

        @torch.no_grad()
        def eval_fn(m):
            m.eval()
            metrics = {}
            # global
            dl = DataLoader(torch.utils.data.Subset(g_ds, g_va.tolist()), batch_size=256, shuffle=False, collate_fn=D.collate)
            gen, eta, okt = [], [], []
            for b in dl:
                out = m(b["tokens"].to(device), b["cls"].to(device), b["ray_map"].to(device), b["meta"].to(device))
                gen.append(torch.sigmoid(out["genus_logits"]).cpu().numpy())
                eta.append(torch.sigmoid(out["etage_logits"]).cpu().numpy())
                okt.append(out["oktas_probs"].cpu().numpy())
            pred = {"genus": np.concatenate(gen), "etage": np.concatenate(eta), "oktas": np.concatenate(okt)}
            part = g_rows.iloc[g_va].reset_index(drop=True)
            e_vals = []
            for ds_name in part.dataset.unique():
                sel = (part.dataset == ds_name).to_numpy()
                sc = score_dataset(part[sel].reset_index(drop=True), {k: v[sel] for k, v in pred.items()})
                for k, v in sc.items():
                    metrics[f"global/{ds_name}/{k}"] = float(v)
                if "etage_f1" in sc:
                    e_vals.append(sc["etage_f1"])
            metrics["global/mean_etage_f1"] = float(np.mean(e_vals)) if e_vals else 0.0
            # dense
            dl_d = DataLoader(d_ds_va, batch_size=16, shuffle=False)
            ev = DD.evaluate(m.net.dense_heads, dl_d, d_val_datasets, device, True)
            sc_all = S.scores_from_confusion(sum(ev["sky"].values()))
            metrics["dense/all/miou"] = sc_all.miou
            for d_name, cm in ev["sky"].items():
                metrics[f"dense/{d_name}/miou"] = S.scores_from_confusion(cm).miou
            ls = DD.layer_scores(sum(ev["layer"].values()))
            metrics["dense/almeria/layer_miou"] = float(np.mean(list(ls.values()))) if ls else float("nan")
            # cbh
            probs = CD.predict(m, DataLoader(c_ds_va, batch_size=256, shuffle=False), device)
            sc_c = CD.score(probs, true_va, cloud_va, mae_clim)
            for k in ("mae_m", "skill", "crps", "coverage_80", "cloud_balanced_acc"):
                metrics[f"cbh/all/{k}"] = float(sc_c[k])
            metrics["val/headline"] = float(np.mean([metrics["global/mean_etage_f1"], metrics["dense/all/miou"],
                                                     max(sc_c["skill"], 0.0)]))
            curves.append({"epoch": engine.epoch + 1, **metrics})
            print(f"  epoch {engine.epoch + 1:2d}: étage F1 {metrics['global/mean_etage_f1']:.3f} | dense mIoU {sc_all.miou:.3f} "
                  f"layer {metrics['dense/almeria/layer_miou']:.3f} | CBH MAE {sc_c['mae_m']:.0f} m skill {sc_c['skill']:+.3f} | "
                  f"headline {metrics['val/headline']:.3f} | {time.perf_counter() - t0:.0f} s", flush=True)
            return metrics

        res = engine.fit(loader_fn, step_fn, eval_fn)
        best = next((c for c in curves if c["epoch"] == res["best_step"] // a.batches_per_epoch), curves[-1])
        (run.dir / "curves.json").write_text(json.dumps(curves, indent=1), encoding="utf-8")
        (run.dir / "grad_cos.json").write_text(json.dumps(cos_log, indent=1), encoding="utf-8")
        run.finish(best=res["best"], best_epoch=best["epoch"], epochs=res["epochs"],
                   **{k.replace("/", "_"): v for k, v in best.items() if k.count("/") == 2 and ("all" in k or "mean" in k)})
    # dense eval helpers need the sub-model interface: DD.evaluate takes a DenseHeads; we passed m.net.dense_heads
    rows = [{"phase": a.phase, "name": a.name, "seed": a.seed, "metric": k, "value": float(v), "best_epoch": best["epoch"],
             "run_id": run.run_id} for k, v in best.items() if k != "epoch"]
    df = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        key = ["name", "seed", "metric"]
        df = pd.concat([old[~old.set_index(key).index.isin(df.set_index(key).index)], df], ignore_index=True)
    df.to_parquet(OUT, index=False)
    write_doc(df, cos_log)
    print(f"done in {time.perf_counter() - t0:.0f} s: headline {res['best']:.3f} at epoch {best['epoch']} -> {DOC}", flush=True)
    return 0


def _single_task_reference() -> dict[str, float]:
    """Headline numbers of the single-task runs for the comparison table."""
    ref = {}
    p = Path("data/stage1_global.parquet")
    if p.exists():
        g = pd.read_parquet(p)
        g = g[(g.name == "indomain_e2s") & (g.split == "val")]
        for r in g.itertuples():
            ref[f"global/{r.dataset}/{r.metric}"] = float(r.value)
    p = Path("data/dense_heads.parquet")
    if p.exists():
        d = pd.read_parquet(p)
        d = d[d.phase == "P074"] if (d.phase == "P074").any() else d[d.phase == "P066"]
        for r in d.itertuples():
            ref[f"dense/{r.dataset}/miou"] = float(r.miou)
            if r.dataset == "almeria" and "layer_miou" in d.columns and pd.notna(r.layer_miou):
                ref["dense/almeria/layer_miou"] = float(r.layer_miou)
    p = Path("data/cbh_head.parquet")
    if p.exists():
        c = pd.read_parquet(p)
        base = c.variant == "with geometry [P075 base]"
        c = c[base] if base.any() else c[c.variant == "with geometry"]
        if len(c):
            for k in ("mae_m", "skill", "crps", "coverage_80", "cloud_balanced_acc"):
                ref[f"cbh/all/{k}"] = float(c[k].iloc[0])
    return ref


def write_doc(df: pd.DataFrame, cos_log: list[dict]) -> None:
    ref = _single_task_reference()
    lines = ["# Stage-4 joint multi-task training on cached features (P076)", "",
             "All heads of the STRATIA model trained together from cached DINOv3 ViT-S/16 features with the P069 sampler "
             "(global / dense / cbh batches), the P070 weighting and the gradient-cosine monitor; the single-task reference "
             "is the best run of "
             "P073 (indomain + Eye2Sky étage), P074 (dense + layers, Kendall) and P075 (CBH base). Validation numbers only.", "",
             "| Head / dataset / metric | Joint | Single-task | Δ |", "|---|---|---|---|"]
    for name, part in df.groupby("name"):
        lines.append(f"| **{name}** (best epoch {int(part.best_epoch.iloc[0])}) | | | |")
        for r in part.sort_values("metric").itertuples():
            if r.metric.count("/") != 2:
                continue
            single = ref.get(r.metric)
            delta = f"{r.value - single:+.3f}" if single is not None else "-"
            lines.append(f"| {r.metric} | {r.value:.3f} | {single:.3f} | {delta} |" if single is not None
                         else f"| {r.metric} | {r.value:.3f} | - | - |")
    if cos_log:
        keys = sorted({k for c in cos_log for k in c if "|" in k})
        lines += ["", "## Task-gradient cosines on the shared parameters (mean over logged steps)", "",
                  "| Pair | Mean cosine | Min |", "|---|---|---|"]
        for k in keys:
            vals = [c[k] for c in cos_log if k in c and c[k] == c[k]]
            if vals:
                lines.append(f"| {k} | {np.mean(vals):+.3f} | {np.min(vals):+.3f} |")
    lines += ["", "Rows in `data/stage4_joint.parquet`; curves and cosines in the run folder."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
