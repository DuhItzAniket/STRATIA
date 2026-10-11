"""Stage-1 training of the global heads on cached features (P073; LODO folds for P082's C1).

    python scripts/train_global_heads.py --name indomain --epochs 20
    python scripts/train_global_heads.py --name lodo_ccsn --lodo ccsn --epochs 20
    python scripts/train_global_heads.py --name indomain_e2s --eye2sky-etage

Model: GeometryEncoder + GlobalHeads (the STRATIA model without dense / CBH heads) on the pooled 8 x 8 DINOv3
ViT-S/16 token grid and cls of every manifest image (P051 cache `dinov3_vits16_512_ffr1-patches-all-pool4`), ray maps
at the same grid for the calibrated Eye2Sky rows, geometry dropout 0.3. Losses (ADR-006): set-valued genus BCE with
per-class known-negative weights, soft étage BCE with per-column weights, CORAL oktas on the Montenegro
distributions, hierarchy consistency (0.1); optional Kendall weighting. Sampler: P069 square-root dataset balance.
Engine: P072 (EMA, clipping, early stopping on the monitored metric).

Per epoch, per dataset, on val and on a fixed train subset (the fit report): genus macro-F1 over native classes
(noisy-OR of the class's genera), étage macro-F1 (threshold 0.5, columns with support), oktas MAE and within-1
(Montenegro), Montenegro agreement with the majority étage set and oktas ±1 (criterion C2 against the human
ceiling, P035). LODO: the held-out source's val units are the val set and étage macro-F1 is the headline (C1).
Results: data/stage1_global.parquet, docs/data/stage1_global.md, the run's metrics.jsonl and curves.json.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.augment.dropout import GeometryDropout  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.data.sampler import SamplerConfig, TaskBatchSampler  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.train import data as D  # noqa: E402
from stratia.train.engine import Engine, EngineConfig  # noqa: E402
from stratia.train.global_eval import GlobalModel, losses_for, predict, score_dataset  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/stage1_global.md")
OUT = Path("data/stage1_global.parquet")










def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="indomain")
    ap.add_argument("--phase", default="P073")
    ap.add_argument("--lodo", default=None, choices=[None, "ccsn", "mgcd", "montenegro", "almeria", "swim"])
    ap.add_argument("--eye2sky-etage", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--batches-per-epoch", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--p-drop", type=float, default=0.3)
    ap.add_argument("--lambda-h", type=float, default=0.1)
    ap.add_argument("--kendall", action="store_true")
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-fraction", type=float, default=1.0, help="P079: subsample the training rows per dataset")
    ap.add_argument("--monitor", default="val/mean_etage_f1")
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = D.global_rows(lodo_fold=a.lodo, eye2sky_etage=a.eye2sky_etage)
    grid = FeatureCache(cache_root, "dinov3_vits16", 512, transform="ffr1-patches-all-pool4")
    cls_cache = FeatureCache(cache_root, "dinov3_vits16", 512)
    have = set(grid.index().sample_id)
    rows = rows[rows.sample_id.isin(have)].reset_index(drop=True)
    if a.train_fraction < 1.0:
        keep = np.ones(len(rows), dtype=bool)
        for _, part in rows[rows.split == "train"].groupby("dataset"):
            units = part.unit.unique()
            drop_units = rng.choice(units, int(round(len(units) * (1 - a.train_fraction))), replace=False)
            keep[part.index[part.unit.isin(drop_units)]] = False
        rows = rows[keep].reset_index(drop=True)
    geo_key = f"{a.name}_{len(rows)}"
    geo = D.GeometryCache.build(rows, data_root, cache_root / "geometry" / f"pool4_{geo_key}.npz")
    ds = D.GlobalDataset(rows, grid.rows_for(rows.sample_id), cls_cache.rows_for(rows.sample_id), grid.load("patches"),
                         cls_cache.load("cls"), geo)
    tr_idx = np.flatnonzero(rows.split == "train")
    va_idx = np.flatnonzero(rows.split == "val")
    print(f"{len(tr_idx):,} train / {len(va_idx):,} val rows; datasets {rows.dataset.value_counts().to_dict()}; "
          f"geometry on {rows.has_geometry.mean():.1%}", flush=True)
    # train subset for the fit report: up to 1,000 rows per dataset
    sub = []
    for _, part in rows.iloc[tr_idx].groupby("dataset"):
        sub.append(rng.choice(part.index.to_numpy(), min(1000, len(part)), replace=False))
    sub_idx = np.concatenate(sub)

    table = rows.iloc[tr_idx].copy()
    table["group_mask"] = [{"global"}] * len(table)
    cfg_s = SamplerConfig(tau=a.tau, group_probs={"global": 1.0}, batch_sizes={"global": a.batch},
                          batches_per_epoch=a.batches_per_epoch)
    sampler = TaskBatchSampler(table, cfg_s, seed=a.seed)
    model = GlobalModel(ds.grid.shape[-1], a.hidden, a.dropout, a.kendall).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * a.batches_per_epoch)
    drop = GeometryDropout(a.p_drop, a.p_drop)

    cfg = OmegaConf.create({"name": a.name, "lodo": a.lodo, "eye2sky_etage": a.eye2sky_etage, "epochs": a.epochs,
                            "batch": a.batch, "batches_per_epoch": a.batches_per_epoch, "lr": a.lr,
                            "weight_decay": a.weight_decay, "dropout": a.dropout,
                            "hidden": a.hidden, "tau": a.tau, "p_drop": a.p_drop, "lambda_h": a.lambda_h, "kendall": a.kendall,
                            "ema": a.ema, "seed": a.seed, "train_fraction": a.train_fraction,
                            "params": sum(p.numel() for p in model.parameters()),
                            "grid_fingerprint": grid.meta()["fingerprint"]})
    curves = []
    with Run(cfg, name=f"global_{a.name}_s{a.seed}", phase=a.phase) as run:
        ecfg = EngineConfig(epochs=a.epochs, accumulate=1, clip_norm=1.0, amp=False, ema_decay=a.ema or None, monitor=a.monitor,
                            mode="max", patience=a.patience, log_every=0, device=device)
        engine = Engine(model, opt, ecfg, scheduler=sched, log=run.log, checkpoint_dir=run.dir / "checkpoints")

        def loader_fn(epoch):
            sampler.set_epoch(epoch)
            for _, idx in sampler:
                items = [ds[tr_idx[i]] for i in idx]
                b = D.collate(items)
                rm, mt, _, _ = drop.batch(b["ray_map"].numpy(), b["meta"].numpy(), rng)
                b["ray_map"], b["meta"] = torch.from_numpy(rm), torch.from_numpy(mt)
                yield {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

        def step_fn(m, b, step):
            out = m(b["tokens"], b["cls"], b["ray_map"], b["meta"])
            parts = losses_for(out, b, a.lambda_h)
            if m.weighting is not None:
                total, _ = m.weighting({k: v for k, v in parts.items() if k != "hier"})
                total = total + parts["hier"]
            else:
                total = sum(parts.values())
            return total, {k: v.detach() for k, v in parts.items()}

        def eval_fn(m):
            metrics = {}
            for split_name, idx in (("val", va_idx), ("train", sub_idx)):
                pred = predict(m, ds, idx, device)
                part = rows.iloc[idx].reset_index(drop=True)
                per = {}
                for ds_name in part.dataset.unique():
                    sel = (part.dataset == ds_name).to_numpy()
                    per[ds_name] = score_dataset(part[sel].reset_index(drop=True), {k: v[sel] for k, v in pred.items()})
                for ds_name, sc in per.items():
                    for k, v in sc.items():
                        metrics[f"{split_name}/{ds_name}/{k}"] = float(v)
                e = [sc["etage_f1"] for sc in per.values() if "etage_f1" in sc]
                g = [sc["genus_f1"] for sc in per.values() if "genus_f1" in sc]
                metrics[f"{split_name}/mean_etage_f1"] = float(np.mean(e)) if e else 0.0
                metrics[f"{split_name}/mean_genus_f1"] = float(np.mean(g)) if g else 0.0
            curves.append({"epoch": engine.epoch + 1, **metrics})
            line = " ".join(f"{k.split('/')[1]}:{v:.3f}" for k, v in metrics.items()
                            if k.startswith("val/") and k.endswith("etage_f1"))
            print(f"  epoch {engine.epoch + 1:2d}: val étage F1 {line} | mean {metrics['val/mean_etage_f1']:.3f} "
                  f"genus {metrics['val/mean_genus_f1']:.3f} | train mean étage {metrics['train/mean_etage_f1']:.3f} | "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
            return metrics

        res = engine.fit(loader_fn, step_fn, eval_fn)
        best_epoch = next((c for c in curves if c["epoch"] == res["best_step"] // a.batches_per_epoch), curves[-1])
        (run.dir / "curves.json").write_text(json.dumps(curves, indent=1), encoding="utf-8")
        run.finish(best=res["best"], best_epoch=best_epoch["epoch"], epochs=res["epochs"],
                   **{k: v for k, v in best_epoch.items() if k.startswith("val/") and ("mean" in k or "c2" in k)})
    # table rows: the best epoch's val and train metrics per dataset
    out_rows = []
    for k, v in best_epoch.items():
        if "/" in k:
            split_name, rest = k.split("/", 1)
            ds_name, metric = rest.split("/", 1) if "/" in rest else ("all", rest)
            out_rows.append({"phase": a.phase, "name": a.name, "lodo": a.lodo or "", "seed": a.seed,
                             "train_fraction": a.train_fraction,
                             "split": split_name, "dataset": ds_name, "metric": metric, "value": float(v),
                             "best_epoch": best_epoch["epoch"], "run_id": run.run_id})
    df = pd.DataFrame(out_rows)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        key = ["name", "seed", "train_fraction", "split", "dataset", "metric"]
        df = pd.concat([old[~old.set_index(key).index.isin(df.set_index(key).index)], df], ignore_index=True)
    df.to_parquet(OUT, index=False)
    write_doc(df)
    print(f"done in {time.perf_counter() - t0:.0f} s: best {a.monitor} {res['best']:.3f} at epoch {best_epoch['epoch']} "
          f"-> {DOC}", flush=True)
    return 0


def write_doc(df: pd.DataFrame) -> None:
    lines = ["# Stage-1 global heads on cached features (P073)", "",
             "GeometryEncoder + GlobalHeads on the pooled 8 x 8 DINOv3 ViT-S/16 grid; grouped in-domain splits (train -> val) "
             "or LODO folds (val = the held-out source's val units). Genus macro-F1 over native classes (noisy-OR of the "
             "class's genera), étage macro-F1 (threshold 0.5, columns with support), Montenegro oktas MAE / within-1 and the "
             "agreement with the majority "
             "étage set and oktas ±1 as a ratio to the P035 human ceiling (criterion C2 asks ≥ 0.90). "
             "'train' rows are a fixed subset "
             "of the training rows (the fit report's gap).", ""]
    for name, part in df.groupby("name"):
        lines += [f"## {name} (best epoch {int(part.best_epoch.iloc[0])}, lodo = {part.lodo.iloc[0] or '-'})", "",
                  "| Dataset | Split | " + " | ".join(sorted(part.metric.unique())) + " |",
                  "|---|---|" + "---|" * part.metric.nunique()]
        metrics = sorted(part.metric.unique())
        for (ds_name, split), g in part.groupby(["dataset", "split"]):
            vals = {r.metric: r.value for r in g.itertuples()}
            lines.append(f"| {ds_name} | {split} | " + " | ".join(f"{vals[m]:.3f}" if m in vals else "-" for m in metrics) + " |")
        lines.append("")
    lines += ["Rows in `data/stage1_global.parquet`; per-epoch curves in each run's `curves.json`."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
