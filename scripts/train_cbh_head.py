"""Train the CBH head on cached DINOv3 tokens with ray maps (P067; P075 variants).

    python scripts/train_cbh_head.py --epochs 10 [--batch 128] [--lr 1e-3] [--p-drop 0.3]
    python scripts/train_cbh_head.py --train-stations OLDLR --variant-suffix " (OLDLR only)"   # P075: WESTE held out
    python scripts/train_cbh_head.py --synthetic --variant-suffix " (+synthetic)"                 # P075: P049 views in train
    python scripts/train_cbh_head.py --day-weight --variant-suffix " (day-weighted)"             # P075: overcast-day guard

Data: the P047 pairs at a 10-minute cadence (train -> val; test days locked), tokens from the P051 caches
`dinov3_vits16_512_ffr1` (cls) and `dinov3_vits16_512_ffr1-patches-cbh-pool4` (8 x 8 x 384 pooled grid), the
contract's ray map at the 8 x 8 grid (patch 64) from the station calibration, the fitted pose and the Sun, and the
Sun meta vector; geometry dropout (P050) at batch time. Model: GeometryEncoder (P062) + CbhHead (ADR-005 §3);
loss: ordinal CE + CRPS (ADR-006 §6). The layer-consistency term needs the étage head and is wired in the full
model (P071); here the head trains alone.

Scores on the val days: MAE / bias / skill of the median height against the train-median climatology (as P058),
CRPS, 80 % interval coverage, cloud-presence balanced accuracy, the prediction histogram against the label
histogram (mode-collapse guard), and the same model scored with the ray map and meta zeroed (geometry ablation).
Results: data/cbh_head.parquet, docs/data/cbh_head.md, one registry run.
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
from stratia.augment.dropout import GeometryDropout  # noqa: E402
from stratia.baselines import cbh as C  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.model import losses as L  # noqa: E402
from stratia.train.cbh_data import (  # noqa: E402
    Model,
    PairDataset,
    load_pairs,
    predict,
    ray_maps_for,
    score,
    synthetic_views,
)
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/cbh_head.md")
OUT = Path("data/cbh_head.parquet")


















def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="P067")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-drop", type=float, default=0.3)
    ap.add_argument("--sigma-log", type=float, default=0.15)
    ap.add_argument("--lambda-crps", type=float, default=1.0)
    ap.add_argument("--variant-suffix", default="", help="appended to the variant names (ablation runs)")
    ap.add_argument("--train-stations", nargs="+", default=None, help="P075: train on these stations only (val on all)")
    ap.add_argument("--synthetic", action="store_true", help="P075: add the labelled P049 synthetic consumer views to train")
    ap.add_argument("--day-weight", action="store_true", help="P075: weight rows by 1 / day_label_share (overcast-day guard)")
    ap.add_argument("--train-fraction", type=float, default=1.0, help="P079: keep this share of the training days")
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = FeatureCache(cache_root, "dinov3_vits16", 512, transform="ffr1-patches-cbh-pool4")
    pooled = FeatureCache(cache_root, "dinov3_vits16", 512)
    pairs = load_pairs(grid)
    rms, metas = ray_maps_for(pairs, data_root)
    valid_geo = rms[:, 3].reshape(len(pairs), -1).max(1) > 0
    print(f"{len(pairs):,} pairs; ray maps valid for {valid_geo.mean():.1%}", flush=True)
    tr, va = (pairs.split == "train").to_numpy(), (pairs.split == "val").to_numpy()
    if a.train_stations:
        tr = tr & pairs.station.isin(a.train_stations).to_numpy()      # a fresh array: the split mask is read-only
    if a.train_fraction < 1.0:
        days = np.array(sorted(pairs.day[tr].astype(str).unique()))
        keep_days = set(rng.choice(days, int(round(len(days) * a.train_fraction)), replace=False))
        tr = tr & pairs.day.astype(str).isin(keep_days).to_numpy()
    weights = (1.0 / np.clip(pairs.day_label_share.to_numpy(), 0.2, 1.0)).astype(np.float32) if a.day_weight else None
    g_rows, c_rows = grid.rows_for(pairs.sample_id), pooled.rows_for(pairs.sample_id)
    g_arr, c_arr = grid.load("patches"), pooled.load("cls")
    heights = pairs.height_m.to_numpy().astype(np.float32)
    def mk(m):
        return PairDataset(g_rows[m], c_rows[m], g_arr, c_arr, rms[m], metas[m], heights[m],
                           None if weights is None else weights[m])

    ds_tr = mk(tr)
    if a.synthetic:
        syn = synthetic_views(cache_root, device)
        if syn is not None:
            print(f"  + {len(syn):,} labelled synthetic views in train", flush=True)
            ds_tr = torch.utils.data.ConcatDataset([ds_tr, syn])
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, drop_last=True)
    dl_va = DataLoader(mk(va), batch_size=256, shuffle=False)
    clim = C.climatology(pairs[tr])
    true_va, cloud_va = heights[va], pairs.cloud.to_numpy()[va]
    cloudy_va = ~np.isnan(true_va)
    mae_clim = C.score_heights(np.full(int(cloudy_va.sum()), clim), true_va[cloudy_va]).mae
    model = Model(g_arr.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl_tr), pct_start=0.1)
    drop = GeometryDropout(a.p_drop, a.p_drop)
    cfg = OmegaConf.create({"model": "geometry_encoder+cbh_head", "epochs": a.epochs, "batch": a.batch, "lr": a.lr,
                            "seed": a.seed, "train_fraction": a.train_fraction,
                            "p_drop": a.p_drop, "sigma_log": a.sigma_log, "lambda_crps": a.lambda_crps,
                            "params": sum(p.numel() for p in model.parameters()), "grid": "8x8", "mae_clim": float(mae_clim)})
    best, best_state = None, None
    with Run(cfg, name=f"cbh_head_s{a.seed}{a.variant_suffix.replace(' ', '_')}", phase=a.phase) as run:
        for epoch in range(a.epochs):
            model.train()
            total, n = 0.0, 0
            for tok, cls, rm, meta, h, w in dl_tr:
                rm_np, meta_np, _, _ = drop.batch(rm.numpy(), meta.numpy(), rng)
                out = model(tok.to(device), cls.to(device), torch.from_numpy(rm_np).to(device),
                            torch.from_numpy(meta_np).to(device))
                if a.day_weight:
                    per = torch.stack([L.cbh_loss(out["cbh_logits"][j:j + 1], h[j:j + 1].to(device),
                                                  torch.ones(1, dtype=torch.bool, device=device), a.sigma_log, a.lambda_crps)[0]
                                       for j in range(len(h))])
                    wt = w.to(device)
                    loss, parts = (per * wt).sum() / wt.sum(), {}
                else:
                    loss, parts = L.cbh_loss(out["cbh_logits"], h.to(device), torch.ones(len(h), dtype=torch.bool, device=device),
                                             a.sigma_log, a.lambda_crps)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += float(loss) * len(h)
                n += len(h)
            probs = predict(model, dl_va, device)
            sc = score(probs, true_va, cloud_va, mae_clim)
            rec = {"train_loss": total / max(n, 1), "val_mae_m": sc["mae_m"], "val_skill": sc["skill"],
                   "val_crps": sc["crps"],
                   "val_coverage_80": sc["coverage_80"], "val_cloud_bal_acc": sc["cloud_balanced_acc"],
                   "sigma_deg": float(np.degrees(model.head.log_sigma.exp().item())), "alpha": float(model.geo.alpha.item()),
                   "seconds": time.perf_counter() - t0}
            run.log(epoch + 1, **rec)
            print(f"  epoch {epoch + 1:2d}/{a.epochs}: loss {rec['train_loss']:.3f} | val MAE {sc['mae_m']:.0f} m skill "
                  f"{sc['skill']:+.3f} CRPS {sc['crps']:.4f} cover80 {sc['coverage_80']:.2f} "
                  f"cloud-acc {sc['cloud_balanced_acc']:.3f} "
                  f"| sigma {rec['sigma_deg']:.1f} deg alpha {rec['alpha']:.3f} | {rec['seconds']:.0f} s", flush=True)
            if best is None or sc["mae_m"] < best["mae_m"]:
                best = dict(sc, epoch=epoch + 1)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        (run.dir / "checkpoints").mkdir(exist_ok=True)
        torch.save({"model": best_state, "cfg": OmegaConf.to_container(cfg), "best_epoch": best["epoch"]},
                   run.dir / "checkpoints" / "best.pt")
        probs_best = predict(model, dl_va, device)
        np.savez_compressed(run.dir / "val_predictions.npz", probs=probs_best, height_m=true_va, cloud=cloud_va,
                            station=pairs.station.to_numpy()[va].astype(str), day=pairs.day.to_numpy()[va].astype(str),
                            sample_id=pairs.sample_id.to_numpy()[va].astype(str), mae_clim=mae_clim)
        stations_va = pairs.station.to_numpy()[va]
        per_station = {}
        for st in sorted(set(stations_va)):
            m = stations_va == st
            if (~np.isnan(true_va[m])).sum() > 20:
                sc_st = score(probs_best[m], true_va[m], cloud_va[m], mae_clim)
                per_station[st] = sc_st
                print(f"  station {st}: MAE {sc_st['mae_m']:.0f} m skill {sc_st['skill']:+.3f} "
                      f"(n cloudy {sc_st['n_cloudy']})", flush=True)
        null = score(predict(model, dl_va, device, null_geometry=True), true_va, cloud_va, mae_clim)
        run.finish(val_mae_m=best["mae_m"], val_skill=best["skill"], val_crps=best["crps"],
                   val_coverage_80=best["coverage_80"],
                   null_geometry_mae_m=null["mae_m"], best_epoch=best["epoch"])
    rows = [{"phase": a.phase, "model": "cbh_head", "variant": "with geometry" + a.variant_suffix, "seed": a.seed,
             "best_epoch": best["epoch"],
             **{k: v for k, v in best.items() if k != "epoch"}, "run_id": run.run_id,
             **{f"{st}_{k}": v for st, sc_st in per_station.items() for k, v in sc_st.items()
                if k in ("mae_m", "skill", "n_cloudy")}},
            {"phase": a.phase, "model": "cbh_head", "variant": "geometry zeroed" + a.variant_suffix, "seed": a.seed,
             "best_epoch": best["epoch"],
             **null, "run_id": run.run_id}]
    df = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        key = ["phase", "model", "variant", "seed"]
        df = pd.concat([old[~old.set_index(key).index.isin(df.set_index(key).index)], df], ignore_index=True)
    df.to_parquet(OUT, index=False)
    write_doc(df, mae_clim)
    run_name = f"cbh_head_s{a.seed}{a.variant_suffix.replace(' ', '_')}"
    print(f"[{run_name}] per-station: " + "; ".join(f"{st} MAE {sc_st['mae_m']:.0f} m skill {sc_st['skill']:+.3f}"
                                                   for st, sc_st in per_station.items()), flush=True)
    print(f"done in {time.perf_counter() - t0:.0f} s: val MAE {best['mae_m']:.0f} m, skill {best['skill']:+.3f} "
          f"(geometry zeroed: {null['mae_m']:.0f} m) -> {DOC}", flush=True)
    return 0


def _st(r, station: str) -> str:
    k = f"{station}_mae_m"
    return f"{r[k]:.0f} / {r[f'{station}_skill']:+.3f}" if k in r and pd.notna(r.get(k)) else "-"


def write_doc(df: pd.DataFrame, mae_clim: float) -> None:
    base_path = Path("data/baseline_cbh.parquet")
    base = pd.read_parquet(base_path) if base_path.exists() else pd.DataFrame()
    lines = ["# CBH head on cached DINOv3 tokens (P067)", "",
             f"P047 pairs, 10-minute cadence, train -> val days (test locked); GeometryEncoder + CbhHead on the 8 x 8 pooled "
             f"token grid with the contract's ray map and Sun meta (geometry dropout 0.3 in training). Skill = 1 - MAE / "
             f"{mae_clim:.0f} m (train-median climatology). 'geometry zeroed' scores the same weights with the ray map and meta "
             "set to the contract's unknown state. The P058 baselines are listed for reference.", "",
             "| Phase | Model | Variant | MAE (m) | Bias (m) | Skill | ≤500 m | CRPS | 80 % coverage | Cloud bal. acc | "
             "Top-bin share pred / true | Entropy ratio | Best epoch | OLDLR MAE / skill | WESTE MAE / skill |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values(["phase", "variant"]).iterrows():
        lines.append(f"| {r.phase} | {r.model} | {r.variant} | {r.mae_m:.0f} | {r.bias_m:+.0f} | {r.skill:+.3f} | "
                     f"{r.within_500m:.2f} | "
                     f"{r.crps:.4f} | {r.coverage_80:.2f} | {r.cloud_balanced_acc:.3f} | {r.top_bin_share_pred:.2f} / "
                     f"{r.top_bin_share_true:.2f} | {r.entropy_ratio:.2f} | {int(r.best_epoch)} | {_st(r, 'OLDLR')} | "
                     f"{_st(r, 'WESTE')} |")
    if len(base):
        for _, r in base.iterrows():
            acc = f"{r.cloud_balanced_acc:.3f}" if pd.notna(r.get("cloud_balanced_acc")) else "-"
            lines.append(f"| P058 | {r.baseline} | - | {r.mae_m:.0f} | {r.bias_m:+.0f} | {r.skill:+.3f} | {r.within_500m:.2f} | "
                         f"- | - | {acc} | - | - | - | - | - |")
    lines += ["", "Rows in `data/cbh_head.parquet`; curves in the run's `metrics.jsonl`."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
