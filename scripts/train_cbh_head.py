"""Train the CBH head on cached DINOv3 tokens with ray maps (P067).

    python scripts/train_cbh_head.py --epochs 10 [--batch 128] [--lr 1e-3] [--p-drop 0.3]

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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.augment.dropout import GeometryDropout  # noqa: E402
from stratia.baselines import cbh as C  # noqa: E402
from stratia.data.eye2sky import load_mask, mask_path, station_calibrations  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.geometry.pose import load_pose_records, pose_for  # noqa: E402
from stratia.geometry.raymap import meta_vector, patch_rays, ray_map  # noqa: E402
from stratia.geometry.sun import sun_position  # noqa: E402
from stratia.model import cbh_bins as B  # noqa: E402
from stratia.model import losses as L  # noqa: E402
from stratia.model.cbh import CbhHead  # noqa: E402
from stratia.model.skyray import GeometryEncoder  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/cbh_head.md")
OUT = Path("data/cbh_head.parquet")
SIZE, PATCH = 512, 64            # the pooled grid: 8 x 8 cells of 64 px


def load_pairs(grid: FeatureCache) -> pd.DataFrame:
    d = pd.read_parquet("data/cbh_pairs.parquet")
    d = d[d.paired & d.daytime & d.zenith_inside_mask & d.split.isin(["train", "val"]) & (d.utc.dt.minute % 10 == 0)].copy()
    d["sample_id"] = "eye2sky:" + d.image_file
    cached = set(grid.index().sample_id)
    d = d[d.sample_id.isin(cached)].reset_index(drop=True)
    d["cloud"] = d.label != "none"
    d["height_m"] = np.where(d.cloud, d.cbh_m, np.nan)
    return d


def ray_maps_for(pairs: pd.DataFrame, data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """(ray_maps [n, 4, 8, 8], metas [n, 3]) per pair from the station calibration, the fitted pose and the Sun."""
    meta_root = data_root / "Eye2Sky" / "asi_meta"
    records = load_pose_records()
    rms = np.zeros((len(pairs), 4, SIZE // PATCH, SIZE // PATCH), dtype=np.float32)
    metas = np.zeros((len(pairs), 3), dtype=np.float32)
    for (station, cid), part in pairs.groupby(["station", "calib_id"]):
        cal = next(c for c in station_calibrations(meta_root, station) if c.file == cid)
        pose = pose_for(cid, None, records)
        if pose is None:
            continue
        mask = load_mask(mask_path(meta_root, cal)) > 0
        rays = patch_rays(cal.model, None, SIZE, PATCH, mask)
        sun = sun_position(part.utc, cal.latitude, cal.longitude, cal.altitude)
        for i, az, zen in zip(part.index, sun.azimuth.to_numpy(), sun.apparent_zenith.to_numpy(), strict=True):
            rms[i] = ray_map(cal.model, pose, float(az), float(zen), out_size=SIZE, patch=PATCH, rays=rays)
            metas[i] = meta_vector(float(zen))
    return rms, metas


class PairDataset(Dataset):
    def __init__(self, rows_grid, rows_cls, grid, cls, rms, metas, heights):
        self.rows_grid, self.rows_cls, self.grid, self.cls = rows_grid, rows_cls, grid, cls
        self.rms, self.metas, self.heights = rms, metas, heights

    def __len__(self):
        return len(self.rows_grid)

    def __getitem__(self, i):
        return (torch.from_numpy(np.asarray(self.grid[self.rows_grid[i]], dtype=np.float32)),
                torch.from_numpy(np.asarray(self.cls[self.rows_cls[i]], dtype=np.float32)),
                torch.from_numpy(self.rms[i]), torch.from_numpy(self.metas[i]),
                torch.tensor(self.heights[i], dtype=torch.float32))


class Model(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.geo = GeometryEncoder(d_model)
        self.head = CbhHead(d_model)

    def forward(self, tokens, cls, rm, meta):
        t, c = self.geo(tokens, cls, rm, meta)
        return self.head(t, c, rm)


@torch.no_grad()
def predict(model: Model, dl: DataLoader, device: str, null_geometry: bool = False) -> np.ndarray:
    model.eval()
    out = []
    for tok, cls, rm, meta, _ in dl:
        if null_geometry:
            rm, meta = torch.zeros_like(rm), torch.zeros_like(meta)
        out.append(model(tok.to(device), cls.to(device), rm.to(device), meta.to(device))["cbh_probs"].float().cpu().numpy())
    return np.concatenate(out)


def score(probs: np.ndarray, heights: np.ndarray, cloud: np.ndarray, mae_clim: float) -> dict:
    cloudy = ~np.isnan(heights)
    pred = B.median_height_m(probs)
    pred_c = np.where(np.isnan(pred), B.centres_m()[probs[:, :B.K_BINS].argmax(1)], pred)   # a height for every cloudy row
    sc = C.score_heights(pred_c[cloudy], heights[cloudy], mae_clim)
    lo, hi = B.interval_m(probs[cloudy])
    cover = float(np.mean((heights[cloudy] >= lo) & (heights[cloudy] <= hi)))
    target = B.soft_target(heights)
    crps_c = ((np.cumsum(probs[cloudy, :B.K_BINS] / probs[cloudy, :B.K_BINS].sum(1, keepdims=True), 1)
               - np.cumsum(target[cloudy, :B.K_BINS], 1)) ** 2).mean()
    pred_cloud = probs[:, B.NO_CLOUD] < 0.5
    bal = C.balanced_accuracy(cloud, pred_cloud)
    # mode-collapse guard: share of predictions in the most common predicted bin vs the labels' most common bin
    pred_bins = probs[cloudy, :B.K_BINS].argmax(1)
    true_bins = B.bin_of(heights[cloudy])
    top_pred = float(np.bincount(pred_bins, minlength=B.K_BINS).max() / len(pred_bins))
    top_true = float(np.bincount(true_bins, minlength=B.K_BINS).max() / len(true_bins))
    def ent(b):
        p = np.bincount(b, minlength=B.K_BINS) / len(b) + 1e-12
        return float(-(p @ np.log(p)))
    return {**sc.row(), "crps": float(crps_c), "coverage_80": cover, "cloud_balanced_acc": bal,
            "top_bin_share_pred": top_pred, "top_bin_share_true": top_true,
            "entropy_ratio": ent(pred_bins) / max(ent(true_bins), 1e-9), "n_cloudy": int(cloudy.sum())}


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
    g_rows, c_rows = grid.rows_for(pairs.sample_id), pooled.rows_for(pairs.sample_id)
    g_arr, c_arr = grid.load("patches"), pooled.load("cls")
    heights = pairs.height_m.to_numpy().astype(np.float32)
    mk = lambda m: PairDataset(g_rows[m], c_rows[m], g_arr, c_arr, rms[m], metas[m], heights[m])  # noqa: E731
    dl_tr = DataLoader(mk(tr), batch_size=a.batch, shuffle=True, drop_last=True)
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
                            "seed": a.seed,
                            "p_drop": a.p_drop, "sigma_log": a.sigma_log, "lambda_crps": a.lambda_crps,
                            "params": sum(p.numel() for p in model.parameters()), "grid": "8x8", "mae_clim": float(mae_clim)})
    best, best_state = None, None
    with Run(cfg, name=f"cbh_head_s{a.seed}{a.variant_suffix.replace(' ', '_')}", phase=a.phase) as run:
        for epoch in range(a.epochs):
            model.train()
            total, n = 0.0, 0
            for tok, cls, rm, meta, h in dl_tr:
                rm_np, meta_np, _, _ = drop.batch(rm.numpy(), meta.numpy(), rng)
                out = model(tok.to(device), cls.to(device), torch.from_numpy(rm_np).to(device),
                            torch.from_numpy(meta_np).to(device))
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
        null = score(predict(model, dl_va, device, null_geometry=True), true_va, cloud_va, mae_clim)
        run.finish(val_mae_m=best["mae_m"], val_skill=best["skill"], val_crps=best["crps"],
                   val_coverage_80=best["coverage_80"],
                   null_geometry_mae_m=null["mae_m"], best_epoch=best["epoch"])
    rows = [{"phase": a.phase, "model": "cbh_head", "variant": "with geometry" + a.variant_suffix, "seed": a.seed,
             "best_epoch": best["epoch"],
             **{k: v for k, v in best.items() if k != "epoch"}, "run_id": run.run_id},
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
    print(f"done in {time.perf_counter() - t0:.0f} s: val MAE {best['mae_m']:.0f} m, skill {best['skill']:+.3f} "
          f"(geometry zeroed: {null['mae_m']:.0f} m) -> {DOC}", flush=True)
    return 0


def write_doc(df: pd.DataFrame, mae_clim: float) -> None:
    base_path = Path("data/baseline_cbh.parquet")
    base = pd.read_parquet(base_path) if base_path.exists() else pd.DataFrame()
    lines = ["# CBH head on cached DINOv3 tokens (P067)", "",
             f"P047 pairs, 10-minute cadence, train -> val days (test locked); GeometryEncoder + CbhHead on the 8 x 8 pooled "
             f"token grid with the contract's ray map and Sun meta (geometry dropout 0.3 in training). Skill = 1 - MAE / "
             f"{mae_clim:.0f} m (train-median climatology). 'geometry zeroed' scores the same weights with the ray map and meta "
             "set to the contract's unknown state. The P058 baselines are listed for reference.", "",
             "| Phase | Model | Variant | MAE (m) | Bias (m) | Skill | ≤500 m | CRPS | 80 % coverage | Cloud bal. acc | "
             "Top-bin share pred / true | Entropy ratio | Best epoch |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values(["phase", "variant"]).iterrows():
        lines.append(f"| {r.phase} | {r.model} | {r.variant} | {r.mae_m:.0f} | {r.bias_m:+.0f} | {r.skill:+.3f} | "
                     f"{r.within_500m:.2f} | "
                     f"{r.crps:.4f} | {r.coverage_80:.2f} | {r.cloud_balanced_acc:.3f} | {r.top_bin_share_pred:.2f} / "
                     f"{r.top_bin_share_true:.2f} | {r.entropy_ratio:.2f} | {int(r.best_epoch)} |")
    if len(base):
        for _, r in base.iterrows():
            acc = f"{r.cloud_balanced_acc:.3f}" if pd.notna(r.get("cloud_balanced_acc")) else "-"
            lines.append(f"| P058 | {r.baseline} | - | {r.mae_m:.0f} | {r.bias_m:+.0f} | {r.skill:+.3f} | {r.within_500m:.2f} | "
                         f"- | - | {acc} | - | - | - |")
    lines += ["", "Rows in `data/cbh_head.parquet`; curves in the run's `metrics.jsonl`."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
