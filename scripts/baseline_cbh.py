"""Cloud-base-height baselines (P058) on the P047 pairs, 10-minute cadence, train -> val (test locked).

    python scripts/baseline_cbh.py [--epochs 6] [--skip-cnn]

Baselines: climatology (train median; conditional by month x 3-hour bin), étage lookup (frozen DINOv3 étage probe
trained on CCSN / MGCD / Montenegro / Almería -> train-split median base per étage), ridge regression of log height
on the frozen DINOv3 ViT-S/16 pooled embedding at 512 px, and a ceilometer-supervised ResNet-18 regressing log
height from the image (224 px). Cloud presence (none vs cloud) from a logistic probe on the same embedding.
Zero-shot DA3 / MoGe-2 are not run: they mask the sky by design (research design §1). Table in
docs/data/baseline_cbh.md; rows in data/baseline_cbh.parquet; one run per baseline in runs/registry.csv.
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
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.baselines import cbh as C  # noqa: E402
from stratia.baselines import probes as P  # noqa: E402
from stratia.baselines import tasks as T  # noqa: E402
from stratia.baselines.train_cnn import ImageClassDataset, build_model  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/baseline_cbh.md")
OUT = Path("data/baseline_cbh.parquet")


def load_pairs(cache: FeatureCache) -> pd.DataFrame:
    d = pd.read_parquet("data/cbh_pairs.parquet")
    d = d[d.paired & d.daytime & d.zenith_inside_mask & d.split.isin(["train", "val"]) & (d.utc.dt.minute % 10 == 0)].copy()
    d["sample_id"] = "eye2sky:" + d.image_file
    cached = set(cache.index().sample_id)
    d = d[d.sample_id.isin(cached)].reset_index(drop=True)
    d["cloud"] = d.label != "none"
    return d


def etage_probe_predictions(cache: FeatureCache, tables: T.Tables, x_eye: np.ndarray) -> np.ndarray:
    """Étage of each Eye2Sky frame from a multi-label probe trained on every étage-labelled source (train + val)."""
    e = T.etage_table(tables)
    e = e[e.source.isin(T.LODO_FOLDS_WITH_ETAGE)]
    x = cache.features(e.sample_id, "cls")
    y = T.etage_matrix(e.etage_set)
    from sklearn.linear_model import LogisticRegression

    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-6
    probs = np.zeros((len(x_eye), 3))
    for j in range(3):
        clf = LogisticRegression(C=0.1, max_iter=2000).fit((x - mu) / sd, y[:, j])
        probs[:, j] = clf.predict_proba((x_eye - mu) / sd)[:, 1]
    names = np.array(["low", "mid", "high"])
    pred = names[probs.argmax(1)]
    pred[probs.max(1) < 0.5] = "none"
    return pred


def train_cnn_regressor(train: pd.DataFrame, val: pd.DataFrame, cache_root: Path, epochs: int, workers: int, seed: int, run: Run,
                        device: str) -> np.ndarray:
    """ResNet-18 regressing log10 height on cloudy train pairs; returns val predictions in metres (cloudy val rows)."""
    torch.manual_seed(seed)
    tr = train[train.cbh_m.notna()]
    va = val[val.cbh_m.notna()]
    y_tr = np.log10(tr.cbh_m.to_numpy())
    ds_tr = ImageClassDataset(tr.image_file.tolist(), np.arange(len(tr)), cache_root, 224, True, (0.8, 1.0), seed)
    ds_va = ImageClassDataset(va.image_file.tolist(), np.arange(len(va)), cache_root, 224, False)
    dl_tr = DataLoader(ds_tr, batch_size=48, shuffle=True, num_workers=workers, pin_memory=device == "cuda", drop_last=True,
                       persistent_workers=workers > 0)
    dl_va = DataLoader(ds_va, batch_size=96, num_workers=workers, persistent_workers=workers > 0)
    model = build_model("resnet18", 1, True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * len(dl_tr), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
    best_mae, best_pred = None, None
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, idx in dl_tr:
            x = (x.to(device).float() / 255.0 - mean) / std
            y = y_tr_t[idx].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = nn.functional.smooth_l1_loss(model(x).float().squeeze(1), y, beta=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss) * len(y)
            n += len(y)
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in dl_va:
                x = (x.to(device).float() / 255.0 - mean) / std
                preds.append(model(x).float().squeeze(1).cpu().numpy())
        pred_m = 10 ** np.concatenate(preds)
        mae = float(np.abs(pred_m - va.cbh_m.to_numpy()).mean())
        run.log(epoch + 1, train_loss=total / max(n, 1), val_mae_m=mae, seconds=time.perf_counter() - t0)
        print(f"  cnn epoch {epoch + 1}/{epochs}: loss {total / max(n, 1):.4f} val MAE {mae:.0f} m", flush=True)
        if best_mae is None or mae < best_mae:
            best_mae, best_pred = mae, pred_m
    return best_pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--phase", default="P058")
    a = ap.parse_args()
    t0 = time.perf_counter()
    paths = load_paths()
    cache_root = Path(paths["cache_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.8)
    cache = FeatureCache(cache_root, "dinov3_vits16", 512)
    pairs = load_pairs(cache)
    train, val = pairs[pairs.split == "train"], pairs[pairs.split == "val"]
    val_cloudy = val[val.cbh_m.notna()]
    true_m = val_cloudy.cbh_m.to_numpy()
    print(f"{len(train):,} train / {len(val):,} val pairs (10-min cadence); cloudy val {len(val_cloudy):,}; "
          f"label mix val {val.label.value_counts().to_dict()}", flush=True)
    tables = T.load_tables()
    rows = []

    def record(name: str, pred_m: np.ndarray, mae_clim: float, cloud_pred=None, extra=None):
        sc = C.score_heights(pred_m, true_m, mae_clim)
        row = {"baseline": name, **sc.row()}
        if cloud_pred is not None:
            row["cloud_balanced_acc"] = C.balanced_accuracy(val.cloud.to_numpy(), cloud_pred)
        row.update(extra or {})
        rows.append(row)
        print(f"{name}: MAE {sc.mae:.0f} m, bias {sc.bias:+.0f} m, skill {sc.skill:+.3f}, within 500 m {sc.within[500]:.2f}",
              flush=True)
        return row

    # 1. climatology
    clim = C.climatology(train)
    with Run(OmegaConf.create({"baseline": "climatology"}), name="cbh_climatology", phase=a.phase) as run:
        sc_clim = C.score_heights(np.full(len(true_m), clim), true_m)
        mae_clim = sc_clim.mae
        record("climatology_median", np.full(len(true_m), clim), mae_clim, extra={"predicted_m": clim})
        cond = C.conditional_climatology(train)
        pred = C.apply_conditional(cond, val_cloudy, clim)
        record("climatology_month_hour", pred, mae_clim)
        run.finish(mae_clim_m=mae_clim, median_m=clim)
    # 2. étage lookup
    x_val = cache.features(val.sample_id, "cls")
    x_val_cloudy = cache.features(val_cloudy.sample_id, "cls")
    cfg_lookup = OmegaConf.create({"baseline": "etage_lookup", "probe": "dinov3_vits16_512_cls"})
    with Run(cfg_lookup, name="cbh_etage_lookup", phase=a.phase) as run:
        medians = C.etage_medians(train)
        pred_etage = etage_probe_predictions(cache, tables, x_val_cloudy)
        pred = C.lookup_heights(pred_etage, medians, clim)
        etage_all = etage_probe_predictions(cache, tables, x_val)
        row = record("etage_lookup", pred, mae_clim, cloud_pred=etage_all != "none", extra={"medians": str(medians)})
        run.finish(mae_m=row["mae_m"], skill=row["skill"])
    # 3. ridge on frozen features (+ logistic cloud presence)
    x_tr = cache.features(train.sample_id, "cls")
    with Run(OmegaConf.create({"baseline": "ridge_dinov3_vits16_512"}), name="cbh_ridge_dinov3", phase=a.phase) as run:
        tr_c = train.cbh_m.notna().to_numpy()
        model, mu, sd, alpha = C.fit_ridge_log_height(x_tr[tr_c], train.cbh_m.to_numpy()[tr_c], x_val=x_val_cloudy,
                                                      cbh_val=true_m)
        pred = C.predict_ridge(model, mu, sd, x_val_cloudy)
        cloud_probe = P.linear_probe(x_tr, train.cloud.to_numpy().astype(str), x_val, val.cloud.to_numpy().astype(str))
        row = record("ridge_dinov3_vits16_512", pred, mae_clim, cloud_pred=cloud_probe.val_pred == "True",
                     extra={"alpha": alpha})
        run.finish(mae_m=row["mae_m"], skill=row["skill"], cloud_balanced_acc=row["cloud_balanced_acc"])
    # 4. ceilometer-supervised CNN
    if not a.skip_cnn:
        cfg = OmegaConf.create({"baseline": "cnn_resnet18_logheight", "epochs": a.epochs, "seed": a.seed, "size": 224,
                                "lr": 3e-4})
        with Run(cfg, name="cbh_cnn_resnet18", phase=a.phase) as run:
            pred = train_cnn_regressor(train, val, cache_root, a.epochs, a.workers, a.seed, run, device)
            row = record("cnn_resnet18_logheight", pred, mae_clim)
            run.finish(mae_m=row["mae_m"], skill=row["skill"])
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    write_doc(df, len(train), len(val), len(val_cloudy), mae_clim)
    print(f"done in {time.perf_counter() - t0:.0f} s -> {DOC}")
    return 0


def write_doc(df: pd.DataFrame, n_train: int, n_val: int, n_cloudy: int, mae_clim: float) -> None:
    bins = [f"mae_{lo}-{hi}" for lo, hi in C.HEIGHT_BINS]
    lines = ["# Cloud-base-height baselines (P058)", "",
             f"P047 pairs at a 10-minute cadence: {n_train:,} train / {n_val:,} val pairs, {n_cloudy:,} cloudy val pairs scored "
             f"on the lowest base in metres. Skill = 1 - MAE / MAE of the train-median climatology ({mae_clim:.0f} m). "
             "Test days locked.", "",
             "| Baseline | n | MAE (m) | Bias (m) | RMSD (m) | Skill | ≤250 m | ≤500 m | ≤1000 m | "
             + " | ".join(b.replace("mae_", "MAE ") for b in bins) + " | Cloud presence bal. acc |",
             "|---|" + "---|" * (9 + len(bins) + 1)]
    for _, r in df.iterrows():
        cells = " | ".join(f"{r[b]:.0f}" if b in r and pd.notna(r[b]) else "-" for b in bins)
        has_acc = "cloud_balanced_acc" in r and pd.notna(r["cloud_balanced_acc"])
        acc = f"{r['cloud_balanced_acc']:.3f}" if has_acc else "-"
        lines.append(f"| {r.baseline} | {int(r.n):,} | {r.mae_m:.0f} | {r.bias_m:+.0f} | {r.rmsd_m:.0f} | {r.skill:+.3f} | "
                     f"{r.within_250m:.2f} | {r.within_500m:.2f} | {r.within_1000m:.2f} | {cells} | {acc} |")
    lines += ["", "Zero-shot geometry foundation models (DA3, MoGe-2) are not run: they mask the sky by design and return no "
              "cloud "
              "geometry (research design §1). The ceilometer-supervised CNN is the reference of criterion C3."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
