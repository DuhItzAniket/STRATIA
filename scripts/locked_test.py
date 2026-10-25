"""P090 — the locked-test evaluation, run once.

    python scripts/locked_test.py --split val                 # develop / check the pipeline on validation rows
    python scripts/locked_test.py --split test --final --reason "..."   # the one run (logged)

The protocol is pre-registered in docs/phases/P090-locked-test.md: the Gate G4 models (in-domain global heads for
three seeds, the four LODO models, the dense heads, the CBH base and OLDLR-only heads), the two B0268 graphs, and
the P054 probe / P058 ridge references refitted on train, each on the lock entries they were pre-registered for,
with cluster-bootstrap 95 % CIs. Outputs: data/locked_test_<split>.parquet (long table), and for the test run
results/P090_locked_test.json (+ .sha256) and docs/data/locked_test.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stratia.baselines import cbh as C  # noqa: E402
from stratia.baselines import probes as P  # noqa: E402
from stratia.baselines import segmentation as S  # noqa: E402
from stratia.baselines import tasks as T  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.data.test_lock import load_lock, require_final_reason, verify_lock  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.model import cbh_bins as B  # noqa: E402
from stratia.model.dense import DenseHeads  # noqa: E402
from stratia.model.ood import OodScore, auroc  # noqa: E402
from stratia.train import data as D  # noqa: E402
from stratia.train import global_eval as G  # noqa: E402
from stratia.train.cbh_data import Model as CbhModel  # noqa: E402
from stratia.train.cbh_data import PairDataset, ray_maps_for  # noqa: E402
from stratia.train.cbh_data import predict as cbh_predict  # noqa: E402
from stratia.train.cbh_data import score as cbh_score  # noqa: E402
from stratia.train.dense_data import SIZE as DENSE_SIZE  # noqa: E402
from stratia.train.dense_data import (  # noqa: E402
    TokenMaskDataset,
    _layer_cms,
    bootstrap_miou,
    class_fractions,
    layer_scores,  # noqa: E402
)
from stratia.train.dense_data import evaluate as dense_evaluate  # noqa: E402
from stratia.utils.run import Run, _git  # noqa: E402

RUNS = ROOT / "runs"
MODELS = {
    "indomain_e2s_s0": "20261011T125754Z_global_indomain_e2s_s0_27d0d07c22",
    "indomain_e2s_s1": "20261011T124825Z_global_indomain_e2s_s1_s1_bb2f2f438a",
    "indomain_e2s_s2": "20261011T125312Z_global_indomain_e2s_s2_s2_9629d925c5",
    "lodo_ccsn_e2s": "20261011T130211Z_global_lodo_ccsn_e2s_s0_2fca96d16c",
    "lodo_mgcd_e2s": "20261011T130503Z_global_lodo_mgcd_e2s_s0_8428c46789",
    "lodo_montenegro_e2s": "20261011T130753Z_global_lodo_montenegro_e2s_s0_6eab87db84",
    "lodo_almeria_e2s": "20261011T130934Z_global_lodo_almeria_e2s_s0_e0d03a19d5",
    "dense_sky_layer_s0": "20261011T150404Z_dense_sky_layer_s0_dcd14f4833",
    "cbh_base_s0": "20261011T101553Z_cbh_head_s0_[P075_base]_e286609cd2",
    "cbh_oldlr_only_s0": "20261011T101430Z_cbh_head_s0_(OLDLR_only;_WESTE_held_out)_dcd5b03fe4",
}
GRAPHS = {"stratia-1.0.0-demo": ROOT / "models" / "stratia-1.0.0-demo",
          "stratia-1.0.1-demo": ROOT / "models" / "stratia-1.0.1-demo"}
LODO_FOLDS = ("ccsn", "mgcd", "montenegro", "almeria")
GENUS_SOURCES = ("ccsn", "mgcd")
PROBE = "probe_dinov3_vitb16_224"           # the P054 best LODO probe (C1, first clause)
PROBE_IND = "probe_dinov3_vitl16_224"       # the P054 best in-domain probe (C1, second clause), as at Gate G4
RIDGE = "ridge_dinov3_vits16_512"
CALIBRATION = ROOT / "configs" / "calibration.json"
RESULTS_JSON = ROOT / "results" / "P090_locked_test.json"
DOC = ROOT / "docs" / "data" / "locked_test.md"
SECTIONS = ("indomain", "ood", "lodo", "probe", "dense", "cbh", "b0268")


# ----------------------------------------------------------------------------------------------- helpers
class Records:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, protocol: str, subset: str, model: str, metric: str, value, ci=None, n: int | None = None,
            note: str = "") -> None:
        lo, hi = (ci if ci is not None else (float("nan"), float("nan")))
        self.rows.append({"protocol": protocol, "set": subset, "model": model, "metric": metric,
                          "value": float(value) if value is not None else float("nan"), "ci_lo": float(lo), "ci_hi": float(hi),
                          "n": int(n) if n is not None else -1, "note": note})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def cluster_bootstrap(units, fn, n_boot: int, seed: int = 0) -> dict[str, tuple[float, float]]:
    """fn(idx) -> dict of metrics; resample the clusters (units) with replacement; 2.5 / 97.5 percentiles per key."""
    units = np.asarray(units)
    uniq = np.unique(units)
    groups = {u: np.flatnonzero(units == u) for u in uniq}
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {}
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([groups[u] for u in pick])
        for k, v in fn(idx).items():
            if isinstance(v, (int, float)) and v == v:
                vals.setdefault(k, []).append(float(v))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items() if v}


def load_global(run_id: str, d_model: int, device: str) -> G.GlobalModel:
    cfg = OmegaConf.load(RUNS / run_id / "config.yaml")
    model = G.GlobalModel(d_model, int(cfg["hidden"]), float(cfg["dropout"]), bool(cfg["kendall"])).to(device)
    state = torch.load(RUNS / run_id / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if state.get("ema"):
        model.load_state_dict({**model.state_dict(), **state["ema"]["shadow"]})
    model.eval()
    return model


def global_dataset(rows: pd.DataFrame, data_root: Path, cache_root: Path, tag: str,
                   grid_key: str = "ffr1-patches-all-pool4", cls_key: str | None = None):
    grid = FeatureCache(cache_root, "dinov3_vits16", D.SIZE, transform=grid_key)
    cls = FeatureCache(cache_root, "dinov3_vits16", D.SIZE) if cls_key is None else \
        FeatureCache(cache_root, "dinov3_vits16", D.SIZE, transform=cls_key)
    have = set(grid.index().sample_id) & set(cls.index().sample_id)
    rows = rows[rows.sample_id.isin(have)].reset_index(drop=True)
    geo = D.GeometryCache.build(rows, data_root, cache_root / "geometry" / f"pool4_locked_{tag}_{len(rows)}.npz")
    ds = D.GlobalDataset(rows, grid.rows_for(rows.sample_id), cls.rows_for(rows.sample_id), grid.load("patches"),
                         cls.load("cls"), geo)
    return rows, ds


@torch.no_grad()
def global_logits(model: G.GlobalModel, ds: D.GlobalDataset, idx: np.ndarray, device: str) -> dict[str, np.ndarray]:
    dl = DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=256, shuffle=False, collate_fn=D.collate)
    gen, pooled = [], []
    for b in dl:
        out = model(b["tokens"].to(device), b["cls"].to(device), b["ray_map"].to(device), b["meta"].to(device))
        gen.append(out["genus_logits"].float().cpu().numpy())
        pooled.append(out["pooled"].float().cpu().numpy())
    return {"genus": np.concatenate(gen), "pooled": np.concatenate(pooled)}


def val_thresholds(probs: np.ndarray, truth: np.ndarray, grid=T.THRESHOLD_GRID) -> np.ndarray:
    """Per-column thresholds that maximise that column's F1 on the given (validation) rows; 0.5 where the column has
    no positive. `T.tuned_thresholds` returns predictions on its own rows, so it cannot carry thresholds to other rows."""
    thr = np.full(truth.shape[1], 0.5)
    for j in range(truth.shape[1]):
        if truth[:, j].sum() == 0:
            continue
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            p = probs[:, j] > t
            tp = (p & (truth[:, j] == 1)).sum()
            f1 = 2 * tp / max(2 * tp + (p & (truth[:, j] == 0)).sum() + (~p & (truth[:, j] == 1)).sum(), 1)
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        thr[j] = best_t
    return thr


def score_rows(rows: pd.DataFrame, pred: dict[str, np.ndarray], idx: np.ndarray) -> dict:
    part = rows.iloc[idx].reset_index(drop=True)
    return G.score_dataset(part, {k: v[idx] for k, v in pred.items()})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------- sections
def section_indomain(rec: Records, split: str, data_root: Path, cache_root: Path, device: str, n_boot: int,
                     with_ood: bool) -> None:
    rows_all = D.global_rows(eye2sky_etage=True, include_test=True)
    sel = rows_all[rows_all.split == split].reset_index(drop=True)
    rows, ds = global_dataset(sel, data_root, cache_root, f"indomain_{split}")
    d_model = int(ds.grid.shape[-1])
    idx_all = np.arange(len(rows))
    print(f"[indomain] {len(rows):,} {split} rows: "
          + ", ".join(f"{d} {n}" for d, n in rows.dataset.value_counts().items()), flush=True)
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for name in ("indomain_e2s_s0", "indomain_e2s_s1", "indomain_e2s_s2"):
        model = load_global(MODELS[name], d_model, device)
        pred = G.predict(model, ds, idx_all, device)
        for dataset, part_idx in rows.groupby("dataset").indices.items():
            part_idx = np.asarray(part_idx)
            sc = score_rows(rows, pred, part_idx)
            ci = {}
            if name.endswith("s0"):
                units = rows.unit.to_numpy()[part_idx]
                ci = cluster_bootstrap(units, lambda i, p=part_idx, pr=pred: score_rows(rows, pr, p[i]), n_boot)
            protocol = "cbh_pairs_etage" if dataset == "eye2sky_cbh" else "in_domain"
            for metric, value in sc.items():
                rec.add(protocol, dataset, name, metric, value, ci.get(metric), n=len(part_idx))
            per_seed.setdefault(dataset, {})[name] = sc
        if name.endswith("s0") and with_ood:
            section_ood(rec, split, rows, ds, model, data_root, cache_root, device)
        del model
    for dataset, seeds in per_seed.items():
        metrics = set().union(*[set(s) for s in seeds.values()])
        for metric in sorted(metrics):
            vals = [s[metric] for s in seeds.values() if metric in s and s[metric] == s[metric]]
            if len(vals) == 3:
                protocol = "cbh_pairs_etage" if dataset == "eye2sky_cbh" else "in_domain"
                rec.add(protocol, dataset, "indomain_e2s_mean3", metric, float(np.mean(vals)),
                        note=f"std over 3 seeds {float(np.std(vals)):.4f}")


def section_ood(rec: Records, split: str, rows: pd.DataFrame, ds: D.GlobalDataset, model: G.GlobalModel,
                data_root: Path, cache_root: Path, device: str) -> None:
    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    st = cal["ood_statistics"]
    ood = OodScore(len(st["mean"]))
    ood.load_state_dict({"mean": torch.tensor(st["mean"]), "precision": torch.tensor(st["precision"]),
                         "stats": torch.tensor(st["stats"]), "fitted": torch.tensor(True)})
    thr = float(cal["ood_threshold"])
    lg = global_logits(model, ds, np.arange(len(rows)), device)
    s_in = ood(torch.from_numpy(lg["pooled"]), torch.from_numpy(lg["genus"]))[:, 0].numpy()
    rec.add("ood", "in_domain_all", "indomain_e2s_s0", "flagged_share", float((s_in > thr).mean()), n=len(s_in))
    for src, part in rows.groupby("source").indices.items():
        rec.add("ood", src, "indomain_e2s_s0", "flagged_share", float((s_in[part] > thr).mean()), n=len(part))
    rows_b = D.b0268_rows(draw="k0_d0")
    rows_b = rows_b[rows_b.split == split].reset_index(drop=True)
    rows_b, ds_b = global_dataset(rows_b, data_root, cache_root, f"b0268_{split}", grid_key="ffr1-patches-b0268-pool4",
                                  cls_key="ffr1-b0268")
    lg_b = global_logits(model, ds_b, np.arange(len(rows_b)), device)
    s_out = ood(torch.from_numpy(lg_b["pooled"]), torch.from_numpy(lg_b["genus"]))[:, 0].numpy()
    rec.add("ood", "b0268", "indomain_e2s_s0", "flagged_share", float((s_out > thr).mean()), n=len(s_out))
    rec.add("ood", "b0268_vs_in_domain", "indomain_e2s_s0", "auroc", auroc(s_in, s_out), n=len(s_in) + len(s_out),
            note="higher score = out of domain; cached-feature path for both")
    print(f"[ood] in-domain flagged {(s_in > thr).mean():.3f}, B0268 {split} flagged {(s_out > thr).mean():.3f}, "
          f"AUROC {auroc(s_in, s_out):.3f}", flush=True)


def section_lodo(rec: Records, split: str, data_root: Path, cache_root: Path, device: str, n_boot: int) -> None:
    for fold in LODO_FOLDS:
        rows_f = D.global_rows(lodo_fold=fold, eye2sky_etage=True, include_test=True)
        held = rows_f[(rows_f.source == fold) & rows_f.split.isin([split, "val"])].reset_index(drop=True)
        rows, ds = global_dataset(held, data_root, cache_root, f"lodo_{fold}_{split}")
        model = load_global(MODELS[f"lodo_{fold}_e2s"], int(ds.grid.shape[-1]), device)
        pred = G.predict(model, ds, np.arange(len(rows)), device)
        ev = np.flatnonzero(rows.split.to_numpy() == split)
        va = np.flatnonzero(rows.split.to_numpy() == "val")
        sc = score_rows(rows, pred, ev)
        ci = cluster_bootstrap(rows.unit.to_numpy()[ev], lambda i, e=ev, r=rows, pr=pred: score_rows(r, pr, e[i]), n_boot)
        name = f"lodo_{fold}_e2s"
        for metric, value in sc.items():
            rec.add("lodo", fold, name, metric, value, ci.get(metric), n=len(ev))
        # secondary: thresholds chosen on the fold's val rows, applied to the evaluated rows
        full_v = (np.stack(rows.etage_w.to_numpy()[va]) >= 1.0).all(1)
        full_e = (np.stack(rows.etage_w.to_numpy()[ev]) >= 1.0).all(1)
        if full_v.any() and full_e.any():
            truth_v = (np.stack(rows.etage.to_numpy()[va])[full_v] > 0.5).astype(int)
            thr = val_thresholds(pred["etage"][va][full_v], truth_v)
            truth_e = (np.stack(rows.etage.to_numpy()[ev])[full_e] > 0.5).astype(int)
            hat = (pred["etage"][ev][full_e] > thr).astype(int)
            rec.add("lodo", fold, name, "etage_f1_valtuned", T.multilabel_macro_f1(truth_e, hat), n=int(full_e.sum()),
                    note="thresholds chosen on the fold's val rows" + (" (= the evaluated rows)" if split == "val" else ""))
        print(f"[lodo] {fold}: étage F1 {sc.get('etage_f1', float('nan')):.3f} on {len(ev)} {split} rows", flush=True)
        del model


def _fit_predict_multilabel(xt: np.ndarray, yt: np.ndarray, xe: np.ndarray, c: float, seed: int = 0) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    mu, sd = xt.mean(0, keepdims=True), xt.std(0, keepdims=True) + 1e-6
    xt, xe = (xt - mu) / sd, (xe - mu) / sd
    pred = np.zeros((len(xe), yt.shape[1]), dtype=yt.dtype)
    for j in range(yt.shape[1]):
        if yt[:, j].min() == yt[:, j].max():
            pred[:, j] = yt[0, j]
            continue
        pred[:, j] = LogisticRegression(C=c, max_iter=2000, random_state=seed).fit(xt, yt[:, j]).predict(xe)
    return pred


def _fit_predict(xt: np.ndarray, yt: np.ndarray, xe: np.ndarray, c: float, seed: int = 0) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    mu, sd = xt.mean(0, keepdims=True), xt.std(0, keepdims=True) + 1e-6
    return LogisticRegression(C=c, max_iter=2000, random_state=seed).fit((xt - mu) / sd, yt).predict((xe - mu) / sd)


def section_probe(rec: Records, split: str, cache_root: Path, n_boot: int) -> None:
    """The P054 best probes (linear on CLS features; C chosen on val, as in P054) refitted on train and scored on the
    evaluated rows: DINOv3 ViT-B/16 @ 224 for the LODO étage clause of C1, ViT-L/16 @ 224 for its in-domain genus
    clause, the two references Gate G4 compared against."""
    tables = T.load_tables()
    cache = FeatureCache(cache_root, "dinov3_vitb16", 224)
    have = set(cache.index().sample_id)
    unit_of = tables.in_domain.set_index("sample_id").unit
    for fold in LODO_FOLDS:
        tab = T.lodo_table(tables, fold, include_test=True)
        tab = tab[tab.sample_id.isin(have)]
        tr, va, ev = tab[tab.role == "train"], tab[tab.role == "val"], tab[tab.role == split]
        xt, xv, xe = (cache.features(t.sample_id, "cls") for t in (tr, va, ev))
        yt, yv, ye = (T.etage_matrix(t.etage_set) for t in (tr, va, ev))
        res = P.linear_probe(xt, yt, xv, yv, multilabel=True)
        pred = res.val_pred if split == "val" else _fit_predict_multilabel(xt, yt, xe, res.params["C"])
        units = unit_of.reindex(ev.sample_id).fillna("?").to_numpy()
        ci = cluster_bootstrap(units, lambda i, y=ye, p=pred: {"etage_f1": T.multilabel_macro_f1(y[i], p[i])}, n_boot)
        rec.add("lodo", fold, PROBE, "etage_f1", T.multilabel_macro_f1(ye, pred), ci.get("etage_f1"), n=len(ev),
                note=f"C {res.params['C']} chosen on val")
        rec.add("lodo", fold, PROBE, "etage_unsupported_fp", T.unsupported_positive_rate(ye, pred), n=len(ev))
        print(f"[probe] lodo {fold}: étage F1 {T.multilabel_macro_f1(ye, pred):.3f}", flush=True)
    cache = FeatureCache(cache_root, "dinov3_vitl16", 224)
    have = set(cache.index().sample_id)
    for source in GENUS_SOURCES:
        tab = T.recognition_table(tables, source, include_test=True)
        tab = tab[tab.sample_id.isin(have)]
        tr, va, ev = tab[tab.split == "train"], tab[tab.split == "val"], tab[tab.split == split]
        xt, xv, xe = (cache.features(t.sample_id, "cls") for t in (tr, va, ev))
        res = P.linear_probe(xt, tr.label.to_numpy(), xv, va.label.to_numpy())
        pred = res.val_pred if split == "val" else _fit_predict(xt, tr.label.to_numpy(), xe, res.params["C"])
        ye = ev.label.to_numpy()
        units = unit_of.reindex(ev.sample_id).fillna("?").to_numpy()
        ci = cluster_bootstrap(units, lambda i, y=ye, p=pred: {"genus_f1": T.macro_f1(y[i], p[i])}, n_boot)
        rec.add("in_domain", source, PROBE_IND, "genus_f1", T.macro_f1(ye, pred), ci.get("genus_f1"), n=len(ev),
                note=f"C {res.params['C']} chosen on val")
        rec.add("in_domain", source, PROBE_IND, "genus_acc", T.accuracy(ye, pred), n=len(ev))
        print(f"[probe] in-domain {source}: genus F1 {T.macro_f1(ye, pred):.3f}", flush=True)


def section_dense(rec: Records, split: str, data_root: Path, cache_root: Path, device: str, n_boot: int) -> None:
    idx = pd.read_parquet("data/segmentation_index.parquet").rename(columns={"split": "native_split"})
    idx["sample_id"] = idx.dataset + ":" + idx.image_file
    sp = pd.read_parquet("data/splits/in_domain.parquet", columns=["sample_id", "split"])
    tab = idx.merge(sp, on="sample_id", how="left")
    tab = tab[tab.split == split].reset_index(drop=True)
    sky_masks, layer_masks = S.load_both_masks(tab, data_root)
    cache = FeatureCache(cache_root, "dinov3_vits16", 512, transform="ffr1-patches-seg")
    patches = cache.load("patches")
    ds = TokenMaskDataset(cache.rows_for(tab.sample_id), patches, sky_masks, layer_masks, False)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    model = DenseHeads(d_model=patches.shape[-1]).to(device)
    state = torch.load(RUNS / MODELS["dense_sky_layer_s0"] / "checkpoints" / "best.pt", map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    datasets = tab.dataset.tolist()
    ev = dense_evaluate(model, dl, datasets, device, True)
    name = "dense_sky_layer_s0"
    print(f"[dense] {len(tab):,} {split} images: {tab.dataset.value_counts().to_dict()}", flush=True)
    lcms = _layer_cms(model, dl, datasets, device)
    for d in ["all", *sorted(set(datasets))]:
        cm = sum(ev["sky"].values()) if d == "all" else ev["sky"][d]
        sc = S.scores_from_confusion(cm)
        lo, hi = bootstrap_miou(ev["per_image"], None if d == "all" else d, n_boot,
                                scorer=lambda c: S.scores_from_confusion(c).miou)
        n = len(datasets) if d == "all" else datasets.count(d)
        rec.add("dense", d, name, "sky_miou", sc.miou, (lo, hi), n=n)
        for cls, v in sc.iou.items():
            rec.add("dense", d, name, f"iou_{cls}", v, n=n)
        rec.add("dense", d, name, "predicted_cloud_fraction", sc.cloud_fraction, n=n)
        lcm = sum(ev["layer"].values()) if d == "all" else ev["layer"][d]
        ls = layer_scores(lcm)
        if ls:
            per_l = [(dd, c) for dd, c in zip(datasets, lcms, strict=True)]
            llo, lhi = bootstrap_miou(per_l, None if d == "all" else d, n_boot,
                                      scorer=lambda c: float(np.mean(list(layer_scores(c).values()) or [np.nan])))
            rec.add("dense", d, name, "layer_miou", float(np.mean(list(ls.values()))), (llo, lhi), n=n)
            for cls, v in ls.items():
                rec.add("dense", d, name, f"layer_iou_{cls}", v, n=n)
        layer_text = f", layer mIoU {np.mean(list(ls.values())):.3f}" if ls else ""
        print(f"[dense] {d}: sky mIoU {sc.miou:.3f}{layer_text}", flush=True)
    # collapse check on the clear Eye2Sky frames of this split (the research design's < 0.20)
    cb = pd.read_parquet("data/cbh_pairs.parquet")
    clear = cb[cb.paired & cb.daytime & (cb.label == "none") & (cb.confidence >= 0.9) & (cb.utc.dt.minute % 10 == 0)
               & (cb.split == split)]
    clear = clear.sample(min(200, len(clear)), random_state=0) if len(clear) else clear
    if len(clear):
        masks = S.camera_masks(clear, data_root, DENSE_SIZE)
        frac = class_fractions(model, clear.image_file.tolist(), masks, cache_root, device)
        for cls, v in frac.items():
            rec.add("dense", f"clear_sky_{split}", name, f"fraction_{cls}", v, n=len(clear))
        print(f"[dense] clear-sky cloud fraction {frac.get('cloud', float('nan')):.3f} on {len(clear)} frames", flush=True)
    del model


def _quantile_height(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(p, 1)
    idx = np.clip((cdf >= q[:, None]).argmax(1), 0, B.K_BINS - 1)
    prev = np.where(idx > 0, cdf[np.arange(len(p)), np.clip(idx - 1, 0, None)], 0.0)
    frac = np.clip((q - prev) / np.clip(cdf[np.arange(len(p)), idx] - prev, 1e-12, None), 0, 1)
    lo_e, hi_e = np.log(np.clip(B.EDGES_M[idx], 1.0, None)), np.log(B.EDGES_M[idx + 1])
    return np.exp(lo_e + frac * (hi_e - lo_e))


def calibrated_coverage(probs: np.ndarray, heights: np.ndarray, temperature: float, radius: float) -> float:
    cloudy = ~np.isnan(heights)
    logits = np.log(np.clip(probs[cloudy, :B.K_BINS], 1e-9, None)) / temperature
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    r = min(radius, 0.5)
    lo = _quantile_height(p, np.full(len(p), 0.5 - r))
    hi = _quantile_height(p, np.full(len(p), 0.5 + r))
    return float(np.mean((heights[cloudy] >= lo) & (heights[cloudy] <= hi)))


def _cbh_pairs(grid: FeatureCache) -> pd.DataFrame:
    d = pd.read_parquet("data/cbh_pairs.parquet")
    d = d[d.paired & d.daytime & d.zenith_inside_mask & (d.utc.dt.minute % 10 == 0)].copy()
    d["sample_id"] = "eye2sky:" + d.image_file
    d = d[d.sample_id.isin(set(grid.index().sample_id))].reset_index(drop=True)
    d["cloud"] = d.label != "none"
    d["height_m"] = np.where(d.cloud, d.cbh_m, np.nan)
    return d


def _cbh_rows(rec: Records, name: str, subset: str, probs: np.ndarray, part: pd.DataFrame, mae_clim: float,
              n_boot: int, cal: dict, note: str = "") -> None:
    heights = part.height_m.to_numpy().astype(np.float32)
    cloud = part.cloud.to_numpy()
    sc = cbh_score(probs, heights, cloud, mae_clim)

    def boot(i):
        s = cbh_score(probs[i], heights[i], cloud[i], mae_clim)
        return {k: s[k] for k in ("mae_m", "skill", "within_500m", "coverage_80", "crps")}

    ci = cluster_bootstrap(part.day.astype(str).to_numpy(), boot, n_boot)
    for metric, value in sc.items():
        if isinstance(value, (int, float)):
            rec.add("cbh_pairs", subset, name, metric, value, ci.get(metric), n=len(part), note=note)
    t, r = float(cal["temperatures"]["cbh"]), cal["cbh_conformal_radius"]
    rec.add("cbh_pairs", subset, name, "coverage_80_calibrated", calibrated_coverage(probs, heights, t, float(r["80"])),
            n=int(sc["n_cloudy"]), note=f"P089 temperature {t} and radius {r['80']}")
    rec.add("cbh_pairs", subset, name, "coverage_95_calibrated", calibrated_coverage(probs, heights, t, float(r["95"])),
            n=int(sc["n_cloudy"]), note=f"P089 temperature {t} and radius {r['95']}")
    print(f"[cbh] {name} on {subset}: MAE {sc['mae_m']:.0f} m, skill {sc['skill']:+.3f}, n cloudy {sc['n_cloudy']}",
          flush=True)


def section_cbh(rec: Records, split: str, data_root: Path, cache_root: Path, device: str, n_boot: int) -> None:
    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    grid = FeatureCache(cache_root, "dinov3_vits16", 512, transform="ffr1-patches-cbh-pool4")
    pooled = FeatureCache(cache_root, "dinov3_vits16", 512)
    d = _cbh_pairs(grid)
    train, val, ev = d[d.split == "train"], d[d.split == "val"], d[d.split == split].reset_index(drop=True)
    rms, metas = ray_maps_for(ev, data_root)
    g_arr, c_arr = grid.load("patches"), pooled.load("cls")
    heights = ev.height_m.to_numpy().astype(np.float32)
    cloudy = ~np.isnan(heights)
    print(f"[cbh] {len(ev):,} {split} pairs ({ev.station.value_counts().to_dict()}), cloudy {int(cloudy.sum()):,}; "
          f"ray maps valid {float((rms[:, 3].reshape(len(ev), -1).max(1) > 0).mean()):.1%}", flush=True)

    def loader(part_idx):
        p = ev.iloc[part_idx]
        ds = PairDataset(grid.rows_for(p.sample_id), pooled.rows_for(p.sample_id), g_arr, c_arr, rms[part_idx],
                         metas[part_idx], heights[part_idx])
        return DataLoader(ds, batch_size=256, shuffle=False)

    def load_cbh(key):
        m = CbhModel(d_model=int(g_arr.shape[-1])).to(device)
        state = torch.load(RUNS / MODELS[key] / "checkpoints" / "best.pt", map_location=device, weights_only=False)
        m.load_state_dict(state["model"] if "model" in state else state)
        m.eval()
        return m

    all_idx = np.arange(len(ev))
    clim = C.climatology(train)
    mae_clim = C.score_heights(np.full(int(cloudy.sum()), clim), heights[cloudy]).mae
    rec.add("cbh_pairs", "all", "climatology_train", "mae_m", mae_clim, n=int(cloudy.sum()), note=f"clim {clim:.0f} m")
    base = load_cbh("cbh_base_s0")
    dl = loader(all_idx)
    for null, name in ((False, "cbh_base_s0"), (True, "cbh_base_s0_geometry_zeroed")):
        probs = cbh_predict(base, dl, device, null_geometry=null)
        _cbh_rows(rec, name, "all", probs, ev, mae_clim, n_boot, cal)
        for station, part_idx in ev.groupby("station").indices.items():
            part_idx = np.asarray(part_idx)
            part = ev.iloc[part_idx].reset_index(drop=True)
            # skill against the climatology MAE of the whole split, as the trainer and Gate G4 computed it
            _cbh_rows(rec, name, station, probs[part_idx], part, mae_clim, n_boot, cal,
                      note="skill against the training climatology's MAE on the whole split (Gate G4 definition)")
            sub_cloudy = cloudy[part_idx]
            mae_clim_st = C.score_heights(np.full(int(sub_cloudy.sum()), clim), heights[part_idx][sub_cloudy]).mae
            sk = cbh_score(probs[part_idx], heights[part_idx], part.cloud.to_numpy(), mae_clim_st)["skill"]
            rec.add("cbh_pairs", station, name, "skill_station_climatology", sk, n=len(part),
                    note=f"secondary: against the climatology's MAE on this station's rows ({mae_clim_st:.0f} m)")
    del base
    # the held-out site: OLDLR-only model on WESTE, climatology from OLDLR train pairs
    held = load_cbh("cbh_oldlr_only_s0")
    w_idx = np.flatnonzero(ev.station.to_numpy() == "WESTE")
    if len(w_idx):
        clim_o = C.climatology(train[train.station == "OLDLR"])
        mae_clim_o = C.score_heights(np.full(int(cloudy.sum()), clim_o), heights[cloudy]).mae    # whole split, as trained
        probs = cbh_predict(held, loader(w_idx), device)
        part = ev.iloc[w_idx].reset_index(drop=True)
        _cbh_rows(rec, "cbh_oldlr_only_s0", "WESTE", probs, part, mae_clim_o, n_boot, cal,
                  note=f"site never seen in training; OLDLR training climatology ({clim_o:.0f} m), its MAE on the whole "
                       "split (Gate G4 definition)")
        w_cloudy = cloudy[w_idx]
        mae_clim_w = C.score_heights(np.full(int(w_cloudy.sum()), clim_o), heights[w_idx][w_cloudy]).mae
        sk = cbh_score(probs, heights[w_idx], part.cloud.to_numpy(), mae_clim_w)["skill"]
        rec.add("cbh_pairs", "WESTE", "cbh_oldlr_only_s0", "skill_station_climatology", sk, n=len(part),
                note=f"secondary: against the OLDLR climatology's MAE on the WESTE rows ({mae_clim_w:.0f} m)")
    del held
    # the P058 ridge reference, refitted on train (alpha chosen on val), scored on the evaluated cloudy pairs
    x_tr, x_va, x_ev = (pooled.features(p.sample_id, "cls") for p in (train, val, ev))
    tr_c, va_c = train.cbh_m.notna().to_numpy(), val.cbh_m.notna().to_numpy()
    model, mu, sd, alpha = C.fit_ridge_log_height(x_tr[tr_c], train.cbh_m.to_numpy()[tr_c], x_val=x_va[va_c],
                                                  cbh_val=val.cbh_m.to_numpy()[va_c])
    pred = C.predict_ridge(model, mu, sd, x_ev[cloudy])
    sc = C.score_heights(pred, heights[cloudy], mae_clim).row()
    days = ev.day.astype(str).to_numpy()[cloudy]
    true_c = heights[cloudy]
    ci = cluster_bootstrap(days, lambda i, p=pred, t=true_c: {"mae_m": float(np.abs(p[i] - t[i]).mean())}, n_boot)
    for metric, value in sc.items():
        rec.add("cbh_pairs", "all", RIDGE, metric, value, ci.get(metric), n=int(cloudy.sum()),
                note=f"alpha {alpha} chosen on val")
    for station in ("OLDLR", "WESTE"):
        m = (ev.station.to_numpy()[cloudy] == station)
        if m.any():
            mae_clim_st = C.score_heights(np.full(int(m.sum()), clim), true_c[m]).mae
            for metric, value in C.score_heights(pred[m], true_c[m], mae_clim_st).row().items():
                rec.add("cbh_pairs", station, RIDGE, metric, value, n=int(m.sum()))
    print(f"[cbh] ridge: MAE {sc['mae_m']:.0f} m, skill {sc['skill']:+.3f}", flush=True)


def section_b0268(rec: Records, split: str, n_boot: int, phase: str) -> None:
    py = sys.executable
    subset = "locked_day" if split == "test" else "adaptation_days"
    for graph, folder in GRAPHS.items():
        suffix = f"locked_{split}_{graph.replace('.', '_').replace('-', '_')}"
        cmd = [py, str(ROOT / "scripts" / "eval_b0268.py"), "--model", str(folder), "--draw", "k0_d0", "--roles", split,
               "--suffix", suffix, "--phase", phase, "--name", "b0268_locked"]
        print("[b0268] " + " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        scores = pd.read_parquet(ROOT / "data" / f"b0268_zero_shot_scores_{suffix}.parquet")
        # the 'all' rows come in table order (each rater averaged first, then the majority / cover / OOD table): the
        # first non-missing value of a metric is the headline one
        row = scores[scores.day == "all"].apply(lambda col: col.dropna().iloc[0] if col.notna().any() else np.nan)
        for metric in ("n", "etage_set_agreement", "c2_etage_ratio", "oktas_within1", "c2_oktas_ratio", "oktas_mae",
                       "oktas_bias", "genus_top1_in_set", "etage_f1", "cloud_fraction_corr_vs_cover",
                       "cloud_fraction_mae_vs_cover", "cloud_fraction_on_le1_okta", "glare_fraction_strong_frames",
                       "ood_flagged_share", "ood_mean"):
            if metric in row and row[metric] == row[metric]:
                rec.add("few_shot_target", subset, graph, metric, float(row[metric]), n=int(row["n"]))
        # per-frame bootstrap against the pair's majority (the per-rater numbers above are the headline)
        frames = pd.read_parquet(ROOT / "data" / f"b0268_zero_shot_{suffix}.parquet")
        etage_hat = (np.array([json.loads(x) for x in frames.etage_probs]) > 0.5).astype(int)      # JSON per frame
        etage_true = np.array([json.loads(x) for x in frames.etage_majority]).astype(int)
        okt_ok = (np.abs(frames.oktas_argmax.to_numpy() - frames.oktas_majority.to_numpy()) <= 1)
        exact = (etage_hat == etage_true).all(1)

        def boot(i, ex=exact, ok=okt_ok):
            return {"etage_set_agreement_majority": float(ex[i].mean()), "oktas_within1_majority": float(ok[i].mean())}

        ci = cluster_bootstrap(np.arange(len(frames)), boot, n_boot)
        rec.add("few_shot_target", subset, graph, "etage_set_agreement_majority", float(exact.mean()),
                ci.get("etage_set_agreement_majority"), n=len(frames), note="frame bootstrap, one day")
        rec.add("few_shot_target", subset, graph, "oktas_within1_majority", float(okt_ok.mean()),
                ci.get("oktas_within1_majority"), n=len(frames), note="frame bootstrap, one day")
        print(f"[b0268] {graph}: étage set {row['etage_set_agreement']:.3f}, oktas ±1 {row['oktas_within1']:.3f}, "
              f"genus top-1 {row['genus_top1_in_set']:.3f}, OOD flagged {row['ood_flagged_share']:.3f}", flush=True)


# ----------------------------------------------------------------------------------------------- report
def headline(df: pd.DataFrame) -> list[dict]:
    """The Gate G4 criteria on the evaluated rows (validation value from P082 next to it)."""
    g4 = pd.read_parquet("data/gate_g4.parquet") if Path("data/gate_g4.parquet").exists() else pd.DataFrame()

    def val_of(prefix):
        if not len(g4):
            return float("nan")
        m = g4[g4.quantity.str.startswith(prefix)]
        return float(m.model.iloc[0]) if len(m) else float("nan")

    def pick(protocol, subset, model, metric):
        m = df[(df.protocol == protocol) & (df.set == subset) & (df.model == model) & (df.metric == metric)]
        if not len(m):
            return float("nan"), ""
        r = m.iloc[0]
        if np.isnan(r.ci_lo):
            ci = ""
        else:
            ci = f"[{r.ci_lo:.0f}, {r.ci_hi:.0f}]" if abs(r.value) >= 100 else f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
        return float(r.value), ci

    def crit(criterion, quantity, val_prefix, model, reference, reference_name, required, ok, ci=""):
        return {"criterion": criterion, "quantity": quantity, "val_p082": val_of(val_prefix), "model": model,
                "reference": reference, "reference_name": reference_name, "required": required, "pass": bool(ok), "ci": ci}

    lodo_model = float(np.mean([pick("lodo", f, f"lodo_{f}_e2s", "etage_f1")[0] for f in LODO_FOLDS]))
    lodo_probe = float(np.mean([pick("lodo", f, PROBE, "etage_f1")[0] for f in LODO_FOLDS]))
    ind_model = float(np.mean([pick("in_domain", s, "indomain_e2s_s0", "genus_f1")[0] for s in GENUS_SOURCES]))
    ind_probe = float(np.mean([pick("in_domain", s, PROBE_IND, "genus_f1")[0] for s in GENUS_SOURCES]))
    c2e, c2e_ci = pick("in_domain", "montenegro", "indomain_e2s_s0", "c2_etage_ratio")
    c2o, c2o_ci = pick("in_domain", "montenegro", "indomain_e2s_s0", "c2_oktas_ratio")
    sk_o, sk_o_ci = pick("cbh_pairs", "OLDLR", "cbh_base_s0", "skill")
    sk_w, sk_w_ci = pick("cbh_pairs", "WESTE", "cbh_oldlr_only_s0", "skill")
    mae, mae_ci = pick("cbh_pairs", "all", "cbh_base_s0", "mae_m")
    mae_ref, _ = pick("cbh_pairs", "all", RIDGE, "mae_m")
    cov, _ = pick("cbh_pairs", "all", "cbh_base_s0", "coverage_80_calibrated")
    probe_name = "probe ViT-B/16 @ 224, refitted"
    probe_ind_name = "probe ViT-L/16 @ 224, refitted"
    ridge_name = "ridge DINOv3 ViT-S/16 @ 512, refitted (the P058 CNN kept no checkpoint; 724 m on val)"
    return [
        crit("C1", "LODO étage macro-F1, mean of the four folds", "LODO étage macro-F1 (mean", lodo_model, lodo_probe,
             probe_name, "≥ reference + 0.05", lodo_model - lodo_probe >= 0.05),
        crit("C1", "in-domain genus macro-F1 (CCSN, MGCD)", "in-domain genus", ind_model, ind_probe, probe_ind_name,
             "≥ reference − 0.01", ind_model >= ind_probe - 0.01),
        crit("C2", "Montenegro étage-set agreement / human ceiling", "Montenegro étage-set", c2e, 0.90,
             "pre-registered bar", "≥ 0.90", c2e >= 0.90, c2e_ci),
        crit("C2", "Montenegro oktas ±1 agreement / human ceiling", "Montenegro oktas", c2o, 0.90,
             "pre-registered bar", "≥ 0.90", c2o >= 0.90, c2o_ci),
        crit("C3", "CBH skill, OLDLR (CDLRA) pairs", "CBH skill, CDLRA", sk_o, 0.20, "pre-registered bar", "≥ 0.20",
             sk_o >= 0.20, sk_o_ci),
        crit("C3", "CBH skill, WESTE (CDLRB) pairs, OLDLR-only training", "CBH skill, held-out", sk_w, 0.10,
             "pre-registered bar", "≥ 0.10", sk_w >= 0.10, sk_w_ci),
        crit("C3", "CBH MAE (m), all pairs, against the frozen-feature ridge", "CBH MAE", mae, mae_ref, ridge_name,
             "≤ 0.9 × reference", mae <= 0.9 * mae_ref, mae_ci),
        crit("C3", "80 % interval coverage with the P089 temperature and radius", "80 % interval", cov, 0.80, "nominal",
             "0.75–0.85", abs(cov - 0.80) <= 0.05),
    ]


def _f(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"


def write_doc(df: pd.DataFrame, head: list[dict], meta: dict, split: str) -> None:
    title = "the one run" if split == "test" else "validation check"
    lines = [f"# Locked-test evaluation (P090, {title})", "",
             f"Split `{split}`; lock `{meta['lock_prefix']}`; run `{meta['run_id']}`; git `{meta['git_sha']}`; "
             f"{meta['n_boot']} cluster-bootstrap resamples; models and protocol as pre-registered in "
             "`docs/phases/P090-locked-test.md`. Thresholds 0.5; one seed except the in-domain global heads (three).", "",
             "## The Gate G4 criteria on these rows", "",
             "| Criterion | Quantity | P082 (val) | This split | 95 % CI | Reference | Required | Holds |",
             "|---|---|---|---|---|---|---|---|"]
    for r in head:
        lines.append(f"| {r['criterion']} | {r['quantity']} | {_f(r['val_p082'])} | {_f(r['model'])} | {r['ci']} | "
                     f"{_f(r['reference'])} ({r['reference_name']}) | {r['required']} | {'yes' if r['pass'] else 'no'} |")
    lines += ["", "## Every number", ""]
    for protocol, part in df.groupby("protocol", sort=False):
        lines += [f"### `{protocol}`", "", "| Set | Model | Metric | Value | 95 % CI | n | Note |",
                  "|---|---|---|---|---|---|---|"]
        for r in part.itertuples(index=False):
            ci = "" if np.isnan(r.ci_lo) else f"[{_f(r.ci_lo)}, {_f(r.ci_hi)}]"
            n = r.n if r.n >= 0 else ""
            lines.append(f"| {r.set} | `{r.model}` | {r.metric} | {_f(r.value)} | {ci} | {n} | {r.note} |")
        lines.append("")
    lines += ["Not evaluated (no model exists): `lodo/swim/test`, `held_out_station/same_days/test`, "
              "`held_out_station/disjoint_days/test`.", ""]
    DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    t0 = time.perf_counter()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="the one locked-test run (needs --split test and --reason)")
    ap.add_argument("--reason", default="")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--only", nargs="+", default=list(SECTIONS), choices=SECTIONS, help="sections to run (development)")
    ap.add_argument("--phase", default="P090")
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        print("the locked test rows are read only with --final and a --reason (research design §6)", file=sys.stderr)
        return 2
    if a.final and a.split != "test":
        print("--final is for --split test", file=sys.stderr)
        return 2
    lock = load_lock()
    problems = verify_lock(lock, ROOT / "data" / "splits")
    if problems:
        print("test lock broken: " + "; ".join(problems), file=sys.stderr)
        return 1
    phase = a.phase if a.split == "test" else f"{a.phase}-dev"
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec = Records()
    out = ROOT / "data" / f"locked_test_{a.split}.parquet"

    def save() -> None:                         # after every section: a crash keeps the numbers already computed
        rec.frame().to_parquet(out, index=False)

    cfg = OmegaConf.create({"split": a.split, "final": a.final, "bootstrap": a.bootstrap, "sections": list(a.only),
                            "models": MODELS, "graphs": {k: str(v) for k, v in GRAPHS.items()},
                            "lock_prefix": lock["manifest_sample_id_sha256_prefix"]})
    with Run(cfg, name=f"locked_test_{a.split}", phase=phase) as run:
        if a.final:
            log = require_final_reason(a.reason, lock, run=run.run_id)
            print(f"final evaluation logged in {log}", flush=True)
        print(f"locked-test evaluation on `{a.split}` rows ({device}); sections {a.only}", flush=True)
        if "indomain" in a.only:
            section_indomain(rec, a.split, data_root, cache_root, device, a.bootstrap, with_ood="ood" in a.only)
            save()
        if "lodo" in a.only:
            section_lodo(rec, a.split, data_root, cache_root, device, a.bootstrap)
            save()
        if "probe" in a.only:
            section_probe(rec, a.split, cache_root, a.bootstrap)
            save()
        if "dense" in a.only:
            section_dense(rec, a.split, data_root, cache_root, device, a.bootstrap)
            save()
        if "cbh" in a.only:
            section_cbh(rec, a.split, data_root, cache_root, device, a.bootstrap)
            save()
        if "b0268" in a.only:
            section_b0268(rec, a.split, a.bootstrap, phase)
            save()
        df = rec.frame()
        save()
        head = headline(df) if set(a.only) == set(SECTIONS) else []
        meta = {"run_id": run.run_id, "git_sha": _git("rev-parse", "--short", "HEAD"),
                "lock_prefix": lock["manifest_sample_id_sha256_prefix"], "n_boot": a.bootstrap, "split": a.split,
                "started_utc": run.started.isoformat()}
        if a.split == "test":
            RESULTS_JSON.parent.mkdir(exist_ok=True)
            payload = {"meta": meta, "reason": a.reason, "models": MODELS,
                       "graphs": {k: str(v.relative_to(ROOT)) for k, v in GRAPHS.items()},
                       "headline": head, "rows": df.to_dict(orient="records")}
            RESULTS_JSON.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
            digest = sha256_file(RESULTS_JSON)
            RESULTS_JSON.with_suffix(".sha256").write_text(f"{digest}  {RESULTS_JSON.name}\n", encoding="utf-8")
            print(f"results {RESULTS_JSON} sha256 {digest}", flush=True)
        if head:
            write_doc(df, head, meta, a.split)
            for r in head:
                verdict = "holds" if r["pass"] else "no"
                print(f"  {r['criterion']} {r['quantity']}: {_f(r['model'])} vs {_f(r['reference'])} -> {verdict}",
                      flush=True)
        run.finish(n_rows=float(len(df)), **{f"c_{i}": float(r["model"]) for i, r in enumerate(head)})
    print(f"done in {time.perf_counter() - t0:.0f} s -> {out}" + (f", {DOC}" if head else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
