"""Fit sign-off report for the Stage H candidates (P081).

    python scripts/fit_report.py --global-run <run dir of the P073 indomain_e2s run> --cbh-run <run dir of the P075 base run>
                                 [--dense-run <P074 run dir>] [--joint-run <P076 run dir>] [--lora-run <P077 run dir>]

For every run: the train / val curves of its headline metric from the run folder (`curves.json` or `metrics.jsonl`),
the train–val gap at the best epoch and the PLAN §4 verdict (overfit: gap > 15 points and val falling after the
best epoch; overfit-leaning: gap only; underfit: train below 0.6; else ok). Calibration: expected calibration error
(15 bins) of the genus and étage probabilities (every (sample, class) Bernoulli), of the oktas top-class confidence
and of the CBH distribution's 80 % interval, on the validation rows; for CBH a temperature is cross-fitted over the
validation days (fit on half the days, apply to the other half, both ways) to show the coverage a calibration step
(P089) will reach without touching test. Output: docs/data/fit_report.md, data/fit_report.parquet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.registry import load_paths  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.model import cbh_bins as B  # noqa: E402
from stratia.train import data as D  # noqa: E402
from stratia.train.global_eval import GlobalModel, predict  # noqa: E402

DOC = Path("docs/data/fit_report.md")
OUT = Path("data/fit_report.parquet")


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def verdict(train: float, val: float, curve_val: list[float], best_i: int) -> str:
    gap = train - val
    fell = len(curve_val) > best_i + 1 and curve_val[-1] <= curve_val[best_i] - 0.01
    if gap > 0.15 and fell:
        return f"overfit (gap {gap:.2f}, val fell after epoch {best_i + 1})"
    if gap > 0.15:
        return f"overfit-leaning (gap {gap:.2f})"
    if train < 0.6:
        return f"underfit (train {train:.2f})"
    return f"ok (gap {gap:.2f})"


def global_curves(run_dir: Path) -> dict:
    curves = json.loads((run_dir / "curves.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    best_epoch = int(summary.get("best_epoch", len(curves)))
    rows = []
    datasets = sorted({k.split("/")[1] for c in curves for k in c
                       if k.startswith("val/") and k.count("/") == 2 and k.endswith("etage_f1")})
    for ds in datasets:
        val = [c.get(f"val/{ds}/etage_f1", np.nan) for c in curves]
        tr = [c.get(f"train/{ds}/etage_f1", np.nan) for c in curves]
        i = best_epoch - 1
        rows.append({"run": run_dir.name, "head": "etage", "dataset": ds, "train": tr[i], "val": val[i], "gap": tr[i] - val[i],
                     "best_epoch": best_epoch, "verdict": verdict(tr[i], val[i], val, i)})
    genus_sets = sorted({k.split("/")[1] for c in curves for k in c
                         if k.startswith("val/") and k.endswith("genus_f1") and k.count("/") == 2})
    for ds in genus_sets:
        val = [c.get(f"val/{ds}/genus_f1", np.nan) for c in curves]
        tr = [c.get(f"train/{ds}/genus_f1", np.nan) for c in curves]
        i = best_epoch - 1
        rows.append({"run": run_dir.name, "head": "genus", "dataset": ds, "train": tr[i], "val": val[i], "gap": tr[i] - val[i],
                     "best_epoch": best_epoch, "verdict": verdict(tr[i], val[i], val, i)})
    return {"rows": rows, "curves": curves, "best_epoch": best_epoch}


def global_calibration(run_dir: Path) -> dict:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    rows = D.global_rows(lodo_fold=cfg.get("lodo"), eye2sky_etage=bool(cfg.get("eye2sky_etage", False)))
    grid = FeatureCache(cache_root, "dinov3_vits16", 512, transform="ffr1-patches-all-pool4")
    cls_cache = FeatureCache(cache_root, "dinov3_vits16", 512)
    rows = rows[rows.sample_id.isin(set(grid.index().sample_id))].reset_index(drop=True)
    geo = D.GeometryCache.build(rows, data_root, cache_root / "geometry" / f"pool4_{cfg['name']}_{len(rows)}.npz")
    ds = D.GlobalDataset(rows, grid.rows_for(rows.sample_id), cls_cache.rows_for(rows.sample_id), grid.load("patches"),
                         cls_cache.load("cls"), geo)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GlobalModel(ds.grid.shape[-1], int(cfg["hidden"]), float(cfg["dropout"]), bool(cfg["kendall"])).to(device)
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if state.get("ema"):
        model.load_state_dict({**model.state_dict(), **state["ema"]["shadow"]})
    va = np.flatnonzero(rows.split == "val")
    pred = predict(model, ds, va, device)
    part = rows.iloc[va].reset_index(drop=True)
    out = {}
    full = np.stack(part.etage_w.to_numpy()).min(1) > 0
    truth = np.stack(part.etage.to_numpy())[full]
    conf = pred["etage"][full].ravel()
    hard = (truth > 0.5).astype(float).ravel()
    pos = (conf > 0.5).astype(float)
    out["etage_ece"] = ece(np.where(conf > 0.5, conf, 1 - conf), (pos == hard).astype(float))
    gm = part.mask_genus.to_numpy() & (np.stack(part.genus_neg.to_numpy()).min(1) > 0)      # rows with every genus known
    if gm.any():
        g_truth = (np.stack(part.genus_set.to_numpy())[gm] > 0.5).astype(float).ravel()
        g_conf = pred["genus"][gm].ravel()
        g_pos = (g_conf > 0.5).astype(float)
        out["genus_ece"] = ece(np.where(g_conf > 0.5, g_conf, 1 - g_conf), (g_pos == g_truth).astype(float))
        out["genus_n"] = int(gm.sum())
    om = part.mask_oktas.to_numpy()
    if om.any():
        dist = np.stack(part.oktas.to_numpy())[om]
        top = pred["oktas"][om].argmax(1)
        out["oktas_ece"] = ece(pred["oktas"][om].max(1), (top == dist.argmax(1)).astype(float))
    out["etage_n"] = int(full.sum())
    return out


def cbh_calibration(run_dir: Path) -> dict:
    z = np.load(run_dir / "val_predictions.npz", allow_pickle=True)
    probs, h, days = z["probs"], z["height_m"], z["day"]
    cloudy = ~np.isnan(h)
    lo, hi = B.interval_m(probs[cloudy])
    out = {"coverage_80_raw": float(np.mean((h[cloudy] >= lo) & (h[cloudy] <= hi)))}
    # cross-fitted temperature on the cloud part of the distribution: fit on half the days, apply to the other half
    logits = np.log(np.clip(probs, 1e-9, None))
    uniq = np.array(sorted(set(days[cloudy])))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    halves = [set(uniq[: len(uniq) // 2]), set(uniq[len(uniq) // 2:])]
    temps = np.exp(np.linspace(np.log(0.3), np.log(5.0), 60))

    def coverage(t, m):
        p = np.exp(logits[m, :B.K_BINS] / t)
        p /= p.sum(1, keepdims=True)
        l_, h_ = B.interval_m(np.concatenate([p, np.zeros((len(p), 1))], 1))
        return float(np.mean((h[m] >= l_) & (h[m] <= h_)))
    covered, mae_after = [], []
    chosen = []
    for fit_half, apply_half in ((0, 1), (1, 0)):
        m_fit = cloudy & np.isin(days, list(halves[fit_half]))
        m_apply = cloudy & np.isin(days, list(halves[apply_half]))
        if m_fit.sum() < 20 or m_apply.sum() < 20:
            continue
        t_best = min(temps, key=lambda t: abs(coverage(t, m_fit) - 0.80))
        chosen.append(float(t_best))
        covered.append(coverage(t_best, m_apply))
        p = np.exp(logits[m_apply, :B.K_BINS] / t_best)
        p /= p.sum(1, keepdims=True)
        med = B.median_height_m(np.concatenate([p, np.zeros((len(p), 1))], 1))
        mae_after.append(float(np.nanmean(np.abs(med - h[m_apply]))))
    if covered:
        out["coverage_80_crossfit"] = float(np.mean(covered))
        out["temperature"] = float(np.mean(chosen))
        out["mae_after_temperature"] = float(np.mean(mae_after))
    return out


def jsonl_curve(run_dir: Path, train_key: str, val_key: str) -> tuple[list, list]:
    tr, va = [], []
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if train_key in rec and val_key in rec:
            tr.append(rec[train_key])
            va.append(rec[val_key])
    return tr, va


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--global-run", required=True)
    ap.add_argument("--cbh-run", required=True)
    ap.add_argument("--dense-run", default=None)
    ap.add_argument("--joint-run", default=None)
    ap.add_argument("--lora-run", default=None)
    a = ap.parse_args()
    rows = []
    g = global_curves(Path(a.global_run))
    rows += g["rows"]
    cal = global_calibration(Path(a.global_run))
    cbh_cal = cbh_calibration(Path(a.cbh_run))
    lines = ["# Fit sign-off (P081)", "",
             "Train / val at the best epoch per head and dataset (global heads: étage and genus macro-F1 on the fixed train "
             "subset vs val), the PLAN §4 verdict, and calibration on validation rows. Validation only; the locked test sets "
             "stay untouched.",
             "", "## Global heads (P073 candidate)", "", "| Head | Dataset | Train | Val | Gap | Verdict |",
             "|---|---|---|---|---|---|"]
    for r in g["rows"]:
        lines.append(f"| {r['head']} | {r['dataset']} | {r['train']:.3f} | {r['val']:.3f} | {r['gap']:+.3f} | {r['verdict']} |")
    lines += ["", f"Calibration (val): étage ECE {cal['etage_ece']:.3f} on {cal['etage_n']} rows"
              + (f", genus ECE {cal['genus_ece']:.3f} on {cal['genus_n']} fully-labelled rows" if "genus_ece" in cal else "")
              + (f", oktas top-class ECE {cal['oktas_ece']:.3f}" if "oktas_ece" in cal else "") + "."]
    for name, run in (("dense", a.dense_run), ("joint", a.joint_run), ("lora", a.lora_run)):
        if not run:
            continue
        rd = Path(run)
        if name == "dense":
            tr, va = jsonl_curve(rd, "train_loss", "val_miou")
            if va:
                i = int(np.argmax(va))
                trend = "val still rising at the last epoch (no overfit)" if i == len(va) - 1 else f"val fell after epoch {i + 1}"
                lines += ["", f"## Dense heads ({rd.name})", "",
                          f"Val mIoU curve: {' '.join(f'{v:.3f}' for v in va)}; best epoch {i + 1} ({va[i]:.3f}); "
                          f"train loss {tr[0]:.3f} → {tr[-1]:.3f}; {trend}."]
                rows.append({"run": rd.name, "head": "sky_parse", "dataset": "all", "train": np.nan, "val": va[i], "gap": np.nan,
                             "best_epoch": i + 1,
                             "verdict": "ok (val rising)" if i == len(va) - 1 else f"val fell after epoch {i + 1}"})
        else:
            curves = json.loads((rd / "curves.json").read_text(encoding="utf-8"))
            epochs = curves["epochs"] if isinstance(curves, dict) else curves
            key = "val/headline"
            va = [c.get(key, np.nan) for c in epochs]
            i = int(np.nanargmax(va))
            lines += ["", f"## {name} ({rd.name})", "",
                      f"Headline curve: {' '.join(f'{v:.3f}' for v in va)}; best epoch {i + 1}."]
            rows.append({"run": rd.name, "head": name, "dataset": "all", "train": np.nan, "val": va[i], "gap": np.nan,
                         "best_epoch": i + 1,
                         "verdict": "ok (val rising)" if i == len(va) - 1 else f"val fell after epoch {i + 1}"})
    lines += ["", "## CBH head (P075 candidate)", "",
              f"80 % interval coverage on val: {cbh_cal['coverage_80_raw']:.3f} raw"
              + (f"; with a temperature cross-fitted over the validation days ({cbh_cal['temperature']:.2f}): "
                 f"{cbh_cal['coverage_80_crossfit']:.3f}, MAE of the median after scaling "
                 f"{cbh_cal['mae_after_temperature']:.0f} m"
                 if "coverage_80_crossfit" in cbh_cal else "") + ". The proper calibration split is P089's."]
    rows.append({"run": Path(a.cbh_run).name, "head": "cbh", "dataset": "all", "train": np.nan, "val": cbh_cal["coverage_80_raw"],
                 "gap": np.nan, "best_epoch": np.nan,
                 "verdict": f"coverage {cbh_cal.get('coverage_80_crossfit', np.nan):.3f} after temperature"})
    pd.DataFrame(rows).to_parquet(OUT, index=False)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"-> {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
