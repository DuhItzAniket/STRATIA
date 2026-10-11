"""Calibration of the Stage H candidates on validation data (P089).

    python scripts/calibrate.py --global-run <run dir> --cbh-run <run dir>

Per-head temperature scaling: one temperature per head (genus and étage sigmoid logits, the oktas CORAL cumulative
logits, the CBH softmax logits), fitted by a 1-D search on the head's own loss. Conformal intervals for the CBH
head: with the temperature applied, the conformity score of a cloudy pair is |F(h) − 0.5| (how far the truth sits
from the predictive median in CDF terms) and the central interval of radius r_α (the ⌈(n+1)(1−α)⌉/n quantile of the
scores) covers 1 − α of exchangeable pairs. OOD threshold: the P068 score refitted on the trained global head
(pooled feature + genus logits), the threshold at the 95th percentile of in-domain validation scores.

Protocol: the validation rows are split into two halves by unit (near-duplicate group / day), every quantity is
fitted on one half and reported on the other, both ways (cross-fit); the parameters written to
`configs/calibration.json` for the model card are then fitted on all validation rows. The locked test sets are not
read. Output: docs/data/calibration.md, data/calibration_report.parquet, data/calibration.json.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.registry import load_paths  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.model import cbh_bins as B  # noqa: E402
from stratia.model.ood import OodScore, auroc  # noqa: E402
from stratia.train import data as D  # noqa: E402
from stratia.train.global_eval import GlobalModel  # noqa: E402

DOC = Path("docs/data/calibration.md")
OUT = Path("data/calibration_report.parquet")
JSON_OUT = Path("configs/calibration.json")
T_GRID = np.exp(np.linspace(np.log(0.2), np.log(5.0), 61))
BINS = 15


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = BINS) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def fit_temperature(loss_fn) -> float:
    """loss_fn(T) -> scalar; grid search over T_GRID."""
    losses = [loss_fn(float(t)) for t in T_GRID]
    return float(T_GRID[int(np.argmin(losses))])


def bernoulli_ece(logits: np.ndarray, target: np.ndarray, weight: np.ndarray, temperature: float = 1.0) -> float:
    p = 1 / (1 + np.exp(-logits / temperature))
    keep = weight > 0
    conf = np.where(p[keep] > 0.5, p[keep], 1 - p[keep])
    correct = ((p[keep] > 0.5) == (target[keep] > 0.5)).astype(float)
    return ece(conf, correct)


def bce_loss(logits: np.ndarray, target: np.ndarray, weight: np.ndarray, temperature: float) -> float:
    z = logits / temperature
    per = np.logaddexp(0, z) - target * z
    return float((per * weight).sum() / max(weight.sum(), 1))


@torch.no_grad()
def global_logits(model: GlobalModel, ds: D.GlobalDataset, idx: np.ndarray, device: str) -> dict[str, np.ndarray]:
    model.eval()
    dl = DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=256, shuffle=False, collate_fn=D.collate)
    gen, eta, cum, pooled = [], [], [], []
    for b in dl:
        out = model(b["tokens"].to(device), b["cls"].to(device), b["ray_map"].to(device), b["meta"].to(device))
        gen.append(out["genus_logits"].float().cpu().numpy())
        eta.append(out["etage_logits"].float().cpu().numpy())
        cum.append(out["oktas_cum_logits"].float().cpu().numpy())
        pooled.append(out["pooled"].float().cpu().numpy())
    return {"genus": np.concatenate(gen), "etage": np.concatenate(eta), "oktas_cum": np.concatenate(cum),
            "pooled": np.concatenate(pooled)}


def halves(units: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = np.array(sorted(set(units.tolist())))
    rng.shuffle(u)
    a = set(u[: len(u) // 2].tolist())
    in_a = np.array([x in a for x in units])
    return in_a, ~in_a


# ----------------------------------------------------------------------------------------------- global heads
def calibrate_global(run_dir: Path, device: str) -> tuple[list[dict], dict, dict]:
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
    model = GlobalModel(ds.grid.shape[-1], int(cfg["hidden"]), float(cfg["dropout"]), bool(cfg["kendall"])).to(device)
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if state.get("ema"):
        model.load_state_dict({**model.state_dict(), **state["ema"]["shadow"]})
    va = np.flatnonzero(rows.split == "val")
    part = rows.iloc[va].reset_index(drop=True)
    out = global_logits(model, ds, va, device)
    units = part.unit.astype(str).to_numpy() + ":" + part.dataset.to_numpy()
    a, b = halves(units)
    report, temps = [], {}
    # genus: the confidence of the predicted native class (noisy-OR score, P073) on CCSN and MGCD rows; T by the ECE
    nat_mask = part.dataset.isin(["ccsn", "mgcd"]).to_numpy() & part.native_label.notna().to_numpy()
    sets = {ds: D.native_class_sets(ds, part.native_label[part.dataset == ds].dropna().unique().tolist())
            for ds in ("ccsn", "mgcd")}

    def genus_conf(T, m):
        probs = 1 / (1 + np.exp(-out["genus"][m] / T))
        conf, correct = [], []
        for ds in ("ccsn", "mgcd"):
            sel = (part.dataset.to_numpy()[m] == ds)
            cs = {k: v for k, v in sets[ds].items() if v[0]}
            keep = sel & np.isin(part.native_label.to_numpy()[m], list(cs))
            if not keep.any():
                continue
            scores, names = D.native_scores(probs[keep], cs)
            conf.append(scores.max(1))
            correct.append((np.array(names)[scores.argmax(1)] == part.native_label.to_numpy()[m][keep]).astype(float))
        return np.concatenate(conf), np.concatenate(correct)

    before, after, ts = [], [], []
    for fit_m, eval_m in ((a & nat_mask, b & nat_mask), (b & nat_mask, a & nat_mask)):
        t = fit_temperature(lambda T, m=fit_m: ece(*genus_conf(T, m)))
        ts.append(t)
        before.append(ece(*genus_conf(1.0, eval_m)))
        after.append(ece(*genus_conf(t, eval_m)))
    temps["genus"] = fit_temperature(lambda T: ece(*genus_conf(T, nat_mask)))
    report.append({"head": "genus", "quantity": "ECE of the predicted native class (CCSN, MGCD)",
                   "before": float(np.mean(before)),
                   "after": float(np.mean(after)), "temperature_crossfit": float(np.mean(ts)), "temperature": temps["genus"],
                   "n": int(nat_mask.sum())})
    # étage: Bernoulli pairs on exhaustively labelled rows, hard targets (shares thresholded), T by the hard BCE
    e_target = (np.stack(part.etage.to_numpy()) > 0.5).astype(float)
    e_weight = (np.stack(part.etage_w.to_numpy()) >= 1.0).all(1)[:, None] * np.ones((1, 3))
    logits = out["etage"]
    before, after, ts = [], [], []
    for fit_m, eval_m in ((a, b), (b, a)):
        t = fit_temperature(lambda T, m=fit_m: bce_loss(logits[m], e_target[m], e_weight[m], T))
        ts.append(t)
        before.append(bernoulli_ece(logits[eval_m], e_target[eval_m], e_weight[eval_m], 1.0))
        after.append(bernoulli_ece(logits[eval_m], e_target[eval_m], e_weight[eval_m], t))
    temps["etage"] = fit_temperature(lambda T: bce_loss(logits, e_target, e_weight, T))
    report.append({"head": "etage", "quantity": "ECE (Bernoulli, exhaustively labelled rows)", "before": float(np.mean(before)),
                   "after": float(np.mean(after)), "temperature_crossfit": float(np.mean(ts)), "temperature": temps["etage"],
                   "n": int((e_weight > 0).sum())})
    o_mask = part.mask_oktas.to_numpy()
    o_dist = np.stack(part.oktas.to_numpy())
    # oktas: CORAL cumulative logits; T by the NLL of the majority class under the class distribution
    from stratia.model.heads import CoralHead

    cum = out["oktas_cum"]

    def oktas_probs(T, m):
        return CoralHead.probs_from_cumulative(torch.from_numpy(cum[m] / T)).numpy()

    def oktas_loss(T, m):
        p = oktas_probs(T, m)
        return float(-np.log(np.clip(p[np.arange(len(p)), o_dist[m].argmax(1)], 1e-9, None)).mean())

    def oktas_ece(T, m):
        p = oktas_probs(T, m)
        return ece(p.max(1), (p.argmax(1) == o_dist[m].argmax(1)).astype(float))

    before, after, ts = [], [], []
    for fit_m, eval_m in ((a & o_mask, b & o_mask), (b & o_mask, a & o_mask)):
        t = fit_temperature(lambda T, m=fit_m: oktas_loss(T, m))
        ts.append(t)
        before.append(oktas_ece(1.0, eval_m))
        after.append(oktas_ece(t, eval_m))
    temps["oktas"] = fit_temperature(lambda T: oktas_loss(T, o_mask))
    report.append({"head": "oktas", "quantity": "ECE (top class vs majority)", "before": float(np.mean(before)),
                   "after": float(np.mean(after)), "temperature_crossfit": float(np.mean(ts)), "temperature": temps["oktas"],
                   "n": int(o_mask.sum())})
    # OOD threshold on the trained head
    tr = np.flatnonzero(rows.split == "train")
    tr_sub = np.random.default_rng(0).choice(tr, min(6000, len(tr)), replace=False)
    out_tr = global_logits(model, ds, tr_sub, device)
    ood = OodScore(out["pooled"].shape[1])
    ood.fit(torch.from_numpy(out_tr["pooled"]), torch.from_numpy(out_tr["genus"]))
    s_in = ood(torch.from_numpy(out["pooled"]), torch.from_numpy(out["genus"]))[:, 0].numpy()
    threshold = float(np.percentile(s_in, 95))
    # an out set the global rows never contain: SWIM segmentation patches
    man = pd.read_parquet("data/manifest.parquet", columns=["sample_id", "dataset", "image_file"])
    swim = man[man.dataset.isin(["swimseg", "swinyseg"])].sample(1500, random_state=0)
    swim = swim[swim.sample_id.isin(set(grid.index().sample_id))]
    sw_rows = pd.DataFrame({"sample_id": swim.sample_id, "dataset": swim.dataset, "source": "swim", "split": "val", "unit": 0,
                            "genus_set": [np.zeros(12, np.float32)] * len(swim), "genus_alt": False,
                            "genus_neg": [np.zeros(12, np.float32)] * len(swim), "mask_genus": False,
                            "etage": [np.zeros(3, np.float32)] * len(swim), "etage_w": [np.zeros(3, np.float32)] * len(swim),
                            "mask_etage": False, "oktas": [np.zeros(9, np.float32)] * len(swim), "mask_oktas": False,
                            "calib_id": None, "station": None, "utc": pd.NaT, "native_label": None, "image_file": swim.image_file,
                            "has_geometry": False}).reset_index(drop=True)
    sw_geo = D.GeometryCache(np.zeros((len(sw_rows), 4, 8, 8), np.float32), np.zeros((len(sw_rows), 3), np.float32))
    sw_ds = D.GlobalDataset(sw_rows, grid.rows_for(sw_rows.sample_id), cls_cache.rows_for(sw_rows.sample_id), ds.grid, ds.cls,
                            sw_geo)
    out_sw = global_logits(model, sw_ds, np.arange(len(sw_rows)), device)
    s_out = ood(torch.from_numpy(out_sw["pooled"]), torch.from_numpy(out_sw["genus"]))[:, 0].numpy()
    ood_info = {"threshold": threshold, "flagged_in_domain_val": float((s_in > threshold).mean()),
                "flagged_swim_patches": float((s_out > threshold).mean()), "auroc_vs_swim": auroc(s_in, s_out),
                "ood_state": {k: v.cpu().numpy().tolist() for k, v in ood.state_dict().items() if k != "fitted"}}
    return report, temps, ood_info


# ----------------------------------------------------------------------------------------------- CBH
def calibrate_cbh(run_dir: Path) -> tuple[list[dict], dict]:
    z = np.load(run_dir / "val_predictions.npz", allow_pickle=True)
    probs, h, days = z["probs"], z["height_m"], z["day"].astype(str)
    cloudy = ~np.isnan(h)
    logits = np.log(np.clip(probs, 1e-9, None))
    target = B.soft_target(h)
    a, b = halves(days[cloudy])
    a_full, b_full = np.zeros(len(h), bool), np.zeros(len(h), bool)
    a_full[np.flatnonzero(cloudy)[a]] = True
    b_full[np.flatnonzero(cloudy)[b]] = True

    def nll(T, m):
        logq = torch.log_softmax(torch.from_numpy(logits[m] / T), 1)
        return float(-(torch.from_numpy(target[m]) * logq).sum(1).mean())

    def scaled(T, m):
        p = np.exp(logits[m, :B.K_BINS] / T)
        return p / p.sum(1, keepdims=True)

    def cdf_at_truth(p, hh):
        cdf = np.cumsum(p, 1)
        idx = B.bin_of(hh)
        lo = np.where(idx > 0, cdf[np.arange(len(p)), np.clip(idx - 1, 0, None)], 0.0)
        hi = cdf[np.arange(len(p)), idx]
        lo_e, hi_e = np.clip(B.EDGES_M[idx], 1.0, None), B.EDGES_M[idx + 1]
        frac = (np.log(np.clip(hh, 1.0, None)) - np.log(lo_e)) / np.log(hi_e / lo_e)
        return lo + np.clip(frac, 0, 1) * (hi - lo)

    def quantile_height(p, q):
        cdf = np.cumsum(p, 1)
        idx = np.clip((cdf >= q[:, None]).argmax(1), 0, B.K_BINS - 1)
        prev = np.where(idx > 0, cdf[np.arange(len(p)), np.clip(idx - 1, 0, None)], 0.0)
        frac = np.clip((q - prev) / np.clip(cdf[np.arange(len(p)), idx] - prev, 1e-12, None), 0, 1)
        lo_e, hi_e = np.log(np.clip(B.EDGES_M[idx], 1.0, None)), np.log(B.EDGES_M[idx + 1])
        return np.exp(lo_e + frac * (hi_e - lo_e))

    def coverage(p, hh, radius):
        lo = quantile_height(p, np.full(len(p), 0.5 - radius))
        hi = quantile_height(p, np.full(len(p), 0.5 + radius))
        return float(np.mean((hh >= lo) & (hh <= hi)))

    report = []
    cov = {"raw80": [], "temp80": [], "conf80": [], "conf95": [], "temp95": [], "raw95": [], "mae_raw": [], "mae_temp": []}
    temps, radii80, radii95 = [], [], []
    for fit_m, eval_m in ((a_full, b_full), (b_full, a_full)):
        t = fit_temperature(lambda T, m=fit_m: nll(T, m))
        temps.append(t)
        p_fit, p_eval = scaled(t, fit_m), scaled(t, eval_m)
        s = np.abs(cdf_at_truth(p_fit, h[fit_m]) - 0.5)
        n = len(s)
        r80 = float(np.quantile(s, min(1.0, np.ceil((n + 1) * 0.8) / n)))
        r95 = float(np.quantile(s, min(1.0, np.ceil((n + 1) * 0.95) / n)))
        radii80.append(r80)
        radii95.append(r95)
        p_raw = scaled(1.0, eval_m)
        cov["raw80"].append(coverage(p_raw, h[eval_m], 0.4))
        cov["raw95"].append(coverage(p_raw, h[eval_m], 0.475))
        cov["temp80"].append(coverage(p_eval, h[eval_m], 0.4))
        cov["temp95"].append(coverage(p_eval, h[eval_m], 0.475))
        cov["conf80"].append(coverage(p_eval, h[eval_m], min(r80, 0.5)))
        cov["conf95"].append(coverage(p_eval, h[eval_m], min(r95, 0.5)))
        med_raw = quantile_height(p_raw, np.full(int(eval_m.sum()), 0.5))
        med_t = quantile_height(p_eval, np.full(int(eval_m.sum()), 0.5))
        cov["mae_raw"].append(float(np.mean(np.abs(med_raw - h[eval_m]))))
        cov["mae_temp"].append(float(np.mean(np.abs(med_t - h[eval_m]))))
    t_all = fit_temperature(lambda T: nll(T, cloudy))
    p_all = scaled(t_all, cloudy)
    s_all = np.abs(cdf_at_truth(p_all, h[cloudy]) - 0.5)
    n = len(s_all)
    r80_all = float(np.quantile(s_all, min(1.0, np.ceil((n + 1) * 0.8) / n)))
    r95_all = float(np.quantile(s_all, min(1.0, np.ceil((n + 1) * 0.95) / n)))
    for q, raw, temp, conf in (("80 %", "raw80", "temp80", "conf80"), ("95 %", "raw95", "temp95", "conf95")):
        report.append({"head": "cbh", "quantity": f"{q} interval coverage (cross-fit)", "raw": float(np.mean(cov[raw])),
                       "temperature": float(np.mean(cov[temp])), "conformal": float(np.mean(cov[conf]))})
    report.append({"head": "cbh", "quantity": "MAE of the median (m, cross-fit)", "raw": float(np.mean(cov["mae_raw"])),
                   "temperature": float(np.mean(cov["mae_temp"])), "conformal": float(np.mean(cov["mae_temp"]))})
    params = {"temperature": t_all, "temperature_crossfit": float(np.mean(temps)), "conformal_radius_80": r80_all,
              "conformal_radius_95": r95_all, "n_cloudy_val": int(cloudy.sum())}
    return report, params


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--global-run", required=True)
    ap.add_argument("--cbh-run", required=True)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g_report, temps, ood_info = calibrate_global(Path(a.global_run), device)
    for r in g_report:                                   # a temperature is kept only when it lowers the held-out ECE
        r["applied"] = bool(r["after"] < r["before"])
        if not r["applied"]:
            temps[r["head"]] = 1.0
    c_report, c_params = calibrate_cbh(Path(a.cbh_run))
    temps["cbh"] = c_params["temperature"]
    cal = {"temperatures": {k: round(float(v), 4) for k, v in temps.items()}, "folded_into_graph": False,
           "cbh_conformal_radius": {"80": round(c_params["conformal_radius_80"], 4),
                                    "95": round(c_params["conformal_radius_95"], 4)},
           "ood_threshold": round(ood_info["threshold"], 4), "ood_statistics": ood_info["ood_state"],
           "fitted_on": "validation rows (cross-fit reported; parameters fitted on all validation rows)",
           "global_run": Path(a.global_run).name, "cbh_run": Path(a.cbh_run).name}
    JSON_OUT.write_text(json.dumps(cal, indent=1), encoding="utf-8")
    rows = g_report + c_report
    pd.DataFrame(rows).to_parquet(OUT, index=False)
    lines = ["# Calibration (P089)", "",
             "Per-head temperatures fitted on one half of the validation units and scored on the other (both ways); the "
             "parameters in `configs/calibration.json` are fitted on all validation rows. ECE with 15 bins. The locked test sets "
             "are not read.", "", "## Global heads", "",
             "| Head | Quantity | n | ECE before | ECE after | Temperature (cross-fit mean / all-val) | Applied |",
             "|---|---|---|---|---|---|---|"]
    for r in g_report:
        lines.append(f"| {r['head']} | {r['quantity']} | {r['n']} | {r['before']:.3f} | {r['after']:.3f} | "
                     f"{r['temperature_crossfit']:.2f} / {r['temperature']:.2f} | {'yes' if r['applied'] else 'no (T = 1)'} |")
    lines += ["", "## CBH head", "", "| Quantity | Raw | Temperature | Temperature + conformal |", "|---|---|---|---|"]
    for r in c_report:
        fmt = "{:.0f}" if "MAE" in r["quantity"] else "{:.3f}"
        lines.append(f"| {r['quantity']} | {fmt.format(r['raw'])} | {fmt.format(r['temperature'])} | "
                     f"{fmt.format(r['conformal'])} |")
    lines += ["", f"CBH temperature {c_params['temperature']:.2f} (cross-fit mean {c_params['temperature_crossfit']:.2f}); "
              f"conformal radii {c_params['conformal_radius_80']:.3f} (80 %) and {c_params['conformal_radius_95']:.3f} (95 %) "
              f"in CDF units on {c_params['n_cloudy_val']} cloudy validation pairs.", "",
              "## OOD threshold", "",
              f"`OodScore` refitted on the trained global head (pooled feature + genus logits, 6,000 training rows); "
              f"threshold {ood_info['threshold']:.3f} = the 95th percentile of in-domain validation scores (flags "
              f"{ood_info['flagged_in_domain_val']:.1%} of them by construction) flags {ood_info['flagged_swim_patches']:.1%} "
              f"of 1,500 SWIM segmentation patches (AUROC {ood_info['auroc_vs_swim']:.3f}).", "",
              "Parameters: `configs/calibration.json` (model-card `calibration.temperatures`, `ood_threshold`)."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines[6:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
