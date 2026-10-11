"""Gate G4: the model against the pre-registered criteria on validation data (P082; research design §5).

    python scripts/gate_g4.py [--lodo-suffix _e2s]

C1: LODO étage macro-F1 averaged over the held-out datasets (CCSN, MGCD, Montenegro, Almería) at least 5 points above
    the best baseline (P059: 0.583), and in-domain macro-F1 no more than 1 point below the best baseline (the best
    probe's mean genus macro-F1 over the sources the model is scored on: CCSN, MGCD, SWIMCAT).
C2: Montenegro model–expert agreement at least 90 % of expert–expert agreement on étage (set) and on total cover
    (oktas within ±1), from the P073 in-domain run.
C3: zenith CBH skill over climatology ≥ 0.20 on the CDLRA (OLDLR) validation weeks and ≥ 0.10 on the held-out CDLRB
    (WESTE) site (the OLDLR-only training run), MAE at least 10 % below the ceilometer-supervised CNN (P058: 724 m),
    80 % interval coverage within ±5 points of nominal (after the cross-fitted temperature of P081).
GO = C3 and (C1 or C2). Output: docs/data/gate_g4.md, data/gate_g4.parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DOC = Path("docs/data/gate_g4.md")
OUT = Path("data/gate_g4.parquet")
FOLDS = ("ccsn", "mgcd", "montenegro", "almeria")
GENUS_SOURCES = ("ccsn", "mgcd", "swimcat")


def val_metric(df: pd.DataFrame, name: str, dataset: str, metric: str, seed: int = 0) -> float:
    r = df[(df.name == name) & (df.seed == seed) & (df.split == "val") & (df.dataset == dataset) & (df.metric == metric)]
    return float(r.value.iloc[0]) if len(r) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lodo-suffix", default="_e2s", help="which LODO runs: '' (sources only) or '_e2s' (+ Eye2Sky étage)")
    ap.add_argument("--indomain", default="indomain_e2s")
    a = ap.parse_args()
    g = pd.read_parquet("data/stage1_global.parquet")
    probes = pd.read_parquet("data/baseline_probes.parquet")
    cbh = pd.read_parquet("data/cbh_head.parquet")
    fit = pd.read_parquet("data/fit_report.parquet") if Path("data/fit_report.parquet").exists() else pd.DataFrame()
    rows = []

    # ---- C1
    lodo = {f: val_metric(g, f"lodo_{f}{a.lodo_suffix}", f, "etage_f1") for f in FOLDS}
    lodo_mean = float(np.mean(list(lodo.values())))
    base_lodo = probes[probes.protocol == "lodo_etage"].groupby(["backbone", "size"]).linear_macro_f1.mean().max()
    ind_model = {s: val_metric(g, a.indomain, s, "genus_f1") for s in GENUS_SOURCES}
    ind_model = {s: v for s, v in ind_model.items() if v == v}            # sources the model is scored on
    ind_mean = float(np.mean(list(ind_model.values())))
    pr = probes[(probes.protocol == "in_domain") & (probes.source.isin(list(ind_model)))]   # same sources as the model
    base_ind = pr.groupby(["backbone", "size"]).linear_macro_f1.mean().max()
    c1_a = lodo_mean - base_lodo >= 0.05
    # secondary (not pre-registered): per-column thresholds chosen on the held-out val, for model and probes alike
    lodo_t = {f: val_metric(g, f"lodo_{f}{a.lodo_suffix}", f, "etage_f1_tuned") for f in FOLDS}
    lodo_t_mean = float(np.mean(list(lodo_t.values())))
    base_t = (probes[probes.protocol == "lodo_etage"].groupby(["backbone", "size"]).linear_macro_f1_tuned.mean().max()
              if "linear_macro_f1_tuned" in probes.columns else float("nan"))
    c1_b = ind_mean >= base_ind - 0.01
    rows += [{"criterion": "C1", "quantity": "LODO étage macro-F1 (mean of folds)", "model": lodo_mean,
              "reference": float(base_lodo),
              "margin": lodo_mean - base_lodo, "required": "≥ +0.05", "pass": bool(c1_a)},
             {"criterion": "C1 (secondary, val-tuned thresholds)", "quantity": "LODO étage macro-F1, thresholds chosen on val",
              "model": lodo_t_mean, "reference": float(base_t), "margin": lodo_t_mean - base_t,
              "required": "≥ +0.05 (not gating)",
              "pass": bool(lodo_t_mean - base_t >= 0.05) if base_t == base_t else False},
             {"criterion": "C1", "quantity": "in-domain genus macro-F1 (" + ", ".join(ind_model).upper() + ")", "model": ind_mean,
              "reference": float(base_ind),
              "margin": ind_mean - base_ind, "required": "≥ −0.01", "pass": bool(c1_b)}]
    # ---- C2
    c2_e = val_metric(g, a.indomain, "montenegro", "c2_etage_ratio")
    c2_o = val_metric(g, a.indomain, "montenegro", "c2_oktas_ratio")
    rows += [{"criterion": "C2", "quantity": "Montenegro étage-set agreement / human ceiling", "model": c2_e, "reference": 0.90,
              "margin": c2_e - 0.90, "required": "≥ 0.90", "pass": bool(c2_e >= 0.90)},
             {"criterion": "C2", "quantity": "Montenegro oktas ±1 agreement / human ceiling", "model": c2_o, "reference": 0.90,
              "margin": c2_o - 0.90, "required": "≥ 0.90", "pass": bool(c2_o >= 0.90)}]
    # ---- C3
    base = cbh[cbh.variant == "with geometry [P075 base]"]
    base = base[base.seed == 0] if (base.seed == 0).any() else base
    held = cbh[cbh.variant == "with geometry (OLDLR only; WESTE held out)"]
    oldlr_skill = float(base.OLDLR_skill.iloc[0]) if len(base) and "OLDLR_skill" in base else float("nan")
    weste_held = float(held.WESTE_skill.iloc[0]) if len(held) and "WESTE_skill" in held else float("nan")
    mae = float(base.mae_m.iloc[0]) if len(base) else float("nan")
    has_cbh = len(fit) and (fit["head"] == "cbh").any()
    cov = float(fit[fit["head"] == "cbh"].verdict.iloc[0].split("coverage ")[1].split(" ")[0]) if has_cbh else float("nan")
    cov_raw = float(fit[fit["head"] == "cbh"].val.iloc[0]) if len(fit) and (fit["head"] == "cbh").any() else float("nan")
    rows += [{"criterion": "C3", "quantity": "CBH skill, CDLRA (OLDLR) val weeks", "model": oldlr_skill, "reference": 0.20,
              "margin": oldlr_skill - 0.20, "required": "≥ 0.20", "pass": bool(oldlr_skill >= 0.20)},
             {"criterion": "C3", "quantity": "CBH skill, held-out CDLRB (WESTE) site, OLDLR-only training", "model": weste_held,
              "reference": 0.10, "margin": weste_held - 0.10, "required": "≥ 0.10", "pass": bool(weste_held >= 0.10)},
             {"criterion": "C3", "quantity": "CBH MAE (m) vs ceilometer-supervised CNN", "model": mae, "reference": 724.0,
              "margin": 0.9 * 724.0 - mae, "required": "≤ 652 (10 % below 724)", "pass": bool(mae <= 0.9 * 724.0)},
             {"criterion": "C3", "quantity": "80 % interval coverage after cross-fitted temperature (raw in reference)",
              "model": cov, "reference": cov_raw, "margin": 0.05 - abs(cov - 0.80), "required": "0.75–0.85",
              "pass": bool(abs(cov - 0.80) <= 0.05)}]
    df = pd.DataFrame(rows)
    c1 = all(df[df.criterion == "C1"]["pass"])                      # the secondary row has its own label
    c2 = all(df[df.criterion == "C2"]["pass"])
    c3 = all(df[df.criterion == "C3"]["pass"])
    go = c3 and (c1 or c2)
    df.to_parquet(OUT, index=False)
    lines = ["# Gate G4 — model vs baselines on validation data (P082)", "",
             f"Pre-registered criteria (research design §5), validation only. LODO runs: `lodo_<fold>{a.lodo_suffix}`; "
             f"in-domain run: "
             f"`{a.indomain}`; CBH: P075 base and the OLDLR-only run; coverage from the P081 fit report.", "",
             "| Criterion | Quantity | Model | Reference | Margin | Required | Pass |", "|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.criterion} | {r.quantity} | {r.model:.3f} | {r.reference:.3f} | {r.margin:+.3f} | {r.required} | "
                     f"{'yes' if r._7 else 'no'} |")
    lines += ["", "LODO folds: " + ", ".join(f"{f} {v:.3f}" for f, v in lodo.items()) + " (val-tuned thresholds: "
              + ", ".join(f"{f} {v:.3f}" for f, v in lodo_t.items()) + ").",
              "In-domain genus macro-F1: " + ", ".join(f"{s} {v:.3f}" for s, v in ind_model.items()) + ".", "",
              f"C1 {'passes' if c1 else 'fails'}; C2 {'passes' if c2 else 'fails'}; C3 {'passes' if c3 else 'fails'}.", "",
              f"**Decision: {'GO' if go else 'NO-GO'}** (GO requires C3 and at least one of C1, C2)."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines[-3:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
