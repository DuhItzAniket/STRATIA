# Gate G4 — model vs baselines on validation data (P082)

Pre-registered criteria (research design §5), validation only. LODO runs: `lodo_<fold>_e2s`; in-domain run: `indomain_e2s`; CBH: P075 base and the OLDLR-only run; coverage from the P081 fit report.

| Criterion | Quantity | Model | Reference | Margin | Required | Pass |
|---|---|---|---|---|---|---|
| C1 | LODO étage macro-F1 (mean of folds) | 0.548 | 0.583 | -0.035 | ≥ +0.05 | no |
| C1 (secondary, val-tuned thresholds) | LODO étage macro-F1, thresholds chosen on val | 0.650 | 0.668 | -0.017 | ≥ +0.05 (not gating) | no |
| C1 | in-domain genus macro-F1 (CCSN, MGCD) | 0.719 | 0.716 | +0.003 | ≥ −0.01 | yes |
| C2 | Montenegro étage-set agreement / human ceiling | 1.047 | 0.900 | +0.147 | ≥ 0.90 | yes |
| C2 | Montenegro oktas ±1 agreement / human ceiling | 1.027 | 0.900 | +0.127 | ≥ 0.90 | yes |
| C3 | CBH skill, CDLRA (OLDLR) val weeks | 0.471 | 0.200 | +0.271 | ≥ 0.20 | yes |
| C3 | CBH skill, held-out CDLRB (WESTE) site, OLDLR-only training | 0.448 | 0.100 | +0.348 | ≥ 0.10 | yes |
| C3 | CBH MAE (m) vs ceilometer-supervised CNN | 566.100 | 724.000 | +85.500 | ≤ 652 (10 % below 724) | yes |
| C3 | 80 % interval coverage after cross-fitted temperature (raw in reference) | 0.802 | 0.917 | +0.048 | 0.75–0.85 | yes |

LODO folds: ccsn 0.489, mgcd 0.749, montenegro 0.360, almeria 0.593 (val-tuned thresholds: ccsn 0.565, mgcd 0.946, montenegro 0.462, almeria 0.630).
In-domain genus macro-F1: ccsn 0.626, mgcd 0.812.

C1 fails; C2 passes; C3 passes.

**Decision: GO** (GO requires C3 and at least one of C1, C2).
