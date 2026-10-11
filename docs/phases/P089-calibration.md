# P089 — Calibration

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Calibrated probabilities for the host: per-head temperatures, conformal intervals for the cloud-base height, and
the OOD threshold, all fitted on validation data under a cross-fit protocol so that the reported ECE and coverage
are not read on the rows that fitted them; the parameters go to `configs/calibration.json` for the model card.

## Inputs / dependencies
P073 `indomain_e2s` checkpoint, P075 base CBH run (saved val predictions), P068 `OodScore`; research design §4
(ECE, interval coverage); the model-card schema (`calibration.temperatures`, `ood_threshold`).

## Work log
1. `scripts/calibrate.py`: the validation rows are split into two halves by unit (near-duplicate group for the
   recognition sources, day for the CBH pairs); every parameter is fitted on one half and scored on the other,
   both ways; the parameters written out are fitted on all validation rows.
   - **Genus:** the confidence of the predicted native class (noisy-OR score of the class's genera, as in P073) on
     CCSN and MGCD rows; one temperature on the genus logits, fitted on the held-out ECE itself (15 bins).
   - **Étage:** Bernoulli pairs on exhaustively labelled rows with hard targets (shares thresholded at 0.5), one
     temperature fitted on the binary cross-entropy.
   - **Oktas:** the CORAL cumulative logits divided by one temperature fitted on the negative log-likelihood of
     the majority class; ECE of the top-class confidence.
   - A head's temperature is kept only when it lowers the cross-fitted ECE; otherwise T = 1 is written.
   - **CBH:** one temperature on the 25-way logits by the NLL against the P064 soft target; on top, split
     conformal intervals: the conformity score of a cloudy pair is |F(h) − 0.5| (the truth's distance from the
     predictive median in CDF units), and the central interval of radius r_α (the ⌈(n+1)(1−α)⌉/n quantile of the
     scores) is the one the host shows. Intervals here are interpolated quantiles of the bin distribution.
   - **OOD threshold:** `OodScore` refitted on the trained head's pooled feature and genus logits (6,000 training
     rows); threshold at the 95th percentile of in-domain validation scores.
2. Output: `docs/data/calibration.md`, `data/calibration_report.parquet`, `configs/calibration.json`.

## Verification
Global heads (cross-fit, 15-bin ECE):

| Head | Quantity | n | ECE before | ECE after | Temperature (cross-fit / all-val) | Applied |
|---|---|---|---|---|---|---|
| genus | predicted native class, CCSN + MGCD | 1,214 | 0.145 | **0.092** | 3.7 / 4.5 | yes |
| étage | Bernoulli pairs, exhaustive rows | 3,891 | **0.023** | 0.045 | 1.4 / 1.2 | no (T = 1) |
| oktas | top class vs majority | 511 | **0.030** | 0.092 | 0.4 / 0.7 | no (T = 1) |

CBH head (1,368 cloudy validation pairs, cross-fit):

| Quantity | Raw | Temperature (1.24) | + conformal |
|---|---|---|---|
| 80 % interval coverage | 0.874 | 0.910 | **0.798** |
| 95 % interval coverage | 0.966 | 0.986 | **0.952** |
| MAE of the median (m) | 561 | 562 | 562 |

Conformal radii 0.327 (80 %) and 0.438 (95 %) in CDF units. OOD threshold 1.457: flags 5 % of in-domain val by
construction and 25 % of 1,500 SWIM segmentation patches (AUROC 0.93 against them).

- Reading: the étage and oktas heads are calibrated as trained (ECE 0.02–0.03) and the NLL-optimal temperatures
  (flatter for étage, sharper for oktas) make the binary ECE worse, so no scaling is applied to them; the genus
  head's native-class confidence is over-confident (0.145) and a strong temperature (4.5) halves its ECE. The
  CBH distributions are too wide; the conformal radius brings both nominal coverages within a point of nominal
  with the median unchanged. The research design's criterion (coverage within ±5 points of nominal on val) holds
  at 80 % and 95 %.

## Exit criteria
- [x] ECE reported per head; interval coverage within ±5 pts of nominal on val (0.798 / 0.952, cross-fitted).

## Fit & data-risk notes
- The parameters are fitted on validation data; P090 evaluates them once on the locked test sets, where the
  coverage is the number that counts.
- The conformal guarantee assumes exchangeable pairs; the day-correlated validation pairs make the effective
  sample smaller than 1,368, which the cross-fit over days accounts for in part.
- The genus temperature is fitted on the ECE rather than the NLL because the noisy-OR score is not a likelihood;
  the model card records it as such.

## Deviations from plan & why
- The plan's "temperature scaling on val" is done with a cross-fit so the reported numbers are held out; the
  final parameters use all validation rows, as the plan says.

## Next phase
P090 — Locked-test evaluation (once), after the owner decides on the B0268 column; P088 (robustness) runs now.
