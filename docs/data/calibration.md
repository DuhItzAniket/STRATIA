# Calibration (P089)

Per-head temperatures fitted on one half of the validation units and scored on the other (both ways); the parameters in `configs/calibration.json` are fitted on all validation rows. ECE with 15 bins. The locked test sets are not read.

## Global heads

| Head | Quantity | n | ECE before | ECE after | Temperature (cross-fit mean / all-val) | Applied |
|---|---|---|---|---|---|---|
| genus | ECE of the predicted native class (CCSN, MGCD) | 1214 | 0.145 | 0.092 | 3.71 / 4.49 | yes |
| etage | ECE (Bernoulli, exhaustively labelled rows) | 3891 | 0.023 | 0.045 | 1.38 / 1.24 | no (T = 1) |
| oktas | ECE (top class vs majority) | 511 | 0.030 | 0.092 | 0.43 / 0.65 | no (T = 1) |

## CBH head

| Quantity | Raw | Temperature | Temperature + conformal |
|---|---|---|---|
| 80 % interval coverage (cross-fit) | 0.874 | 0.910 | 0.798 |
| 95 % interval coverage (cross-fit) | 0.966 | 0.986 | 0.952 |
| MAE of the median (m, cross-fit) | 561 | 562 | 562 |

CBH temperature 1.24 (cross-fit mean 1.24); conformal radii 0.326 (80 %) and 0.438 (95 %) in CDF units on 1368 cloudy validation pairs.

## OOD threshold

`OodScore` refitted on the trained global head (pooled feature + genus logits, 6,000 training rows); threshold 1.457 = the 95th percentile of in-domain validation scores (flags 5.0% of them by construction) flags 25.3% of 1,500 SWIM segmentation patches (AUROC 0.928).

Parameters: `configs/calibration.json` (model-card `calibration.temperatures`, `ood_threshold`).
