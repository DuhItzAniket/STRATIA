# P058 — B7: CBH baselines

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
The cloud-base-height references of criterion C3 on the P047 image–ceilometer pairs: what climatology, a
genus→étage lookup, a frozen-feature regressor and a ceilometer-supervised CNN achieve, with MAE, bias and skill by
height bin, before STRATIA's own height head exists.

## Inputs / dependencies
P047 pairs (`data/cbh_pairs.parquet`, day splits in `data/splits/cbh_pairs.parquet`, test days locked); P051 cache
`dinov3_vits16_512` (the 10-minute frames were added for this); P054 étage tables; P022 image cache for the CNN.

## Work log
1. `stratia/baselines/cbh.py`: `score_heights` (MAE, bias, RMSD, within 250 / 500 / 1,000 m, MAE per height bin,
   skill = 1 − MAE / MAE of the train-median climatology), `balanced_accuracy` for cloud presence, climatology
   (train median) and conditional climatology (month × 3-hour bin, train medians, fallback to the global median),
   étage medians and lookup, ridge regression of log10 height on standardised features (α from {1, 10, 100, 1000}
   chosen on val).
2. `scripts/baseline_cbh.py`, 10-minute cadence, train → val (14,366 / 2,193 pairs; 1,368 cloudy val pairs scored
   on the lowest reported base):
   - **climatology_median**, **climatology_month_hour**;
   - **etage_lookup**: the frozen DINOv3 ViT-S @ 512 étage probe (trained on the CCSN / MGCD / Montenegro / Almería
     étage sets, as in P054) predicts low / mid / high / none for each frame; the prediction maps to the train
     pairs' median base per étage (none → climatology); cloud presence = probe says any étage;
   - **ridge_dinov3_vits16_512**: ridge on the same embedding (cloudy train pairs); cloud presence from a logistic
     probe on the embedding (none vs cloud);
   - **cnn_resnet18_logheight**: ResNet-18 (ImageNet init) regressing log10 height from the 224 px image, 6 epochs,
     cloudy train pairs, seed 0.
   One registry run per baseline (phase P058); rows in `data/baseline_cbh.parquet`, table in `docs/data/baseline_cbh.md`.
3. Zero-shot DA3 / MoGe-2 are not run: they mask the sky by design and return no cloud geometry (research design §1);
   the plan's "show sky masked" item is that sentence.

## Verification
- `tests/test_cbh_baselines.py` (3): scores on a synthetic set (skill 0 for the climatology by construction, bins),
  conditional climatology and étage lookup on toy tables, the ridge recovers a linear log-height relation.
- Cloudy val pairs (n = 1,368), metres:

| Baseline | MAE | Bias | Skill | ≤ 500 m | ≤ 1 km | MAE 0–1 km | 1–2 km | 2–4 km | 4–8 km | 8–20 km |
|---|---|---|---|---|---|---|---|---|---|---|
| climatology (median) | 1147 | −312 | 0.000 | 0.27 | 0.67 | 1052 | 360 | 916 | 3515 | 7883 |
| climatology (month × 3 h) | 1131 | −330 | +0.014 | 0.32 | 0.70 | 818 | 475 | 911 | 3506 | 8104 |
| étage lookup (frozen probe) | 1124 | −767 | +0.020 | 0.44 | 0.68 | 451 | 359 | 1495 | 3976 | 6852 |
| ridge on frozen DINOv3 | **699** | +34 | **+0.390** | 0.61 | 0.78 | 303 | 486 | 853 | 1859 | 2903 |
| ResNet-18, ceilometer-supervised | 724 | −42 | +0.368 | 0.63 | 0.80 | 322 | 468 | 816 | 2247 | 3171 |

  Cloud presence (balanced accuracy, all 2,193 val pairs): étage lookup 0.656, logistic probe on the embedding 0.909.
- Reading: a linear map from frozen DINOv3 features explains 39 % of the climatology error and matches the
  supervised CNN (the CNN is better inside 1 km, the ridge above 4 km). Both are bias-free on the mean. The étage
  lookup is no better than climatology on MAE: a correct étage still leaves the base anywhere inside a 0–2 / 2–6 /
  6+ km band, and the probe's cross-dataset étage error (P054) adds to it; its −767 m bias is the "low" median
  applied to mid and high clouds. Error grows with height for every method: the 8–20 km bin (cirrus) is where the
  ceilometer's own returns are weakest and the picture carries the least base information.

## Exit criteria
- [x] MAE / bias / skill table by height bin.

## Fit & data-risk notes
- The val days are the P047 day split (12 val days per station against 81 train days): no day is shared with train,
  and the 2,193 val pairs are day-correlated, so the intervals are wider than n suggests.
- The ridge's α and the CNN's best epoch are chosen on val; the comparison is fair across baselines and will be
  repeated once on the locked test days at P090.

## Deviations from plan & why
- No DA3 / MoGe-2 run (work log 3). The CNN is ResNet-18 at 6 epochs (plan: "ceilometer-only CNN regressor",
  architecture unspecified): enough to converge on 14k pairs within the stage's budget.

## Next phase
P059 — Baseline leaderboard.
