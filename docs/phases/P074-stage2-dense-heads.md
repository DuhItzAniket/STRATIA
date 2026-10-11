# P074 — Stage-2: dense heads (frozen backbone)

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Train the two dense heads together on the frozen backbone with the P070 weighting, and install the
predicted-class-fraction monitor on clear-sky Eye2Sky frames and on the owner's B0268 frames: the guard against the
legacy "85–99 % cloud" collapse.

## Inputs / dependencies
P065 / P066 decoder, trainer and val scores; P070 `UncertaintyWeighting`; P047 clear-sky frames and the 25 legacy
B0268 images in the manifest (the P041 captures are still pending); P051 patch-token cache.

## Work log
1. `scripts/train_dense_heads.py`: `--kendall` (one learned log-variance per dense loss), the class-fraction monitor
   (`class_fractions`: mean predicted share of invalid / sky / cloud / sun-glare inside the camera mask) on the
   clear-sky frames and on the B0268 frames, both written to the result rows; the data helpers moved to
   `stratia/train/dense_data.py` for P076 / P077.
2. Run: `--phase P074 --epochs 15 --layers --kendall` (18 minutes).

## Verification
Grouped val, 891 images; the P066 fixed-weight joint run and the P065 sky-only run as references:

| Run | Sky mIoU (all) | Almería layer mIoU [95 % CI] | low / mid / high | Clear-sky cloud fraction | B0268 cloud fraction |
|---|---|---|---|---|---|
| P065 sky only | **0.919** | — | — | **0.220** | — |
| P066 sky + layer, weight 1.0 | 0.894 | 0.559 [0.463, 0.659] | 0.575 / 0.344 / 0.758 | 0.357 | — |
| **P074 sky + layer, Kendall** | 0.905 | 0.559 [0.465, 0.656] | 0.577 / 0.341 / 0.758 | 0.299 | 0.443 |

- The learned weights settle at 6.2 (sky) against 1.6 (layer): the weighting recovers 0.011 of the 0.025 mIoU the
  fixed-weight run had lost and 0.06 of the clear-sky cloud fraction, at an unchanged layer score. The per-class
  layer picture is the same as P066 (mid 0.34 is the hard class).
- Monitor on the clear-sky Eye2Sky frames: invalid 0.000, sky 0.701, cloud 0.299, sun-glare 0.000 (the glare
  class has no labels yet and is never predicted). On the 25 B0268 legacy frames: sky 0.556, cloud 0.443,
  invalid 0. These frames are not known to be clear (no P041 labels yet), so the B0268 number is a reference
  for the P041 / P083 comparison rather than a collapse verdict; the prototype's 85–99 % is far away.

## Exit criteria
- [x] Per-class IoU (sky, cloud, invalid; low / mid / high).
- [x] Predicted-class-fraction monitor on clear-sky and B0268 frames recorded in the result rows.

## Fit & data-risk notes
- The sky-only head remains the best sky parser (0.919 vs 0.905); the joint model pays 0.014 mIoU for the layer
  head on the shared trunk even with the weighting. P076 reads this trade across all heads; a separate dense
  trunk for layers is the fallback if the joint loss keeps the gap.
- No photometric augmentation is possible on cached tokens; the glare class waits for P077 (backbone in the
  graph) or P083's B0268 labels.

## Deviations from plan & why
- The plan's "B0268 frames" are the 25 legacy images (the logger captures of P041 do not exist yet); documented
  as a reference number, not a clear-sky test.

## Next phase
P075 — Stage-3: CBH head.
