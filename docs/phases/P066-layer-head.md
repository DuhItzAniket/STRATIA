# P066 — Layer head

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
The dense cloud-layer output (low / mid / high on cloud pixels) from the same decoder trunk as sky parsing, trained
on Almería's layer masks, with confidence intervals because the labelled sets are tiny.

## Inputs / dependencies
P065 decoder and trainer; P018 Almería masks (classes 2 / 3 / 4 → low / mid / high on cloud pixels, 255 elsewhere;
654 images in the grouped train + val, 48-image test set from four cameras locked); P064 `dense_ce`.

## Work log
1. `LayerHead` (1 × 1 on the decoder's 64-channel map, P065) trained jointly with sky parsing:
   `scripts/train_dense_heads.py --phase P066 --epochs 15 --layers` (layer weight 1.0), loss = sky CE + layer CE
   on cloud pixels; the SWIM datasets contribute sky-parsing targets only (their layer targets are all 255).
2. Scoring: per-class layer IoU on Almería val (the only dataset with layer truth), layer mIoU with a bootstrap
   95 % interval over val images (1,000 resamples of per-image confusion matrices), sky parsing as in P065.

## Verification
- `tests/test_dense.py` covers the shapes of both heads; the layer target convention (255 off cloud) is tested in
  `tests/test_segmentation.py` (P018 `load_masks`).
- Almería val (grouped), 15 epochs, best epoch 15:

| Quantity | Value |
|---|---|
| Layer mIoU | **0.559** [0.463, 0.659] |
| IoU low / mid / high | 0.575 / 0.344 / 0.758 |
| Sky-parsing mIoU, Almería (joint run) | 0.873 [0.854, 0.892] (P065 alone: 0.902) |
| Sky-parsing mIoU, all datasets (joint run) | 0.894 [0.889, 0.899] (P065 alone: 0.919) |
| Clear-sky cloud fraction (joint run) | 0.357 (P065 alone: 0.220) |

- Reading: the layer head learns the extremes (high 0.76, low 0.58) and confuses the middle layer (0.34), which is
  what one expects from a per-pixel label whose definition is a height band seen through texture alone. The wide
  interval (±0.10) is the small val set: the plan's guard. **Negative transfer:** with the layer loss at weight 1.0
  on the shared trunk, sky parsing loses 0.025 mIoU and the clear-sky cloud fraction rises from 0.22 to 0.36; the
  layer loss is larger (it dominates the 0.42 total) and pulls the trunk. This is the first concrete case for the
  P070 weighting and the P076 joint-vs-single comparison; P074 trains the dense heads with the Kendall weights.

## Exit criteria
- [x] Trains on Almería; confidence intervals reported (bootstrap 95 % over images).

## Fit & data-risk notes
- Almería's layer masks come from one site and one labeller; the per-class IoU is a within-site number. The
  ceilometer weak labels (P075) are the second source of layer supervision and the test of whether "mid" is
  learnable at all from one image.
- The layer-head curve was still rising at epoch 15 (0.43 → 0.56); longer training is a Stage H choice, taken with
  the sky-parsing trade-off in view.

## Deviations from plan & why
- Trained jointly with sky parsing rather than alone (one decoder trunk, ADR-005); the joint cost on sky parsing is
  reported instead of hidden.

## Next phase
P067 — CBH head.
