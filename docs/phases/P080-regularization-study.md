# P080 — Regularization study

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Which regularisers move the global heads on validation, in-domain and on the two weak LODO folds, with the seed
spread as the yardstick; plus the like-for-like reference for the secondary C1 comparison (probes with the same
val-tuned thresholds the model's secondary number uses).

## Inputs / dependencies
P073 trainer (`--weight-decay`, `--dropout`, `--ema`, `--hidden`, `--lambda-h`, `--seed`); P078's flat response surface;
P054 probe script (now with `linear_macro_f1_tuned`).

## Work log
1. In-domain (`indomain_e2s` configuration, 20 epochs, early stopping): seeds 0 / 1 / 2; weight decay 0 and 0.2
   (default 0.05); trunk dropout 0 and 0.3 (default 0.1); EMA off and 0.999 (default 0.99). Label smoothing and
   augmentation strength do not apply to cached features with soft / set-valued targets (noted as the plan's two
   knobs that live in P077's image pipeline); drop-path has no place in the heads.
2. LODO on the two weak folds (Montenegro, Almería; with Eye2Sky étage): `small` (hidden 128, dropout 0.3, weight
   decay 0.3), `noh` (hierarchy weight 0), `smallnoh` (both), 12 epochs.
3. `scripts/baseline_probes.py` rerun for the DINOv2 / DINOv3 backbones to add the val-tuned-threshold column.

## Verification
In-domain val mean étage macro-F1 (train subset in brackets):

| Setting | Val | Train | Δ vs default |
|---|---|---|---|
| default, seeds 0 / 1 / 2 | 0.817 / 0.817 / 0.804 (std **0.008**) | 0.936 / 0.940 / 0.928 | — |
| weight decay 0 / 0.2 | 0.816 / 0.815 | 0.934 / 0.927 | −0.001 / −0.002 |
| dropout 0 / 0.3 | 0.804 / 0.817 | 0.931 / 0.909 | −0.013 / 0.000 |
| EMA off / 0.999 | 0.820 / 0.812 | 0.930 / 0.940 | +0.003 / −0.005 |

LODO étage macro-F1 on the held-out val (fixed 0.5 thresholds; val-tuned in brackets):

| Fold | Base (`_e2s`) | Small head | No hierarchy term | Small + no hierarchy |
|---|---|---|---|---|
| Montenegro | 0.360 [0.462] | **0.410** [0.458] | 0.356 [0.456] | 0.415 [0.453] |
| Almería | 0.593 [0.630] | **0.619** [0.643] | 0.591 [0.635] | 0.609 [0.638] |

Probe reference with the same two rules (mean over the four folds): fixed 0.583 (ViT-B @ 224), val-tuned 0.668
(ViT-L @ 224), 0.657 (ViT-B), 0.640 (ViT-S @ 224).

- Reading: in-domain, every regulariser is inside the seed spread except dropout 0 (−0.013, with the largest
  train–val gap); the defaults stand. On the LODO folds a smaller, more regularised head gains 0.03–0.05 at fixed
  thresholds and nothing with tuned thresholds: the regulariser sharpens the calibration a little, not the
  ranking. The hierarchy term is neutral everywhere. Even taking the small head on the two weak folds, the LODO
  mean (0.567) stays below the probe's 0.583 at the pre-registered rule, and 0.650 vs 0.668 at the tuned rule.

## Exit criteria
- [x] Table; choice justified: defaults kept in-domain (dropout 0.1–0.3 indifferent, no weight-decay or EMA
  effect); the small head is the LODO-oriented variant for P093's analysis, not a gate change.

## Fit & data-risk notes
- Two LODO folds × three settings is model selection on the held-out val; the numbers are reported for what they
  are and the gate reads the pre-registered run, not the best of these.
- The tuned-threshold rule favours whoever has the most val rows per column; Almería's 84 rows make its tuned
  numbers the least stable.

## Deviations from plan & why
- Label smoothing and augmentation strength are not studied here (cached features); P077 holds the image
  pipeline and ran one augmentation-strength contrast (1.0 vs 0.5).

## Next phase
P081 — Fit sign-off.
