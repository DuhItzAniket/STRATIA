# P079 — Learning curves

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Performance against the amount of training data for the two heads the gate reads (global étage / genus and CBH):
is each regime data-limited (more data would help) or capacity-limited (it would not)?

## Inputs / dependencies
P073 trainer (`--train-fraction`: drops a share of the training *units*, i.e. near-duplicate groups, per dataset);
P075 trainer (`--train-fraction`: keeps a share of the training *days*); the P073 / P075 full-data runs.

## Work log
1. Global heads at 10 / 25 / 50 / 100 % of the training units (in-domain split with Eye2Sky étage rows, 20 epochs,
   early stopping), names `lc_0.1`, `lc_0.25`, `lc_0.5`; CBH head at 10 / 25 / 50 / 100 % of the training days
   (10 epochs), variants `(train days f)`. Same validation sets throughout.

## Verification
| Training share | Global mean étage F1 (val) | Global mean genus F1 (val) | Train étage F1 | CBH MAE (m) | CBH skill |
|---|---|---|---|---|---|
| 10 % | 0.662 | 0.542 | 0.930 | 756 | +0.330 |
| 25 % | 0.698 | 0.661 | 0.989 | 661 | +0.420 |
| 50 % | 0.774 | 0.687 | 0.956 | 580 | +0.503 |
| 100 % | **0.817** | **0.720** | 0.936 | **566** | **+0.506** |

- Global heads: every doubling still helps (+0.036, +0.076, +0.043 étage; +0.12, +0.03, +0.03 genus) and the train
  score stays above 0.93 at every share: **data-limited** with a memorising head. The next step is more labelled
  variety, not a bigger head: P083's B0268 labels and the Stage I pseudo-labels are the data the curve asks for.
- CBH head: the last doubling of days buys 14 m (580 → 566, close to the 10 m seed std) after 81 m and 95 m for the
  two before: the frozen-feature CBH regression is **approaching its capacity** at the 8 × 8 pooled grid; the
  levers from here are the representation (P077's LoRA, the full-resolution grid) and the geometry
  (P067's finding), not more Eye2Sky days of the same two stations.

## Exit criteria
- [x] Curves + regime diagnosis.

## Fit & data-risk notes
- Units, not images, are subsampled for the global heads, so the small-share runs keep the grouped protocol; the
  10 % run stops at epoch 1 (it overfits the moment it has learned the few units), the 25 % run at epoch 16.
- The CBH days are subsampled at random; at 10 % (8 days) the climatology itself changes little (the median is
  stable), so the skill numbers remain comparable.

## Deviations from plan & why
- None.

## Next phase
P080 — Regularization study.
