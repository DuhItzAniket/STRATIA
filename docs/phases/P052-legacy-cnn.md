# P052 — B1: legacy CNN

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
The CloudScope prototype's recipe (MobileNetV3-L, ImageNet init, Adam 1e-3, plateau schedule, early stopping) on
CCSN under a per-image random split and under the grouped split, so the paper has a measured protocol-gap number
instead of the usual assumption that random splits flatter.

## Inputs / dependencies
P022 image cache; P038 grouped in-domain split (units = near-duplicate groups, P026); P026 label-conflict findings;
P040 test lock.

## Work log
1. `stratia/baselines/train_cnn.py` (shared with P053): trainer, dataset, evaluation and the PLAN §4 fit verdict.
   Legacy recipe: batch 32, lr 1e-3, weight decay 1e-4, no label smoothing, ReduceLROnPlateau on val macro-F1,
   patience 5, up to 20 epochs, 224 px, AMP.
2. `scripts/baseline_cnn.py --phase P052 --recipe legacy --protocols legacy_random grouped --seeds 0 1 2`.
   **Protocol design, revised during the phase.** The first version recreated the prototype's 70/15/15 split over
   the whole dataset, which reads the locked test rows (as training images and as a "leaky test"), and its train set
   differed in size from the grouped one. The final version draws the random split from the same unlocked pool
   (grouped train + val, 2,031 images) with the grouped split's per-class val counts, one draw per seed, so the two
   protocols differ only in whether near-duplicate units stay together (train 1,776 / val 255 in both). The random
   val is scored whole and in two parts: *leaked* images (their unit has a member in the random train set) and
   *clean* ones. No locked row is read.
3. Six runs (3 seeds × 2 protocols), 6 minutes; rows in `data/baseline_cnn.parquet`, table in
   `docs/data/baseline_cnn.md`, the protocol-gap table in `docs/data/baseline_leaderboard.md`.

## Verification
- `tests/test_train_cnn.py` (3, `slow`): dataset tensors, a tiny end-to-end run, verdict rules.
- Val macro-F1 (accuracy), mean over seeds 0–2:

| Protocol | Whole val | Leaked part | Clean part | Std over seeds |
|---|---|---|---|---|
| grouped | **0.588** (0.595) | none by construction | 0.588 | 0.018 |
| random, per image | 0.511 (0.525) | 0.146 (0.160), n = 15–29 | 0.543 (0.563) | 0.029 |

- Gap random → grouped: **−0.076**. The random split scores *lower*, and the reason is in its leaked part: of the
  15 / 29 / 29 leaked val images per seed, 14 / 24 / 26 have a near-duplicate in train filed under a *different*
  genus (the P026 label conflicts). The network predicts the twin's label and is scored wrong: leaked accuracy
  0.13–0.21 against 0.54–0.59 for the clean part. CCSN's near-duplicate groups are overwhelmingly conflicting
  duplicates, not benign copies, so a random split is contaminated rather than flattered. The clean part of the random
  val (0.543) still sits 0.045 below the grouped val: the grouped val units were drawn with the P038 `test_ok` flag
  (19 flagged images of 255 against 14–26 in a random val), which removes some of the ambiguous pictures.
- Fit: every run overfits or leans that way (train–val gap 0.16–0.35, best epoch 5–15); the recipe's early stopping
  took the best epoch.

## Exit criteria
- [x] Protocol-gap number (random vs honest): −0.076 on CCSN with MobileNetV3-L, with the per-image decomposition
  that explains its sign.

## Fit & data-risk notes
- The leaked part is 15–29 images; its macro-F1 is dominated by absent classes, which is why accuracy is quoted next
  to it. The sign of the effect is the same in all three seeds.
- The prototype's own published number (a random split over all 2,543 images, duplicates on both sides) cannot be
  reproduced without the locked rows and is not attempted; its protocol is characterised by the experiment above.

## Deviations from plan & why
- "Random split" is size-matched and drawn inside the unlocked pool (work log 2), not the prototype's exact split;
  the plan's "random vs honest" number is reported per image rather than as a leaky test score.

## Next phase
P053 — B2: fine-tuned CNNs.
