# P053 — B2: fine-tuned CNNs

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
The supervised convolutional baselines of the paper: ResNet-50 and ConvNeXt-T fine-tuned end to end on each
recognition source under the grouped protocol, three seeds each, with fit curves and a mean ± std the gate can judge.

## Inputs / dependencies
P022 image cache; P038 grouped splits; P034 native classes; P052 trainer; timm 1.0.30 ImageNet weights.

## Work log
1. `stratia/baselines/train_cnn.py`: one trainer for P052 and P053. `TrainConfig` (arch, size, epochs, batch, lr,
   weight decay, label smoothing, schedule cosine | plateau, patience, seed, AMP, crop scale); `ImageClassDataset`
   reads the 768 px cache, random-resized crop (0.7–1.0) + flip for training, centre resize for eval;
   `train_classifier` logs per-epoch train / val loss, accuracy and macro-F1 through `Run.log`, keeps the best val
   macro-F1 epoch, and returns the curves; `fit_verdict` reads them the PLAN §4 way (overfit: gap > 15 points and
   val falling after the best epoch; overfit-leaning: gap only; underfit: train acc < 0.6; else ok).
2. Modern recipe: AdamW, lr 3e-4, weight decay 0.05, label smoothing 0.1, cosine, 10 epochs, batch 32, 224 px, AMP,
   GPU memory capped at 80 % (6 GB card).
3. `scripts/baseline_cnn.py --phase P053 --arch resnet50 convnext_tiny --sources ccsn mgcd swimcat montenegro
   --seeds 0 1 2 --epochs 10`: 24 runs, 62 minutes, 151 s per run on average (343 s for the MGCD ConvNeXt runs).

## Verification
- `tests/test_train_cnn.py` (3, marked `slow`): the dataset returns the expected tensors, a two-epoch run on a tiny
  synthetic set improves on chance and returns curves, verdict rules on synthetic curves.
- Val macro-F1, mean ± std over seeds 0, 1, 2 (`docs/data/baseline_cnn.md`):

| Arch | CCSN (11) | MGCD (7) | SWIMCAT (5) | Montenegro (5) |
|---|---|---|---|---|
| ConvNeXt-T | **0.595 ± 0.010** | **0.793 ± 0.016** | 1.000 ± 0.000 | **0.695 ± 0.024** |
| ResNet-50 | 0.527 ± 0.009 | 0.768 ± 0.026 | 1.000 ± 0.000 | 0.599 ± 0.011 |

  Largest seed std 0.026 (ResNet-50 on MGCD); every std is below the gaps the paper will claim on.
- Fit verdicts: Montenegro and SWIMCAT ok for both; CCSN and MGCD ConvNeXt runs overfit or overfit-leaning
  (train–val gap 0.16–0.31, val falling after epoch 6–8); ResNet-50 on MGCD overfits (gap 0.20–0.23). The best epoch
  is taken, so the numbers stand, and the curves are the evidence that longer training would not help.
- Against the frozen probes (P054): ConvNeXt-T beats the best linear probe on Montenegro (0.695 vs 0.651) and is
  below it on CCSN (0.595 vs 0.646, DINOv3 ViT-L) and MGCD (0.793 vs 0.831, ViT-S @ 512). A frozen DINOv3 probe is
  therefore the stronger recognition reference on two of the three non-saturated sources.

## Exit criteria
- [x] Mean ± std over 3 seeds.

## Fit & data-risk notes
- CCSN has 1,776 training images for 11 genera: both nets memorise it (ConvNeXt train acc ≈ 0.9 at a val acc of
  0.62). The near-duplicate groups left inside the training split (P026: 12 % of CCSN) make memorising cheaper
  still; the grouped split keeps them out of val.
- SWIMCAT's 79 val patches (5 classes) are saturated by every method and carry no information for the comparison.

## Deviations from plan & why
- 10 epochs instead of 12 (the modern recipe's default) to fit the stage's GPU budget; the best epoch was 4–10, and
  the verdicts show the val curves had turned.

## Next phase
P054 — B3: frozen probes.
