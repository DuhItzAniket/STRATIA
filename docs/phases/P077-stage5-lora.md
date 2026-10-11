# P077 — Stage-5: partial unfreeze

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Put the backbone in the graph: LoRA adapters on the last blocks of DINOv3 ViT-S/16 with the heads from the joint
checkpoint, trained on images with the P048 augmentation, and measure the gain over the frozen model on the same
validation images and the forgetting on the datasets the adaptation does not target.

## Inputs / dependencies
P076 joint checkpoint; P048 augmentation policy; P022 image cache; peft 0.21 (LoRA), transformers DINOv3 with
gradient checkpointing; P010 (batch 16 at 512 px).

## Work log
1. `scripts/train_lora.py`: `LoraBackbone` (peft LoRA on the q / k / v projections of the last N blocks, gradient
   checkpointing, layer-wise learning-rate decay 0.8 per block), `LoraModel` (backbone + the P071 heads),
   image datasets for the global rows and the dense rows (masks warped with the image through the geometric
   chain, encoded as one label map), the P069 sampler over two groups (global 16 images, dense 8 images with
   masks), bfloat16 autocast, the P072 engine; evaluation on the val images at the full-frame 512 px resize,
   **before** training with the adapters disabled (the frozen reference through this pipeline) and after every
   epoch. Ray maps are the null state throughout (the uncalibrated sources); geometry-aware fine-tuning on Eye2Sky
   is Stage I's.
2. Runs (rank 8, α 16, last 4 blocks = 12 projections, 73,728 trainable backbone parameters): `lora` (heads lr 3e-4,
   augmentation strength 1.0, 4 epochs of 500 batches, 64 min) and `lora_lowhead` (heads lr 1e-4, strength 0.5,
   3 epochs, 43 min). `docs/data/stage5_lora.md`.

## Verification
Validation, frozen reference → best LoRA epoch:

| Metric | Frozen | `lora` (epoch 2) | `lora_lowhead` (epoch 1) |
|---|---|---|---|
| Global mean étage macro-F1 | 0.805 | 0.810 | 0.810 |
| Dense sky mIoU (all) | 0.869 | **0.878** | 0.877 |
| Almería layer mIoU | 0.517 | 0.471 | 0.509 |
| CCSN genus macro-F1 | 0.644 | 0.599 | 0.624 |
| MGCD genus macro-F1 | 0.786 | 0.754 | **0.812** |
| CCSN étage F1 | 0.784 | 0.736 | 0.773 |
| Montenegro étage F1 | 0.638 | **0.719** | 0.671 |
| Montenegro C2 ratios (étage / oktas) | 1.018 / 0.980 | 1.047 / 0.991 | 1.038 / 1.004 |

- Gain over frozen: +0.009 sky mIoU, +0.005 mean étage, +0.03 to +0.08 on Montenegro's étage; the headline moves
  from 0.837 to 0.844 / 0.843.
- Forgetting: the first run loses 0.03–0.05 on CCSN and MGCD genus and 0.05 on the layer head; with the lower head
  learning rate and half the augmentation strength the losses shrink to −0.02 on CCSN genus (inside the ±0.05
  interval of 255 images) and −0.008 on layers, while MGCD genus gains 0.026. The heads, trained on clean cached
  features, are what drifts under augmented inputs; the adapters themselves are small (0.3 % of the backbone).
- Verdict: at this budget LoRA on the last four blocks is a small, unsystematic improvement; it does not change
  which heads win or lose against the baselines. The frozen-feature candidates remain the Stage H reference; the
  LoRA path is ready for Stage I, where the adaptation target (B0268) is a real distribution shift.

## Exit criteria
- [x] Gain over frozen on val measured (small); forgetting measured and, with the low head learning rate, within
  the validation intervals.

## Fit & data-risk notes
- One epoch of 500 batches is 8,000 global images and 4,000 dense images: the val curves peak at epoch 1–2 and
  drift down after, the usual sign of a small-data fine-tune; more epochs would need more regularisation, not
  more steps.
- The "before" reference (0.805 étage, 0.869 mIoU) is below the cached-feature numbers (0.817 / 0.905) because the
  joint checkpoint, not the single-task heads, initialises this stage and because augmentation-free val images go
  through the live backbone; the comparison inside the table is like for like.

## Deviations from plan & why
- Rank 8 only (the plan says 8–16): the second run spent its budget on the head learning rate instead, which the
  first run showed to be the forgetting lever. No geometry in this stage (null ray maps), as stated in the script.

## Next phase
P078 — Budgeted HPO.
