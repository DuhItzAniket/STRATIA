# P065 — Sky-parsing head

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
The dense sky-parsing output (invalid / sky / cloud / sun-glare at H/4) from a light DPT-style decoder on the
frozen DINOv3 tokens; trained on the segmentation datasets it must beat the P057 token probe and must not collapse
on clear-sky frames.

## Inputs / dependencies
P051 patch-token cache (`dinov3_vits16_512_ffr1-patches-seg`, 8,870 × 32 × 32 × 384); P018 masks on the grouped
splits (P057's tables, now in `stratia/baselines/segmentation.py`); P064 `dense_ce`; ADR-005 §2; P057 numbers.

## Work log
1. `stratia/model/dense.py`: `tokens_to_grid`, `FusionBlock` (3 × 3 conv, GroupNorm(8), GELU, twice, with a
   residual), `DenseDecoder` (LayerNorm → 1 × 1 to 128 → block → ×2 → 96 @ 64² → block → ×2 → 64 @ 128²),
   `SkyParseHead` / `LayerHead` (1 × 1), `DenseHeads` (decoder + both heads; 0.6 M parameters).
2. `scripts/train_dense_heads.py --phase P065 --epochs 15`: token grids read from the memmap, masks resized to
   512 (nearest), horizontal flip of grid and mask (p = 0.5), AdamW + one-cycle, AMP, loss on the logits upsampled
   to 512; per-dataset mIoU over the classes present (P057's rule), 95 % bootstrap intervals over val images, the
   clear-sky collapse check with the backbone run live on 200 Eye2Sky frames; `data/dense_heads.parquet`,
   `docs/data/dense_heads.md`. 13 minutes for 15 epochs (53 s per epoch, disk-bound on the 6.5 GB memmap).
3. The P057 helper functions (`seg_tables`, `load_sky_masks`, `clear_sky_frames`, `camera_masks`) moved into the
   library with a `load_both_masks` for P066; the baseline script imports them. DataLoader workers are off by
   default: Windows spawn pickles the mask lists to each worker and fails on the payload size.

## Verification
- `tests/test_dense.py` (2): contract shapes ([N, 4, H/4, W/4] and [N, 3, H/4, W/4]) and the token → grid layout;
  the decoder learns a per-patch sky / cloud pattern from positional tokens to > 90 % pixel accuracy.
- Val mIoU (grouped val, 891 images), decoder vs the P057 token probe on the same tokens:

| Dataset | Decoder mIoU [95 % CI] | IoU sky / cloud | Probe mIoU |
|---|---|---|---|
| all | **0.919** [0.916, 0.923] | 0.887 / 0.897 | 0.875 |
| SWIMSEG | 0.892 [0.875, 0.909] | 0.888 / 0.897 | 0.847 |
| SWINSEG | 0.911 [0.891, 0.928] | 0.917 / 0.904 | 0.866 |
| SWINySEG | 0.895 [0.889, 0.901] | 0.889 / 0.902 | 0.843 |
| SHWIMSEG | 0.833 [0.755, 0.890] | 0.807 / 0.858 | 0.786 |
| Almería | 0.902 [0.884, 0.918] | 0.883 / 0.847 (invalid 0.974) | 0.865 |

  Clear-sky cloud fraction on 200 Eye2Sky frames: **0.220** (probe 0.323, U-Net 0.237, prototype 85–99 %).
- The curve rises monotonically to epoch 15 (0.849 → 0.919); no train–val divergence (train loss 0.31 → 0.11).

## Exit criteria
- [x] Trains on the dataset(s); IoU > probe on every dataset and overall (+0.044 mIoU).

## Fit & data-risk notes
- The sun-glare class (index 3) has no labels yet; it is never predicted and never scored here. P074 adds targets
  from the augmentation's glare mask (P048) and from the P047 Sun pixel.
- 0.22 cloud on ceilometer-clear frames is not all error (horizon cloud, aureole); the number is the reference for
  the P074 monitor, not a target of 0.
- The CI on SHWIMSEG (125 images, 3 exposures of 52 scenes) is wide; it is reported, not interpreted on its own.

## Deviations from plan & why
- Trained on all five segmentation datasets at once rather than "one dataset": the probe it must beat was scored
  that way, and the grouped val keeps the comparison honest per dataset.

## Next phase
P066 — Layer head.
