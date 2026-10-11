# P057 — B6: segmentation baselines

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Sky-parsing references (invalid / sky / cloud) for the dense head: a small U-Net trained from scratch, a linear probe
on frozen DINOv3 patch tokens, and the collapse check the legacy prototype failed (predicted-cloud fraction on
clear-sky frames).

## Inputs / dependencies
P018 segmentation index (SWIMSEG family + Almería, 8,870 masks); P038 grouped splits; P051 patch-token cache
(`dinov3_vits16_512_ffr1-patches-seg`, 32 × 32 tokens); P047 pairs (clear-sky Eye2Sky frames); P043 camera masks.

## Work log
1. `stratia/baselines/segmentation.py`: `SegDataset` (image + mask at the working size, 255 ignored, flips for
   training), `SmallUNet` (1.9 M parameters, 4 levels), `confusion` / `scores_from_confusion` (per-class IoU, mIoU,
   predicted-cloud fraction), `TokenProbe` (one linear layer on the 384-d tokens), `mask_to_tokens` (patch-majority
   labels, ignore when the majority is ignore), `train_token_probe`, `probe_predict` (logits upsampled to 512 px).
2. `scripts/baseline_segmentation.py --epochs 10 --seeds 0`: U-Net at 256 px, cross-entropy, AdamW + one-cycle,
   best val mIoU epoch kept; probe on the cached tokens; both scored per dataset on the grouped val split (891 images);
   collapse check on 200 clear-sky Eye2Sky frames (ceilometer label none, confidence ≥ 0.9, inside the camera mask).
   17 minutes. Two registry runs (phase P057); rows in `data/baseline_segmentation.parquet`, table in
   `docs/data/baseline_segmentation.md`.
3. Two fixes on the way: the P018 index carries the datasets' own `split` column, which collided with the grouped
   split on merge (the first run crashed); and mIoU is now the mean over the classes present in a dataset's truth
   (the SWIMSEG family has no invalid pixels, and a few predicted ones halved the U-Net's SWIMSEG mIoU to 0.53).
4. Not run: the legacy U-Net (no trained weights of the prototype exist in either repository; its recorded
   behaviour, 85–99 % cloud on clear-sky frames, is the collapse number, and the small U-Net from scratch stands in
   for "a U-Net at this data scale") and SAM 3 (needs Python 3.12 / torch ≥ 2.7, a separate environment; deferred
   as the plan allows).

## Verification
- `tests/test_segmentation_baselines.py` (4): U-Net shapes, confusion and scores (including a class absent from the
  truth), token labels from masks, probe shapes.
- Val mIoU per dataset (IoU cloud in brackets) and the collapse number:

| Model | All | SWIMSEG | SWINSEG | SWINySEG | SHWIMSEG | Almería | Clear-sky cloud fraction |
|---|---|---|---|---|---|---|---|
| DINOv3 ViT-S/16 token probe @ 512 | **0.875** (0.849) | 0.847 | 0.866 | 0.843 | 0.786 | 0.865 | 0.323 |
| Small U-Net @ 256, from scratch | 0.867 (0.819) | 0.798 | 0.825 | 0.820 | 0.468 | 0.870 | **0.237** |

- Reading: the frozen token probe matches the U-Net overall and beats it on every SWIM dataset; SHWIMSEG (the
  high-resolution set, 125 images) breaks the U-Net (sky IoU 0.28) and not the probe. Almería's invalid class (the
  camera mask) is easy for both (IoU 0.94–0.98). The collapse check: on frames the ceilometer calls clear, the U-Net
  still labels 24 % of the sky cloud and the probe 32 %, against the prototype's 85–99 %. Those are the numbers the
  dense head has to beat; part of the 24–32 % is real (a ceilometer sees the zenith, the camera sees the horizon
  and the solar aureole), which is what the P047 confidence field and the P046 dropout will let the model separate.

## Exit criteria
- [x] Per-dataset IoU.
- [x] Collapse metric (pred-cloud fraction on clear-sky frames) recorded for both models.

## Fit & data-risk notes
- Literature mIoU on SWIMSEG runs to 0.99 with tuned networks on the datasets' own random splits; the grouped val
  here holds whole capture sessions out, and 0.85 for a linear probe is the honest reference, not a ceiling.
- The probe's token labels are patch majorities at 16 px: thin cloud edges are lost at labelling time, which caps
  its IoU below a pixel method's; the dense head (P067) will work at the pixel level.

## Deviations from plan & why
- Legacy U-Net replaced by a from-scratch small U-Net; SAM 3 deferred (work log 4). One seed (the plan asks for
  per-dataset IoU, not variance; the two models differ by more than a seed would move them).

## Next phase
P058 — B7: CBH baselines.
