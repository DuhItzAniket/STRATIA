# P059 — Baseline leaderboard

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
One generated page with every Stage F number the paper and the gates will refer to: recognition in-domain and
leave-one-dataset-out, the protocol gap, segmentation, cloud-base height, and the "best baseline" lines that
criteria C1 and C3 compare STRATIA against.

## Inputs / dependencies
`data/baseline_probes.parquet` (P054, P055), `data/baseline_cnn.parquet` (P052, P053), `data/baseline_zeroshot.parquet`
(P055), `data/baseline_segmentation.parquet` (P057), `data/baseline_cbh.parquet` (P058), `runs/registry.csv`.

## Work log
1. `scripts/baseline_leaderboard.py` → `docs/data/baseline_leaderboard.md`. Every method becomes a row keyed by
   (method, protocol, source); the in-domain and LODO tables are sorted by the mean over sources, and a method that
   lacks a source has no mean (it sorts last) rather than a mean over fewer columns. The protocol-gap table carries
   the P052 random-vs-grouped numbers with the leaked / clean decomposition, and the best probe per source for
   in-domain and LODO. The segmentation and CBH tables are copied from their result files. The page ends with the
   registry count per phase (70 Stage F runs).
2. Reference lines: **C1** = best LODO baseline, linear probe on DINOv3 ViT-B/16 @ 224, mean étage macro-F1 0.583
   over the CCSN, MGCD, Montenegro and Almería folds; **C3** = ceilometer-supervised ResNet-18, MAE 724 m, skill
   +0.368 (the ridge on frozen features is better at 699 m and is listed next to it; C3's wording names the
   supervised CNN, and both are on the page).

## Verification
- Regenerated three times during the stage (after the probe metric fix, the P052 protocol change and the
  segmentation rerun); the page is a pure function of the result files, which the registry ties to run ids.
- Top of the in-domain table: ConvNeXt-T fine-tuned 0.771 mean, DINOv3 ViT-B probe 0.765, ViT-L probe 0.763,
  ViT-S @ 512 probe 0.754; zero-shot CLIP 0.413 at the bottom. Top of LODO: DINOv3 ViT-B probe 0.583, DINOv2-B
  0.563, SigLIP 2 0.529.

## Exit criteria
- [x] Auto-generated from the run registry and result files.

## Fit & data-risk notes
- All numbers are validation numbers; the locked test sets are evaluated once at P090 and the page will then gain a
  test column next to each val column, never replacing it.
- The LODO MGCD fold is scored on low and high only (no mid-étage val rows, P054), and SWIMCAT is saturated: both
  columns are kept for completeness and are not where differences between methods will be read.

## Deviations from plan & why
- None.

## Next phase
P060 — Gate G3.
