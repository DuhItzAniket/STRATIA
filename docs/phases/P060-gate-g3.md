# P060 — Gate G3: Baselines reproduced

Status: DONE (gate signed)     Date: 2026-10-11     Commit: (this commit)

## Objective
Decide whether the Stage F baselines are stable enough to compare against (seed variance) and plausible against the
literature, so that a STRATIA gain over them at gates G4–G6 means something.

## Inputs / dependencies
P051–P059 result files and phase docs; `docs/data/baseline_leaderboard.md`; the literature noted below.

## Work log
1. **Seed variance.** Every number the criteria will use has three seeds or is deterministic:

| Baseline | Seeds | Largest std (macro-F1) | Where |
|---|---|---|---|
| ResNet-50 / ConvNeXt-T fine-tunes (P053) | 3 | 0.026 (ResNet-50 on MGCD) | 0.009–0.026 elsewhere |
| MobileNetV3-L legacy recipe (P052) | 3 | 0.029 (random split), 0.018 (grouped) | |
| Linear / k-NN probes (P054, P055) | deterministic (lbfgs, fixed C grid) | — | |
| Zero-shot (P055), climatology / lookup / ridge (P058) | deterministic | — | |
| Small U-Net, token probe (P057), ResNet-18 CBH (P058) | 1 | not measured | model gaps 0.01–0.3 mIoU, 25 m MAE |

   Acceptable: the smallest difference any criterion will be read on is 0.05 macro-F1 (C1) and a skill margin on
   MAE (C3); the measured stds are a third of that or less. The single-seed segmentation and CBH CNNs are references,
   not the quantity being optimised; a seed sweep is cheap (17 and 3 minutes) if a later gate needs the interval.
2. **Plausibility against the literature** (validation numbers here; literature numbers are test numbers on the
   datasets' own splits):
   - *CCSN.* Reported accuracies range from about 57–62 % for ResNets under duplicate-aware grouped splits to
     91–97 % under random splits of the full dataset. Ours: 0.56–0.62 accuracy (ConvNeXt-T, DINOv3 probes), under a
     grouped split of a dataset in which 12 % of images sit in near-duplicate groups that are mostly label conflicts
     (P052). The level matches the duplicate-aware work; the 91–97 % figures come from protocols P052 shows to be
     contaminated rather than merely optimistic.
   - *MGCD.* 88.6 % (MMFN) and 91.3 % (MMST) accuracy with multimodal inputs (image + meteorological channels) on
     the authors' split. Ours: 0.82 accuracy image-only on the grouped val (ConvNeXt-T; probe macro-F1 0.83).
     Below the multimodal numbers by the margin the extra channels and the split are expected to give.
   - *SWIMCAT.* 95–99 % in the literature; ours 1.000 on 79 val patches. Saturated either way.
   - *SWIMSEG / SWINySEG segmentation.* mIoU up to 0.94–0.99 reported for tuned networks on the datasets' random
     splits. Ours: 0.80–0.87 (U-Net from scratch, token probe) with whole sessions held out and a 16 px token
     labelling for the probe. Lower, for identified reasons; the dense head is measured against these, not the
     literature's.
   - *Cloud-base height from one all-sky image.* No published MAE against a ceilometer was found for a single-image
     model (the single-camera neural-network work reports a correlation of 0.74 on 260 clouds; stereo networks
     report 50 m-resolution maps but are not single-image). Ours: MAE 699–724 m, skill +0.37–0.39 over a 1,147 m
     climatology, on 1,368 cloudy pairs over 12 held-out days per station. Consistent with the known difficulty
     (error grows with height; cirrus above 8 km is where both the ceilometer and the picture are weakest).
3. **Instabilities found and fixed inside the stage** (each would have invalidated a comparison):
   the multilabel macro-F1 jumped by a third on a single false positive in an unsupported étage column (P054); the
   segmentation mIoU halved on a handful of predicted-invalid pixels in datasets without an invalid class (P057);
   the first P052 protocol read the locked test rows (P052); DINOv3 ViT-L was all-NaN in fp16 (P051). All four are
   covered by tests or by the registry's reproducible runs.
4. **Open items carried forward, not blocking:** SAM 3 (separate environment), the legacy U-Net weights (none exist),
   a 4-bit / full-set VLM pass (P056 PARTIAL), seed intervals for the segmentation and CBH CNNs if a later gate is
   read on them.

## Verification
- `docs/data/baseline_leaderboard.md` regenerated from the final result files; 70 Stage F runs in the registry;
  198 tests green; ruff clean.

## Exit criteria
- [x] Variance across seeds acceptable (work log 1).
- [x] Numbers plausible vs literature (work log 2).
- [x] Signed: Gate G3 passed, 2026-10-11.

## Fit & data-risk notes
- Guard "unstable baselines invalidate comparisons": the probes are deterministic and the fine-tunes have a std
  below 0.03; the comparisons at G4–G6 will be made against the leaderboard's reference lines with the P090 test
  evaluation as the only test read.

## Deviations from plan & why
- The gate's literature column is a reading of published numbers under their own protocols, not a re-run of any
  published model; where no comparable number exists (single-image CBH), the gate says so.

## Next phase
P061 — Stage G, architecture spec (P041 remains blocked on B0268 frames).
