# P088 — Robustness suite

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
How the Stage H candidates degrade under the corruptions a sky camera meets: glare, haze, a dirty lens, JPEG,
exposure shifts, sensor noise and blur; a table per head, run with the backbone live on corrupted validation images.

## Inputs / dependencies
P048 photometric functions (exposure, dirt and drops, sun glare, JPEG, sensor noise); the Stage H candidates
(P073 global heads, P074 dense heads with a saved checkpoint, P075 CBH head); P047 ray maps (the Sun's pixel for the
glare corruption on Eye2Sky frames); P022 image cache.

## Work log
1. `scripts/robustness_suite.py`: nine conditions (clean, exposure × 0.6 and × 1.5, haze 40 %, dirty lens, sun
   glare at the Sun's pixel when the ray map is known, JPEG quality 30, sensor noise read 0.03 / shot 0.06, Gaussian
   blur σ = 2 px) applied to the cached 768 px image before the full-frame 512 px resize; the DINOv3 backbone runs
   on every image; the heads are scored as in Stage H on subsets of the validation sets (1,200 recognition rows,
   400 segmentation images, 100 clear-sky frames, 800 CBH pairs; 18 minutes). `docs/data/robustness.md`,
   `data/robustness.parquet`.
2. The dense trainer now saves its best checkpoint; the P074 run was repeated for it (val mIoU 0.904, layer
   0.559, clear-sky cloud fraction 0.290: the same run to within 0.001 / 0.009).

## Verification
Change against clean (the clean row is the subset's own number):

| Corruption | Mean étage F1 | Genus F1 | Sky mIoU | Clear-sky cloud | CBH MAE (m) | CBH skill |
|---|---|---|---|---|---|---|
| clean | 0.838 | 0.703 | 0.901 | 0.292 | 606 | +0.446 |
| exposure × 0.6 / × 1.5 | −0.008 / −0.016 | +0.004 / −0.010 | −0.002 / −0.001 | +0.014 / +0.019 | +6 / −4 | −0.005 / +0.004 |
| haze 40 % | −0.022 | +0.010 | −0.008 | +0.016 | +58 | −0.053 |
| dirty lens | −0.034 | −0.068 | −0.008 | +0.016 | +2 | −0.002 |
| sun glare | **−0.078** | −0.048 | **−0.085** | **+0.174** | +140 | −0.128 |
| JPEG q30 | −0.026 | +0.025 | −0.007 | −0.086 | +174 | −0.159 |
| sensor noise | **−0.098** | −0.026 | −0.053 | −0.034 | **+408** | **−0.373** |
| blur 2 px | −0.032 | −0.024 | −0.006 | −0.046 | +137 | −0.125 |

- Reading: exposure shifts cost nothing (the features are exposure-invariant enough), haze and a dirty lens cost
  little except genus on a dirty lens (−0.07). **Sun glare** is the dense head's weak spot: the bloom is called
  cloud (clear-sky cloud fraction 0.29 → 0.47) and sky parsing loses 0.085 mIoU; the glare class exists in the
  contract but has no labels yet, which is exactly the P074 / P083 item. **Sensor noise, JPEG and blur** hit the
  CBH head hardest (+140 to +408 m): the height cue lives in fine cloud texture at the pooled token grid, and
  anything that removes texture removes it. The global heads lose 0.02–0.10 étage F1 across the board, most under
  noise.
- These are the corruptions P048 trains against in the image pipeline (P077's LoRA stage saw them at strength
  1.0 and 0.5); the frozen-feature heads never did, which is the first explanation of the CBH fragility.

## Exit criteria
- [x] Corruption-robustness table.

## Fit & data-risk notes
- Subsets of val, one seed of each corruption's randomness: the table's resolution is about ±0.01 on the global
  numbers and ±20 m on CBH; the big effects (glare on dense, noise / JPEG / blur on CBH) are well above it.
- The glare corruption on recognition images has no Sun position and lands at a random place; on the Eye2Sky
  frames it lands on the Sun's pixel from the ray map, which is the realistic case.

## Deviations from plan & why
- Run out of order (P083–P087 wait for the B0268 frames); it needs nothing from them.

## Next phase
P090 — Locked-test evaluation (once), when the B0268 column is decided; P083 when the frames exist.
