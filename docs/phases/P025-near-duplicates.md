# P025 — Near-duplicates

Status: DONE     Date: 2026-10-06     Commit: (this commit)

## Objective
Find images that are the same picture changed a little (re-sized, re-encoded, flipped, rotated, cropped) within
and across datasets, choose the threshold from a review of contact sheets, and turn the clusters into groups that
the split generator keeps together.

## Inputs / dependencies
P021 manifest; P022 image cache (`cache/img768`); P024 exact duplicates (the floor: cosine 1.0, pHash 0);
DINOv3 ViT-S/16 weights (Hugging Face, gated access verified in P010).

## Work log
1. `stratia/data/near_duplicates.py`: dHash and pHash (64 bits each) computed for all eight flips and rotations
   of an image, so an augmented copy matches its source; DINOv3 CLS embeddings (224 px, ImageNet normalisation,
   L2-normalised); k-nearest-neighbour search by cosine on the GPU in chunks; candidate pairs; union-find grouping;
   contact sheets.
2. `scripts/find_near_duplicates.py`: embeddings cached to `cache/features/dinov3_vits16_224_cls.npy` (reused by
   the shortcut audit, P029), hashes to `data/perceptual_hashes.parquet`, pairs to `data/near_duplicate_pairs.parquet`,
   groups to `data/near_duplicate_groups.parquet`, report `docs/data/near_duplicates_report.md`, contact sheets in
   `docs/data/figures/`.
3. Tests (`tests/test_near_duplicates.py`, 9): hash stability and discrimination, dihedral matching of a
   rotated-and-flipped copy, hashing from a file and a bad file, nearest neighbours and pair listing, connected
   components, ImageNet normalisation, contact sheet geometry.
4. Reviewed the contact sheets and chose the thresholds (Verification).

## Verification
- Runs: `python scripts/find_near_duplicates.py` (embeddings 141 s on the RTX 4050 for 52,032 images at 224 px,
  ViT-S/16, fp16; hashes 590 s on 12 processes; neighbour search 1 s). Outputs: `cache/features/dinov3_vits16_224_cls.npy`,
  `data/perceptual_hashes.parquet`, `data/near_duplicate_pairs.parquet` (303,925 candidate pairs with cosine >= 0.90),
  `data/near_duplicate_groups.parquet`, `data/swinyseg_sources.parquet`; report `docs/data/near_duplicates_report.md`;
  contact sheets `docs/data/figures/near_dup_*.jpg`.
- **Threshold review** (contact sheets, 16 pairs each). Band cosine 0.97–1.00 within a dataset: every pair shown is
  the same scene (consecutive MGCD frames seconds apart, SWIM patches of one capture). pHash <= 10 with cosine < 0.97
  (the first rule tried): Montenegro frames of *different* nights and uniform skies matched spuriously, so a loose
  pHash rule is wrong for smooth sky images; pHash 0–2 pairs were real copies, flipped or rotated. Cross-dataset:
  SWIMSEG/SWINySEG pairs at pHash 0 are identical patches, often mirrored; pairs at pHash 24–28 are merely
  similar skies. **Rule chosen:** a pair is a *copy* if its dihedral pHash distance is <= 2; it shows the *same
  scene* if its cosine is >= 0.97; either makes it a near-duplicate. The first rule tried (cosine >= 0.97 or
  pHash <= 10) is reported in the log as the rejected alternative.
- **Result:** 45,999 near-duplicate pairs (16,459 copies, 42,025 same-scene; 2,229 cross-dataset); 14,112 images in
  1,656 groups (largest 448). Same-station Eye2Sky pairs (196,767 of the candidates) are excluded from the groups.

| Dataset | Images in a group | Share | Groups | Largest | What it is |
|---|---|---|---|---|---|
| MGCD | 7,775 | 97.2 % | 227 | 291 | consecutive frames of one camera: 8,000 images are a few hundred scenes |
| SWIMSEG | 984 | 97.1 % | 719 | 9 | patches of the same capture, and copies in SWINySEG |
| SWINSEG | 109 | 94.8 % | — | — | every image has a copy in SWINySEG |
| SWINySEG | 3,065 | 45.3 % | 1,014 | 30 | copies of SWIMSEG/SWINSEG patches plus internal exact duplicates (P024) |
| Almería | 356 | 43.5 % | 86 | 117 | consecutive frames |
| Montenegro | 883 | 35.0 % | 106 | 248 | 20-minute cadence frames of similar skies |
| SHWIMSEG | 42 | 26.9 % | 18 | 4 | |
| SWIMCAT | 132 | 16.8 % | 51 | 13 | |
| CCSN | 310 | 12.2 % | 149 | 4 | re-posted photographs (several with different labels, P026) |
| Eye2Sky | 456 | 1.6 % | 4 | 448 | AURIC and BARSE at the same moments under similar overcast skies |
| B0268 | 0 | | | | |

- **Exhaustive check of the SWIM family** (every SWINySEG × source pair by dihedral pHash, no embedding involved):
  **2,342 of the 6,768 SWINySEG images (34.6 %) are flipped, rotated or re-encoded copies of a SWIMSEG (2,096) or
  SWINSEG (246) image.** Exact hashing (P024) found none of these: a byte or pixel hash cannot see a flip.
- Tests: 9 in `tests/test_near_duplicates.py`; `python -m pytest -q`: all pass; `ruff check .` clean.

## Exit criteria
- [x] Threshold chosen from review: Verification states the rule and what each band of the contact sheets showed.
- [x] Clusters become `group_id`s: `data/near_duplicate_groups.parquet` (image → group), applied together with the
  exact-duplicate groups when the splits are generated (P038).

## Fit & data-risk notes
- **Cross-dataset leakage, confirmed:** training on SWIMSEG or SWINSEG and testing on SWINySEG (or any mix of
  them) tests on copies of the training images. The LODO protocol (P037) must treat the SWIM family as one source,
  and the data card must say so.
- **Time-series datasets have far fewer independent samples than images:** MGCD's 8,000 images form 227
  same-scene groups plus about 225 singletons; Montenegro and Almería are similar. Random splits of these datasets
  (the published protocols) leak heavily; our splits go by group and, where time is known, by temporal blocks
  (P027, P037). Reported accuracies on MGCD in the literature should be read with this in mind.
- **DINOv3 cosine is not rotation-invariant and pHash is not sky-invariant**: each tier catches what the other
  misses (flipped copies at cosine 0.95–0.97; consecutive overcast frames at pHash 10–18). Neither alone would do.
- **Limits of the candidate set:** pairs were taken from the 10 nearest embedding neighbours with cosine >= 0.90,
  so a copy whose embedding fell below 0.90 (an extreme crop or rotation) is missed unless the exhaustive pHash
  check covered it, which it did only for the SWIM family. Rotated copies in other datasets would need the same
  exhaustive pass; the cost is small and it can be added when a dataset's construction suggests it.
- The Eye2Sky group of 448 images joins AURIC and BARSE frames at the same moments (15 km apart): not copies, but
  evidence that overcast skies look the same across stations; cross-station splits (P037) must expect it.

## Deviations from plan & why
- Same-station Eye2Sky pairs are excluded from the groups: consecutive all-sky frames 30 s apart are near-identical
  by nature, and grouping them would fuse whole days into one cluster. They are a time-series question and are
  handled by temporal blocking (P027 measures the autocorrelation, P037 sets the split gaps).
- Hashes are computed over all eight dihedral variants (beyond the plan's "pHash/dHash"), because SWINySEG is an
  augmented construction and a plain hash would miss its flipped and rotated copies.

## Next phase
P026: Label-conflict audit.
