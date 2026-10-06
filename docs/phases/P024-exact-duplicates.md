# P024 — Exact duplicates

Status: DONE     Date: 2026-10-06     Commit: (this commit)

## Objective
Find every exact duplicate within and across the datasets, by file bytes and by decoded pixels, and fix the policy
that keeps duplicates from leaking across splits.

## Inputs / dependencies
P021 manifest; P023 (all images decode).

## Work log
1. `stratia/data/duplicates.py`: `hash_image()` (SHA-256 of the file and of the decoded RGB pixels with their shape),
   `duplicate_groups()` (groups keyed by pixel hash, or by file hash for an undecodable file), `pair_summary()`,
   `report_markdown()`.
2. `scripts/find_duplicates.py`: process pool over the manifest; writes `data/image_hashes.parquet`,
   `data/duplicate_groups.parquet`, `docs/data/duplicates_report.md`.
3. Tests: byte-identical copy, re-saved copy with identical pixels, a different picture, an undecodable file;
   the empty case.

## Verification
- Run: `python scripts/find_duplicates.py --workers 12`: 52,032 images hashed twice in 329 s, 0 decoding errors.
  Results `data/image_hashes.parquet`, `data/duplicate_groups.parquet`; report `docs/data/duplicates_report.md`.
- **661 images (1.3 %) have an exact duplicate, in 312 groups (279 pairs, 29 triples, 4 groups of four). None of
  the groups spans two datasets.**

| Dataset | Images in a group | Groups | Of which byte-identical | Note |
|---|---|---|---|---|
| SWINySEG | 533 (7.9 % of 6,768) | 248 | 533 | identical files under different names |
| SWIMSEG | 52 (5.1 % of 1,013) | 26 | 6 | 46 are re-saved copies: same pixels, different bytes |
| CCSN | 34 (1.3 % of 2,543) | 17 | 34 | **3 groups carry two different labels** (Cc/Cs, Ac/As, Ac/As) |
| SWIMCAT | 34 (4.3 % of 784) | 17 | 28 | |
| SHWIMSEG | 6 | 3 | 6 | |
| Almería | 2 | 1 | 2 | |
| Eye2Sky, MGCD, Montenegro, SWINSEG, B0268 | 0 | 0 | 0 | |

- The three CCSN groups with two labels are the first label conflicts of the audit; they go to P026 as the plan says.
- Tests: in `tests/test_integrity_duplicates.py` (byte copy, pixel copy, distinct, undecodable, empty case).

## Exit criteria
- [x] Duplicate table: `data/duplicate_groups.parquet` (one row per image that has a duplicate; group id, size,
  datasets spanned, bytes-identical or pixels-only).
- [x] Resolved policy: written in `docs/data/duplicates_report.md` and `docs/research_log.md`: one `group_id` per
  duplicate group at split time (P038); cross-dataset groups kept and reported in the data card; label conflicts to
  P026.

## Fit & data-risk notes
- **Within-dataset duplicates are leakage in the published protocols of these datasets**: SWINySEG's own random
  split and SWIMCAT's cross-validation both let identical images fall on both sides. Our split generator (P038)
  must merge duplicate groups into one `group_id`; with that, the 661 images are harmless.
- **No exact duplicate crosses datasets**, so the usual suspicion (SWINySEG re-releasing SWIMSEG/SWINSEG frames)
  is not an exact-copy relation: if those datasets overlap, they do so through resizing, augmentation or
  re-encoding, which only the near-duplicate search (P025) can show.
- 46 of the 52 SWIMSEG duplicates differ in bytes but not in pixels: a byte hash alone would have missed 88 % of
  that dataset's duplicates. The pixel hash earned its place.
- Duplicates with different labels (CCSN) are label noise with a known cause; counted once, labelled by whichever
  policy P026 decides, never auto-corrected.

## Deviations from plan & why
- The plan says "SHA-256"; a second hash of the decoded pixels was added, because a re-saved copy is the same
  picture for a model and a different file for SHA-256.

## Next phase
P025: Near-duplicates (perceptual hashes and DINOv3 embeddings).
