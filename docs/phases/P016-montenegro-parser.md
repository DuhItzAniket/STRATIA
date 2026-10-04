# P016 — Montenegro parser

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Turn the 11,191 independent expert annotations of the Montenegro dataset into per-image soft labels, exactly as the dataset README prescribes.

## Inputs / dependencies
P007 registry; `DATA/Montenegro` (README sections 4 and 7).

## Work log
1. `stratia/data/montenegro.py`: coded columns read as strings (`/` is a valid "cannot be determined" code); empty cells become missing (distinct from `/`); `Low` merged into `Low clouds` (README §7.1); per-image soft labels for N, Nh, h, CL, CM, CH and altitude class: answer count, code distribution (JSON), majority code(s) with ties kept explicit, majority share.
2. **Image-name mismatch found:** the archive's files are named `<item_id>_snapshot_...jpg`, while the CSVs (and README) list `snapshot_...jpg`; `image_file()` accepts both.
3. `scripts/montenegro_labels.py`: builds `data/montenegro_soft_labels.parquet` (ignored) and `docs/data/montenegro_labels.md`, and checks every count published in the README.
4. Tests on a synthetic file: string codes, `Low` merge, `/` vs missing, soft labels, explicit ties, both naming styles.

## Verification
All README checks pass (exit code 0):

| Check | Parsed = README |
|---|---|
| Annotations / images / annotators | 11,191 / 2,522 / 9 |
| Raters per image | {1: 252, 2: 23, 4: 781, 5: 1,277, 6: 11, 7: 108, 8: 68, 9: 2} |
| Missing answers | CM 59, CH 56, CL 9, Nh 5, altitude 1, N 0, h 0 |
| Image files found | 2,522 / 2,522 |
| Duplicate (image, annotator) pairs | 0 |

Mean majority share on images with ≥ 4 raters: altitude class 0.743, N 0.691, Nh 0.595, **h (cloud-base height) 0.580**, CL 0.689, CM 0.659, CH 0.710.

## Exit criteria
- [x] Per-image soft-label table; counts match the README.

## Fit & data-risk notes
- Experts disagree most on cloud-base height (`h`, majority share 0.58) and low-cloud amount: the human ceiling for these targets is low, so STRATIA must be compared with expert–expert agreement (P035), not with a single "true" label.
- Splits must be by day or multi-day block (README §7.6): consecutive frames are 20 minutes apart.
- Timestamps are local time stated as UTC+1; whether October 2025 frames (before the end of daylight saving on 26 Oct) are UTC+2 is not confirmed. The site coordinates are not published, so Sun-based checks are not possible for this dataset.

## Deviations from plan & why
None.

## Next phase
P017 — MGCD parser + weather.
