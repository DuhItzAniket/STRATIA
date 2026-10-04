# P011 — Inventory local data

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
A per-file inventory of everything under the data root: owning dataset, format, size, resolution, colour mode, junk and unreadable files.

## Inputs / dependencies
P007 registry; `configs/paths.yaml`.

## Work log
1. `scripts/inventory.py`: walks `data_root` with 16 threads, reads image headers only (no pixel decoding), assigns each file to the most specific registered folder, flags junk (`__MACOSX`, `._*`, zero-byte), records unreadable images; writes `data/inventory.parquet` (derived, git-ignored) and `docs/data/inventory.md` (committed summary).
2. Tests (`tests/test_inventory.py`): dataset assignment, nested registrations, junk and unreadable-file handling.

## Verification
Full run in 35 s: 64,406 files, 15.0 GB, **0 unreadable images**. Summary (`docs/data/inventory.md`):

| Dataset | Valid images | Main resolution(s) | Notes |
|---|---|---|---|
| CCSN | 2,543 | 400×400 (2,332), **256×256 (211)** | Mixed resolutions |
| MGCD | 8,000 | 1024×1024 | + 14 xlsx weather files |
| Montenegro | 2,522 | **640×480** | Low-resolution camera |
| SWIMCAT | 784 | 125×125 | Patches |
| SWIMSEG family | 16,208 (8,052 images + masks) | 300×300, 600×600, 500×500 | Masks in mode L |
| Almería | 1,636 (818 images + masks) | 512×512 | |
| Eye2Sky | 29,628 | 2112×2048 (29,327) | + 258 NetCDF, 39 calibration YAML, 39 mask files |
| B0268 | 25 | 4656×3496 | |
| Unregistered | 0 | — | 2,555 macOS junk files (`CCSN/__MACOSX`) |

## Exit criteria
- [x] Report matches the known counts (CCSN 2,543; MGCD 8,000; Montenegro 2,522; Almería 818; SWIMCAT 784).

## Fit & data-risk notes
- **Shortcut risk:** CCSN mixes 400×400 and 256×256 images; according to the CloudScope prototype report the 256×256 images belong to only three classes, so image size alone can reveal the label. P029 must confirm and neutralise this (uniform resizing before any feature extraction; resolution-randomising augmentation).
- Resolutions differ by more than an order of magnitude across datasets (125 px to 4656 px): a dataset-identity shortcut is likely and is measured in P029.

## Deviations from plan & why
None.

## Next phase
P012 — Eye2Sky inventory.
