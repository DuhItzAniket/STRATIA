# P017 — MGCD parser + weather

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Load MGCD images with their official split, class and the paired weather measurements.

## Inputs / dependencies
P007 registry; `DATA/MGCD/MGCD/{train,test}` with one spreadsheet per class and split.

## Work log
1. `stratia/data/mgcd.py`: image list from class folders (`<k>_<class>`), joined by file stem to the spreadsheet rows (temperature °C, relative humidity %, pressure hPa, wind m/s).
2. Test on a synthetic folder + spreadsheet (skipped in CI if `openpyxl` is absent).

## Verification
- 8,000 images; **weather joined for all 8,000** (0 missing).
- Per class (train / test): altocumulus 400/331, cirrus 650/673, clear sky 650/688, cumulonimbus 600/587, cumulus 690/748, mixed 510/510, stratocumulus 500/463.
- Weather ranges: temperature 5.1–41.5 °C (median 30.8), humidity 13.6–100% (median 54.6), pressure 998–1033.5 hPa, wind 0–6.5 m/s.
- In every class, all training image numbers precede all test numbers.

## Exit criteria
- [x] Join rate 100%.

## Fit & data-risk notes
- MGCD has **no timestamps**; the official split (lower numbers train, higher numbers test, per class) is consistent with a chronological split and is used as given. Near-duplicates across the boundary are checked in P025.
- Weather is used only for analysis; STRATIA's inputs stay image-only (ADR-003), so weather cannot become a shortcut.

## Deviations from plan & why
None.

## Next phase
P018 — Segmentation datasets loader.
