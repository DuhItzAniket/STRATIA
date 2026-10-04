# P012 — Eye2Sky inventory

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Know exactly which Eye2Sky images, calibrations and ceilometer days are on disk, and how many images can be paired with a ceilometer for cloud-base-height (CBH) training.

## Inputs / dependencies
P007 registry; Eye2Sky station list, `asi_meta/`, `ceilometer/`, `2022/` image tree.

## Work log
1. `scripts/eye2sky_inventory.py` (re-runnable when new downloads arrive): parses image file names (`<UTC YYYYmmddHHMMSS>_<exposure>.jpg`) per station and day; reads every calibration YAML (validity window `mounted` → `demounted`, empty or broken files reported); lists ceilometer days per instrument; computes great-circle distances from the station list; counts image-days that have a camera within the pairing radius (default 1 km), ceilometer data that day and a valid calibration. Output: `docs/data/eye2sky_inventory.md`.
2. Tests: file-name patterns, haversine (OLDLR–OLUOL ≈ 0.41 km), calibration validity windows (end exclusive).

## Verification
| Item | Result |
|---|---|
| Images on disk | AURIC and BARSE, 9 days each (1–9 Apr 2022), 29,288 images, ≈ 1,630 per station-day (daytime, 30 s), all with exposure suffix `160` (fixed day exposure) |
| Size | ≈ 0.3 GB per station-day (AURIC 1 Apr: 315 MB) |
| Ceilometers | CDLRA (Oldenburg) and CDLRB (Westerstede): 118 days each, 1 Apr – 30 Jul 2022 |
| Cameras within 1 km of a ceilometer | CDLRA: **OLDLR** 0.00 km, **OLUOL** 0.41 km, **OLWIN** 0.41 km; CDLRB: **WESTE** 0.00 km |
| Calibration | OLDLR valid throughout Apr–Jul 2022 (files from 2022-01-31 and 2022-04-22); OLUOL has an empty 2020 file (reported, skipped) |
| **Pairable image-days now** | **0** — no images from OLDLR, OLUOL, OLWIN or WESTE are on disk |

## Exit criteria
- [x] Coverage matrix produced; number of usable (image, CBH) pairs known (0).
- Escalation (per plan): the owner must download images at the ceilometer sites before P047 (pairing). Recommended minimum: **OLDLR and WESTE, as many April–July 2022 days as possible** (≈ 0.3 GB per station-day; all 118 days of both ≈ 75 GB; OLUOL/OLWIN add a second view of CDLRA).

## Fit & data-risk notes
- The fixed exposure (suffix 160 on every image) removes exposure as a confound within Eye2Sky, but makes Eye2Sky photometrically different from auto-exposed consumer cameras (B0268): handled by augmentation (P048) and consumer-view synthesis (P049).
- AURIC/BARSE remain useful for self-supervised features and sky parsing, not for CBH.

## Deviations from plan & why
None. (The earlier estimate of ≈ 1 GB per station-day was corrected to ≈ 0.3 GB by measurement.)

## Next phase
P013 — Eye2Sky readers.
