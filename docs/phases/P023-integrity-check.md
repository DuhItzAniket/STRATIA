# P023 — Integrity check

Status: DONE     Date: 2026-10-06     Commit: (this commit)

## Objective
Decode every manifest image completely and account for every file that is not a sound 8-bit picture of the size
the manifest claims: corrupt or truncated files, empty files, odd pixel modes, EXIF rotation, size outliers.

## Inputs / dependencies
P021 manifest (`data/manifest.parquet`, 52,032 rows); the data root from `configs/paths.yaml`.

## Work log
1. `stratia/data/integrity.py`: `inspect_image()` decodes one image fully with Pillow (truncated files are errors,
   not tolerated) and returns size, format, mode, EXIF orientation, grey mean and standard deviation, and flags;
   `add_size_outliers()` judges pixel counts against each dataset's median; `summarise()` and `report_markdown()`.
2. `scripts/integrity_check.py`: process pool over the manifest, writes `data/integrity.parquet` and
   `docs/data/integrity_report.md`.
3. Tests (`tests/test_integrity_duplicates.py`): a sound JPEG; empty, missing, truncated and non-image files; an RGBA
   PNG; an EXIF-rotated JPEG; a size mismatch; outlier judgement and summary.
4. Added `docs/research_log.md` (the paper's working notes, now part of the phase protocol in `CONTRIBUTING.md`).

## Verification
- Run: `python scripts/integrity_check.py --workers 12`: 52,032 images decoded in 318 s (≈ 160 img/s, the Eye2Sky
  2112 × 2048 JPEGs dominate). Results in `data/integrity.parquet`; report `docs/data/integrity_report.md`.
- **No corrupt, truncated, empty or missing file; no odd pixel mode (every image is 8-bit RGB); no EXIF rotation
  anywhere; every decoded size equals the manifest's; no size outlier.** 51,989 images carry no flag.
- **43 "blank" images (grey-level standard deviation below 2): 39 in SWIMCAT, 4 in SHWIMSEG.** All are 125 × 125
  patches of class *clear sky* (`A-sky`), uniformly blue by nature; their means range from 33 to 254 grey levels,
  so they are not black or saturated frames. Explained: a featureless patch is a valid sample of "clear sky" in a
  patch dataset, not a faulty file. They stay in the manifest.
- Eye2Sky AURIC/BARSE (April 2022) has no black night frame: the lowest standard deviation is 13.3 grey levels
  (the cameras record a lit horizon and noise at night). Night handling remains a protocol question (P030, P037).
- Tests: 5 new (`tests/test_integrity_duplicates.py`); `python -m pytest -q`: all pass; `ruff check .` clean.

## Exit criteria
- [x] 0 unexplained failures: every flagged image is explained in `docs/data/integrity_report.md` and above.

## Fit & data-risk notes
- The datasets are mechanically sound: nothing in this phase removes an image. The data risks of Stage C lie
  elsewhere: duplicates (P024), near-duplicates (P025), label conflicts (P026), shortcuts (P029).
- "Blank" is a statistic of the patch, not a defect: in the sky-patch datasets a constant image *is* the class.
  A model trained on them learns that uniform blue means clear sky, which is correct but trivially so; the
  imbalance report (P030) should weigh how many clear-sky samples are near-constant.
- Per-dataset contrast differs a lot (median standard deviation 8 in SWIMCAT, 23 in Eye2Sky, 68 in MGCD): a
  cheap signature of the camera and dataset that the shortcut audit (P029) must test for.

## Deviations from plan & why
- "Blank" (no-contrast) frames and the grey-level statistics were added beyond the plan's list: for all-sky cameras
  they are the most common kind of unusable image, and the statistics cost nothing once the image is decoded.

## Next phase
P024: Exact duplicates.
