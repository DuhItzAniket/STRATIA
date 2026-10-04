# P021 — Unified sample schema

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
One validated manifest row per image across all datasets, so every later phase (audits, splits, training, evaluation) reads the same table.

## Inputs / dependencies
P011 inventory (sizes), P013 Eye2Sky readers, P016 Montenegro soft labels, P017 MGCD, P018 segmentation index, P019 B0268 ingest.

## Work log
1. `stratia/data/manifest.py`: 31-column schema (identity, camera, time and its source, location, Sun position, calibration, size, source label, official split, harmonised labels, expert distributions, segmentation file, ceilometer station, licence, group, split) with `validate()` (non-null rules, UTC timestamps, valid JSON, unique IDs, allowed camera types and splits, coordinate and Sun ranges, image sizes). Columns filled by later phases are present as nulls.
2. `scripts/build_manifest.py`: assembles all datasets, joins image sizes from the inventory, computes Sun zenith/azimuth with pvlib wherever UTC time and location are known, validates, writes `data/manifest.parquet` (ignored) and `docs/data/manifest_summary.md`.
3. `tabulate` added to `requirements.in` and the lock file (used for Markdown reports).
4. Tests: valid frame, detection of duplicate IDs, unknown camera types and splits, broken JSON, out-of-range coordinates, naive timestamps.

## Verification
- 52,032 rows, 31 columns, **validation passed** (17 s).
- Per dataset: Eye2Sky 29,288 (UTC + Sun for all); MGCD 8,000 (official 4,000/4,000); SWINySEG 6,768; CCSN 2,543; Montenegro 2,522 (UTC + expert distributions for all; no Sun: site coordinates unpublished); SWIMSEG 1,013; Almería 818 (official 616/154/48); SWIMCAT 784; SHWIMSEG 156; SWINSEG 115; B0268 25 (UTC + Sun).
- Six camera types: fisheye all-sky (Eye2Sky, MGCD, Almería), consumer photos (CCSN), fixed low-cost (Montenegro), wide-angle USB (B0268), whole-sky-imager crops and patches (SWIM family).

## Exit criteria
- [x] Manifest built for all datasets; schema validated in tests.

## Fit & data-risk notes
- `group_id` currently equals `sample_id`; duplicate groups (P025) and temporal blocks (P027) must be assigned before any split is generated (P038), otherwise leakage is possible.
- Almería file names contain capture times, but whether they are UTC or local (metadata says UTC+1) is unverified; they are left without UTC until a Sun-position check (P044).
- Montenegro times are converted with the stated UTC+1; daylight saving in October 2025 is unverified.

## Deviations from plan & why
None.

## Next phase
P022 — Loader performance.
