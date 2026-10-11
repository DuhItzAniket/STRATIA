# P047 — Image–ceilometer pairing

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
The cloud-base-height supervision set: every usable all-sky frame at the two ceilometer sites paired with the
ceilometer's label at its time, with the geometry that says where in the image the label applies, subsampled
against autocorrelation, split by day, and its test days locked.

## Inputs / dependencies
Eye2Sky images at OLDLR and WESTE, 1 April–31 July 2022, complete on disk (122 days each, `tools/download_eye2sky.py`);
P036 target table (`data/ceilometer_targets.parquet`, 30 s grid, ±30 s window = the P031 tolerance); P013
calibrations and masks; P044 Sun-fitted poses (refitted here, see below); P027 (day blocks, adjacent frames 82–92 %
the same scene); P037/P038 (in-domain Eye2Sky day assignment, shared across cameras); P039 (lock); P049 (views).

## Work log
1. **P044 prerequisite.** The later calibration windows (OLDLR from 22 April, WESTE from 27 June) had no usable
   pose at the 7-day sampling: per-week fits exposed whole-week false Sun tracks from the dome's ghost reflection
   when the high summer Sun clips beyond the detector's area limit. `scripts/sun_validation.py` now gates
   detections to 15° around the declared orientation's prediction, accepts larger discs, samples the ceilometer
   sites every day and refits a subset of stations into the records on file. All four site calibrations are `ok`
   (medians 0.44–0.94°), and the decisive check for this phase holds: the fitted zenith pixel agrees to 1 px between
   OLDLR's two files and 5 px between WESTE's, and stays within 5–10 px across clean fortnights all season. The
   cameras never moved; their declared orientations differ from the Sun by up to 11°. Details in P044 §Update.
2. `stratia/labels/cbh_pairs.py`: `frames_on_disk` (cadence grid, 120 s), `join_targets` (nearest grid point within
   15 s via `merge_asof`, valid windows), `zenith_geometry` (zenith pixel from the fitted pose, radius of a 10° cone,
   inside-mask test), `day_strata`, `assign_days` (inherited days keep their split; the rest greedy to 70/10/20
   inside each stratum, deterministic), `split_table`, `add_day_label_share`, `summary`, `report_markdown`.
3. `scripts/make_cbh_pairs.py` → `data/cbh_pairs.parquet` (every cadence frame with target, geometry, split,
   `day_label_share`), `data/splits/cbh_pairs.parquet` (protocol file), `docs/data/cbh_pairs.md`, figures.
4. `configs/splits.yaml` `cbh_pairs` section; `stratia/data/test_lock.py` knows the protocol; `test_lock.json`
   re-locked with `--force`: the 13 existing entries are byte-identical, two entries added
   (`cbh_pairs/OLDLR/test` 10,700 ids, `cbh_pairs/WESTE/test` 11,159 ids), manifest hash unchanged.
5. P049's `label_from_pairs`: synthetic views inherit the pair of their source frame from this table (pose, mask and
   daytime requirements included) instead of the provisional direct join.

## Verification
- `tests/test_cbh_pairs.py` (7): cadence enumeration ignores off-grid frames and leftover zips; the join takes the
  nearest grid point within the gap and refuses invalid windows and distant frames; zenith geometry (pixel at the
  optical centre for a zenith camera, ~11 px per degree, moves off-centre under a 3° tilt, inside-mask test); day
  assignment inherits, balances every stratum and is deterministic; split table drops masked-out and night
  frames; summary and report; the lock covers the new protocol. `tests/test_synthesis.py` +1 for the hook.
- `python scripts/make_cbh_pairs.py` (71 s):

| Station | Frames (2 min) | Daytime | Pairs | Days | none | low | mid | high | mixed | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| OLDLR | 67,976 | 54,990 | 51,675 | 118 | 15,430 | 19,781 | 10,355 | 4,964 | 1,145 | 0.98 |
| WESTE | 67,254 | 54,598 | 53,008 | 117 | 15,088 | 21,069 | 10,656 | 5,070 | 1,125 | 0.98 |

  Every pair sits on the 30 s grid (max gap 0 s); every zenith pixel is inside its mask. Lowest-base bins (both
  stations): 5,400 below 500 m, 11,954 at 500–1,000 m, 25,059 at 1–2 km, 14,351 at 2–4 km, 7,022 at 4–6 km, 3,912 at
  6–8 km, 6,392 at 8–12 km. Splits over 118 days: train 81 days / 71,867 pairs, val 12 / 10,957, test 25 / 21,859
  (104,683 pairs in the splits). Label mix is stable across months (low 33–45 %, none 24–38 %).
- `check_test_lock.py`: 15 test sets, split files match the lock. Whole suite 179 passed; `ruff check .` clean.
- Figures: `docs/data/figures/cbh_pairs_histogram.png` (base heights and labels per station),
  `docs/data/figures/cbh_pairs_days.png` (pairs per day coloured by split).

## Exit criteria
- [x] Pair table with counts per height bin (above and in `docs/data/cbh_pairs.md`).

## Fit & data-risk notes
- **Overcast-day over-representation** (the plan's guard): low cloud is 39 % of the pairs and "none" 29 %; the
  2-minute cadence keeps every day at ~450 pairs per station regardless of weather, so a slow overcast day
  contributes as many identical labels as a changing day. `day_label_share` is on every row; P075 weights by it or
  caps per day, and reports skill per label and per height bin, never a single average.
- The val split has few `high` pairs (307): days are the unit and high cloud rarely dominates a day. P075 should
  report val by label with that caveat or move one high-cloud day from train with a logged reason (the lock covers
  test only).
- Pose accuracy ~0.5° against a 10° ROI: the geometry is not the limiting error; the ceilometer's ±30 s window and
  the horizontal distance between the ceilometer beam and the camera's zenith column (both at the same site, tens of
  metres) are.
- Night frames (80 ms exposure) are in `data/cbh_pairs.parquet` with `daytime = False` and no split; a night model
  is out of scope.
- 4 days have no ceilometer data (118 of 122 paired); WESTE 18 April lacks daytime frames (camera gap on the server).

## Deviations from plan & why
- The plan lists OLUOL and OLWIN as CDLRA partners; no images for them are on disk (not downloaded: OLDLR is the
  co-located camera). The config maps them for a later download.
- Subsampling is a fixed 2-minute cadence rather than a per-day cap; the cap is left to the training sampler with
  `day_label_share` as its handle.
- Re-locking the test sets was done with `--force` as P039 requires; the commit explains it and this doc shows that
  nothing previously locked changed.

## Next phase
Stage F — P051 (baselines). Owner items unchanged: P041 frames and labellers, B0268 checkerboard.
