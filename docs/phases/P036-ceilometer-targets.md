# P036 — Ceilometer → targets

Status: DONE     Date: 2026-10-08     Commit: (this commit)

## Objective
Turn the ceilometer records into the labels a camera frame at the same site receives: zenith cloud-base height
(lowest layer), layer count, étage of the lowest base and "no cloud overhead", at the pairing tolerance of P031,
with the multi-layer ambiguity recorded rather than discarded.

## Inputs / dependencies
P031 (±30 s tolerance; QC flags); P030/P031 record cache; P033 étage heights (weak 2 / 6 km) and the WMO
mid-latitude boundary (2 / 7 km) for the planned ablation; the Eye2Sky station list (site coordinates for the
daytime flag). No camera frames exist at the ceilometer sites yet (P015), so the table is built on a 30 s grid and
pairing (P047) will look frames up in it.

## Work log
1. `stratia/labels/ceilometer_targets.py`: `targets_at` (vectorised window counts, per-window medians, spread,
   second layer, labels under both threshold sets, confidence), `time_grid`, `daytime` (pvlib, site coordinates),
   `summary`, report.
2. `scripts/ceilometer_targets.py`: writes `data/ceilometer_targets.parquet` (679,680 rows: both sites, every 30 s
   of every day) and `docs/data/ceilometer_targets.md`.
3. Tests (`tests/test_ceilometer_targets.py`, 2): the window rule on a synthetic stream (cloud block with a flagged
   record and a second layer, clear block, half-cloudy block, empty window), thresholds, grid, daytime, summary.

## Verification
- Run: `python scripts/ceilometer_targets.py` (34 s).

| Site | Daytime grid points | With a label | none | low | mid | high | mixed | WMO thresholds (cloudy): low / mid / high | ≥ 2 layers | Base spread in window (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| CDLRA | 220,153 | 94.0 % | 30.0 % | 38.2 % | 20.1 % | 9.6 % | 2.1 % | 56.3 / 33.0 / 10.8 % | 15.7 % | 33 m |
| CDLRB | 220,360 | 97.1 % | 29.0 % | 39.5 % | 19.9 % | 9.5 % | 2.1 % | 57.3 / 31.4 / 11.3 % | 15.9 % | 35 m |

- The unlabelled 3–6 % of daytime windows are the flagged records (rain, window particles, optics, error bits) and
  the missing days; 6.7–6.9 % of labelled windows are mixed (some records see cloud, some not), of which 2.1 % fall
  below the half-share rule and carry the label "mixed". The base spread inside a window is 33–35 m at the median,
  far below the étage band widths. The weak/WMO switch moves 3–4 points of cloudy windows from high to mid.
- Tests: 2 new; `python -m pytest -q`: all pass; `ruff check .` clean.

## Exit criteria
- [x] Target table for all (future) paired frames: `data/ceilometer_targets.parquet`, one row per site and 30 s
  grid point, with label, confidence, base, spread, second layer, layer count, QC share, daytime flag.

## Fit & data-risk notes
- **The CBH head's class balance** (daytime, labelled): none 30 %, low 38–40 %, mid 20 %, high 10 %, mixed 2 %: the
  class-balanced weights of P030 apply, and "none" is a first-class output (contract v1).
- **Mixed windows are information, not noise**: the confidence (share behind the label) weights the loss, and the
  2 % "mixed" windows are evaluated separately (broken-cloud edges are where a single-camera CBH model will be
  least certain).
- **Two thresholds, one table**: the P036 ablation (weak vs WMO) is a column choice.
- Everything here is on the ceilometer side; the pairing itself, and the check of the tolerance against real frames,
  wait for the OLDLR and WESTE images (P047).

## Deviations from plan & why
- "Target table for all paired frames" became "for every 30 s grid point", because no frames exist at the sites
  yet; `targets_at` also accepts exact frame times so P047 can recompute instead of looking up.

## Next phase
P037: Split design.
