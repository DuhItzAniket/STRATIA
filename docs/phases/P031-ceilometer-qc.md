# P031 — Ceilometer QC & pairing tolerance

Status: DONE     Date: 2026-10-08     Commit: (this commit)

## Objective
Check the two Eye2Sky ceilometers' record streams (completeness, physical range, layer order, sky-condition and
instrument flags) and decide how far in time a camera frame may be from the ceilometer records that label it.

## Inputs / dependencies
P014 reader and its findings (15 s cadence, −1 = no layer, laser-ageing bit on CDLRA); P030's cached records
(`data/ceilometer_records.parquet`, 1,359,329 records, 118 days per site).

## Work log
1. `stratia/data/ceilometer_qc.py`: QC summary per instrument; agreement of two records Δ apart (presence, étage of
   the lowest base, height change); within-window unanimity on a one-minute grid; the tolerance rule; cross-site
   agreement at the same instants; figure; report.
2. `scripts/ceilometer_qc.py`: writes `data/ceilometer_qc.parquet`, `docs/data/ceilometer_qc.md`,
   `docs/data/figures/ceilometer_pairing.png`.
3. Tests (P031 part of `tests/test_imbalance_ceilometer_qc.py`): completeness, gaps and layer-order violations on a
   synthetic stream; agreement falling with offset; window unanimity and the tolerance rule; cross-site pairing;
   report and figure.

## Verification
- Run: `python scripts/ceilometer_qc.py` (20 s).
- **QC:** 679,663 (CDLRA) and 679,666 (CDLRB) records over 118 days each, 5,759–5,760 per day (completeness
  99.998 %), no duplicate times, two gaps over 60 s per site (two missing days each, between the months),
  no base above 15 km, no layer out of order. Cloud present 67–68 %, two or more layers 15 %. Flags: rain 2.9 / 3.1 %,
  particles on the window 3.0 / 0.4 %, optics below 50 % 2.9 / 0.4 %, error bits 0.6 / 0.01 %, any flag 6.1 / 3.5 %;
  CDLRA's laser warning on 57.9 % of records (P014: no effect on the base statistics; CDLRB 0 %).
- **Two records Δ apart** (QC-clean, both sites alike): presence / étage agreement 97.7 / 97.3 % at 15 s, 96.4 / 96.1 % at
  30 s, 94.8 / 94.6 % at 1 min, 92.7 / 92.8 % at 2 min, 89.0 / 89.4 % at 5 min, 82 / 81 % at 30 min; the median height
  change grows from 10 m (15 s) to 44 m (2 min), 85 m (5 min) and 236 m (30 min); within 200 m: 93 % at 15 s, 80 % at 2 min.
- **All records inside ±w** (one-minute grid): ±30 s holds 4.0 clean records and all agree on presence in 94.3 % and on
  étage in 94.0 % of windows; ±1 min 8 records, 90 %; ±2 min 16 records, 84 %; ±5 min 39 records, 74–76 %; ±10 min 65–68 %.
  Coverage (windows with at least one clean record) 92–96 %.
- **The two sites at the same instant** (627,840 pairs within 10 s): presence agreement 82.0 %, étage agreement 79.6 %,
  median |difference| of the base 255 m, within 500 m 65.6 %.
- Tests: 4 for this phase; full suite passes; `ruff check .` clean.

## Exit criteria
- [x] Pairing tolerance chosen with justification: **±30 s** (median of the clean records' lowest base; share of
  cloudy records as confidence; no label when the window has no clean record). Two records 30 s apart disagree in
  3.6 % (presence) and 3.9 % (étage) of cases, barely above the instrument's own 15 s consistency (2.3 / 2.7 %); at
  ±2 min the disagreement doubles and at ±5 min it triples, for one to two points more coverage. ±15 s was rejected
  because a single dropout would decide the label.

## Fit & data-risk notes
- **A cloud base is local.** Two instruments 15 km apart disagree on cloud presence 18 % of the time and differ by
  255 m at the median when both see cloud, far more than two records minutes apart at one site. CDLRB is therefore
  a genuine held-out site for criterion C3, and no frame may be labelled from a distant ceilometer.
- **Time mismatch is a small, measured part of the label noise** (about 4 % at the chosen tolerance), to be quoted
  when the CBH error of a model is compared with the ceilometer's own consistency.
- The 34 % "no cloud overhead" records are as much a target as the heights; the pairing keeps them as labels with
  the cloudy share as confidence (P036).
- Rain, fog and particles on the window (3–6 % of records) are excluded from labels; the same frames are the
  camera's "obscured" cases and should be kept for the obscured class, not thrown away (P034).

## Deviations from plan & why
- The plan's windows were ±30 s / ±2 / ±5 min; ±1 min and ±10 min were added to see the shape of the curve, and the
  cross-site comparison was added because the station-held-out protocol needs it.
- No camera frames exist at the ceilometer sites yet (P015), so the tolerance is derived from the ceilometer's own
  time series; it will be rechecked against the frames when they arrive (P047).

## Next phase
P032: Gate G1 — Data audit sign-off (data card).
