# P014 — Ceilometer reader

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Read the Eye2Sky Lufft CHM15k ceilometer files correctly (time, layers, units, quality), because they are STRATIA's ground truth for cloud-base height (CBH).

## Inputs / dependencies
P012; `Eye2Sky/ceilometer/ceil/data/{CDLRA,CDLRB}` (118 daily NetCDF3 files each) and the vendor plots in `.../plots/`.

## Work log
1. Inspected the file conventions before writing code: time = seconds since 1904-01-01 UTC at ~15 s; `cbh/cbe/cdp` 4 layers lowest-first, −1 = none; heights above the instrument (`altitude` = 0, `cho` = 0 in the files); `sci` 0/1/2/3/4 = nothing/rain/fog/snow/precipitation or particles on window.
2. `stratia/data/ceilometer.py`: `read_chm15k` (DataFrame indexed by UTC, heights in m with NaN for no layer, layer count, QC flags), `read_backscatter` (for plots), `lowest_cbh_near` (median lowest base in a time window, for image pairing in P047).
3. `scripts/ceilometer_summary.py`: per-day statistics for all 236 files (`data/ceilometer_days.parquet`, ignored), `docs/data/ceilometer_summary.md`, and time-height figures.
4. **Two data problems found and handled:**
   - CDLRB's own file states latitude **5.325°** (should be 53.25°): file coordinates are ignored; the station list is authoritative.
   - **58% of CDLRA records carry error bit 0x8000.** It appears on all 118 days, rising from 25% (April) to 88% (late July) as laser quality falls (median 60% vs 72%), while CBH statistics are essentially unchanged (with bit / without: median 1,887 / 1,934 m; 10th percentile 554 / 566 m; cloud fraction 0.662 / 0.656). Treated as a **laser-ageing warning** (`qc_laser_warning`), not an error; the other error bits (0.6% of CDLRA records) remain errors. The slightly lower 90th percentile with the bit (7,972 vs 8,270 m) suggests weaker detection of the highest clouds and is tracked.

## Verification
- **Visual check against the vendor plot** (CDLRA, 2022-06-01): every structure matches in time and height — the ~2 km deck 00–08 UTC, the 0.3–0.6 km layer ~08–09:30, high cloud at 5–7.5 km 13:30–18:00, the ~1.3 km layer 19–20 UTC, clear sky after 21 UTC; the vendor's rain marks (~02, 09, 12, 14–14:30 UTC) coincide with `sci` = 1 records. Our figures: `docs/data/figures/ceilometer_{CDLRA,CDLRB}_20220601.png`.
- Summary over 118 days per instrument (679,663 / 679,666 records):

| | CDLRA | CDLRB |
|---|---|---|
| Lowest layer present | 67.4% | 68.1% |
| Median lowest CBH (QC-clean) | 1,910 m | 1,848 m |
| ≥ 2 layers | 14.8% | 14.8% |
| Rain / particles on window / optics < 50% | 2.9% / 3.0% / 2.9% | 3.1% / 0.4% / 0.4% |
| CBH 0–1 / 1–2 / 2–4 / 4–8 / 8–12 km | 22.1 / 29.8 / 19.8 / 17.8 / 10.4% | 24.1 / 29.1 / 18.9 / 17.3 / 10.5% |

- Tests (synthetic NetCDF3 file): epoch and 15 s spacing, layer parsing and NaN handling, every QC flag including the 0x8000 rule, windowed lowest-CBH median.

## Exit criteria
- [x] Days plotted and matched to the vendor plots (2022-06-01 checked visually; all 236 days parsed).

## Fit & data-risk notes
- The two instruments ~15 km apart have nearly identical climatologies, which makes CDLRB a fair held-out site for CBH (pre-registered criterion C3).
- The CBH climatology above is the baseline a single-camera model must beat (skill = 1 − MAE / MAE_climatology).
- Figures are derived from Eye2Sky data (CDLA-Sharing 1.0); they are shared under the same terms.

## Deviations from plan & why
- One day (not a full week) was compared visually; all days were parsed and summarised, and the per-day table is in the Parquet file.
- Line-length limit raised from 120 to 130 characters (`pyproject.toml`) to keep Markdown-table code readable.

## Next phase
P015 — Acquire missing public data.
