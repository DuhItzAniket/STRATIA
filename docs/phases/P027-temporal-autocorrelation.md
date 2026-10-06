# P027 — Temporal autocorrelation

Status: DONE     Date: 2026-10-06     Commit: (this commit)

## Objective
Measure how long two frames of one camera stay "the same scene", for every dataset that has timestamps, and decide
per dataset the unit a split may not cut through (block) and the buffer two blocks need (gap).

## Inputs / dependencies
P021 manifest (Eye2Sky, Montenegro and B0268 timestamps; Almería's are parsed from its file names here); P025
instruments: the cached DINOv3 ViT-S/16 CLS embeddings (`cache/features/dinov3_vits16_224_cls.npy`) and the plain
pHashes (`data/perceptual_hashes.parquet`). DeepSky (named in the plan) is not available (P015).

## Work log
1. `stratia/data/temporal.py`: pair sampling at time gaps in doubling bins (30 s … 23 days) within one series or
   across two (absolute gap, with a "same moment" bin); two baselines from the same camera, *different day* (pairs
   more than a day apart) and *same hour, other day* (time of day within 30 min, different days: the sun alone);
   cosine and pHash measures; curves per bin; the gap rule (mean-cosine excess <= 10 % of the adjacent-frame excess,
   and same-scene share within one point of the baseline); day structure; figure; report.
2. `scripts/temporal_autocorrelation.py`: curves for eye2sky-AURIC, eye2sky-BARSE, montenegro-lowcost,
   almeria-Kontas and AURIC × BARSE; writes `data/temporal_curves.parquet`, `docs/data/temporal_report.md`,
   `docs/data/figures/temporal_autocorrelation.png`; the reviewed splitting units are in the report's last section.
3. Tests (`tests/test_temporal.py`, 6): pairs respect the bins within and across series; same-hour pairs are on other
   days at the same time of day; a drifting synthetic series gives a decaying curve and a finite gap, an unreachable
   baseline gives none; file-name timestamps and day structure; report and figure.
4. Reviewed the curves and decided the blocks (Verification).

## Verification
- Run: `python scripts/temporal_autocorrelation.py` (3 s; 5,000 pairs per bin, 20,000 per baseline, seed 0).

| Series | Days | Cadence | Adjacent cosine (same-scene share) | Same-scene share < 1 % after | Mean cosine at the different-day level after | Different day / same hour other day |
|---|---|---|---|---|---|---|
| eye2sky-AURIC | 9 | 30 s | 0.979 (82 %) | 2.1–4.3 h | 8.5–17 h | 0.774 / 0.791 |
| eye2sky-BARSE | 9 | 30 s | 0.985 (92 %) | 4.3–8.5 h | 4.3–8.5 h | 0.788 / 0.806 |
| montenegro-lowcost | 69 | 20 min | 0.921 (18 %) | 64 min–2.1 h | 8.5–17 h, **but 0.30 excess again at 17–34 h**, 0.09–0.17 up to 6 d | 0.643 / 0.683 |
| almeria-Kontas | 206 | bursts, 46 min median | 0.987 (100 %) | 8.5–17 h | 17–34 h | 0.808 / 0.802 |
| AURIC × BARSE | 9 | same moment | 0.813 (0.4 %) | — | 8.5–17 h | 0.702 / — |

- **Eye2Sky:** the curve is a smooth decay: half of the adjacent pairs are the same scene after 1–2 min, the share
  is below 1 % after 2–4 h, and the mean cosine meets the different-day level between 8.5 and 17 h; the 8.5–17 h
  bin (morning against evening of one day) is *below* that level (excess −0.10/−0.08), and the 17–34 h bin (the same
  hour next day) equals the same-hour-other-day baseline (0.790 vs 0.791; 0.806 vs 0.806): consecutive days are no
  more alike than any two days at the same hour. The same hour on another day is 0.02 above a random different-day
  pair: the sun's contribution, present in any cross-day split. Adjacent 30 s frames have a mean pHash distance of
  6.8, so P025's copy rule (<= 2) rarely fires on them; the P025 decision to leave same-station pairs to this phase
  was right.
- **Montenegro:** the excess is gone within a day (0.03 at 8.5–17 h) but returns at one day (0.30) and stays at
  0.09–0.17 up to six days, reaching the baseline only at 11–23 days (−0.03): consecutive days share weather and
  season (the series runs October–December with a lowering sun), so single-day blocks would leak.
- **Almería:** frames come in bursts (100 % same scene below 2 min, 52 % at 4–8 min) with hours or days between
  bursts; the excess is 0.05 at one day.
- **Across stations:** AURIC and BARSE frames at the same moment have cosine 0.813 against a different-day level of
  0.702, and 0.4 % of them are the same scene: the two cameras 15 km apart see correlated weather, not the same
  picture; the correlation decays with the same hours-long time scale as within a station.
- Tests: 6 in `tests/test_temporal.py`; `python -m pytest -q`: 72 passed; `ruff check .` clean.

## Exit criteria
- [x] Minimum block size / gap per dataset decided (also at the end of `docs/data/temporal_report.md`):

| Series | Block (the unit a split may not cut) | Gap between blocks | Why |
|---|---|---|---|
| Eye2Sky (each station) | one calendar day | none beyond the night (10.5 h) | excess at the baseline from 8.5–17 h; next-day same-hour pairs equal the same-hour baseline |
| Eye2Sky across stations | a held-out station is a legitimate out-of-camera test | hold out its days as well to remove shared weather | same-moment cosine 0.81 vs 0.70, same-scene 0.4 % |
| Montenegro | contiguous run of >= 7 days; splits take whole blocks | none (an 11-day buffer would cost a sixth of the data); residual 0.09–0.17 excess across block boundaries reported | excess 0.30 at one day, 0.09–0.17 up to six days |
| Almería (Kontas) | one calendar day | none | excess 0.05 at one day |
| B0268 | one block (25 frames in 5 min) | — | no split until the logger runs for days |
| MGCD, CCSN, SWIM family | P025 same-scene groups | — | no timestamps |

## Fit & data-risk notes
- **Random splits of all-sky time series are leakage by construction:** at 30 s cadence, 82–92 % of adjacent pairs
  are the same scene and the share stays above 10 % for 16–32 min. Day blocks (P037, P038) are the smallest honest
  unit for Eye2Sky and Almería; Montenegro needs weeks.
- **The sun is a confound that no temporal split removes** (0.02 cosine at the same hour on any day; larger for the
  Montenegro camera with its fixed foreground, 0.04). The shortcut audit (P029) should test whether a probe can read
  the time of day from the features, and the model's ray-map input (sun position) turns the confound into a known
  variable instead of a hidden one.
- **Nine days of Eye2Sky in the manifest** is few blocks for a day-level split; the April–July download on disk has
  about 120 days per station and should be ingested (P021 follow-up) before the splits are generated.
- Almería's timestamps live only in file names; the manifest has no `utc` for it. A P021 follow-up should parse them
  into the manifest (sun position then becomes available for Almería too).

## Deviations from plan & why
- DeepSky is absent (no access, P015); the cross-station Eye2Sky curve was added instead because the station-held-out
  protocol (P037) needs to know whether the two stations are near-duplicates of each other at the same moment.
- The gap rule uses two criteria (mean-cosine excess and same-scene share) and takes the larger; the plan said
  "curves", not a rule, so the rule and its 10 % / 1-point tolerances are stated here and in the module docstring.

## Next phase
P028: MGCD ≟ GRSCD (deferred: GRSCD not obtainable) → P029: Shortcut audit.
