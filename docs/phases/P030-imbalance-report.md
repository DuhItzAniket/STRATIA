# P030 — Imbalance report

Status: DONE     Date: 2026-10-08     Commit: (this commit)

## Objective
Measure how unbalanced every label we will train on is, per dataset and per published split, and choose the
sampling rules that keep a minority class or a small source from collapsing.

## Inputs / dependencies
P021 manifest (native labels, official splits, sun position); P018 masks; P016 Montenegro rater distributions;
P014 ceilometer files (the Paper B targets); P029 (the source-identity shortcut that the sampling has to live with).

## Work log
1. `stratia/data/imbalance.py`: distributions with three imbalance numbers (largest/smallest ratio, normalised
   entropy, effective number of classes), cloud-base / sun-zenith / cloud-fraction binning, mask cloud fraction,
   the two sampling rules (`source_weights` with a power, `class_balanced_weights` after Cui et al. 2019), report
   and figure. `stratia/data/ceilometer.py::load_all` loads and caches all 236 ceilometer day files.
2. `scripts/imbalance_report.py`: 30 distributions over the manifest, the masks (8,870 images) and 1.36 M
   ceilometer records; writes `data/imbalance_tables.parquet`, `data/mask_cloud_fraction.parquet`,
   `data/ceilometer_records.parquet` (cache), `docs/data/imbalance_report.md`, `docs/data/figures/imbalance.png`.
3. Tests (`tests/test_imbalance_ceilometer_qc.py`, P030 part): distributions and metrics, the bins, mask fractions
   for a SWIM and an Almería mask, the sampling rules, report and figure.
4. Reviewed the distributions and decided the sampling (Verification, Decision).

## Verification
- Run: `python scripts/imbalance_report.py` (29 s with the ceilometer cache cold, 7 s warm).
- **Sources** (natural share → square-root share): Eye2Sky 56.3 % → 35.0 %; SWIM family 17.0 → 19.2; MGCD 15.4 → 18.3;
  CCSN 4.9 → 10.3; Montenegro 4.8 → 10.3; Almería 1.6 → 5.9; B0268 0.05 → 1.0 (21× oversampled; 25 frames).
- **Classes:** CCSN 11 genera, ratio 2.4 (Sc 13.4 % … Ci 5.5 %), entropy 0.99; MGCD 7 types, ratio 2.0 (train 1.7,
  test 2.3); SWIMCAT 5, ratio 3.0 (thick-dark 32 %, veil 10.8 %); Montenegro primary altitude class ratio 18.7
  (low 57.8 %, clear 21.8 %, high 13.4 %, middle 3.9 %, vertical 3.1 %).
- **Oktas (Montenegro majority):** U-shaped: 0 oktas 21.1 %, 8 oktas 25.9 %, 4 oktas 2.4 % (ratio 13.9), obscured 1.9 %.
- **Cloud-base code h (Montenegro majority):** 1–1.5 km 35.7 %, ≥ 2.5 km or none 22.6 %, 0.6–1 km 18.6 %, unknown 14.7 %;
  above 1.5 km almost never (1.4 %): the raters put nearly every base below 1.5 km.
- **Segmentation:** pixels are close to half sky, half cloud in every set (ratio 1.1–1.3), but per image SWINySEG is
  mid-cover (50–75 %: 36.9 %; clear < 5 %: 1.4 %; overcast: 2.7 %; ratio 25.8), SWIMSEG and SWINSEG the same (ratios 32
  and 54), while Almería spreads evenly over cover (ratio 2.2; clear 18.2 %, overcast 14.2 %) and its layer pixels are
  low 18.5 %, mid 13.7 %, high 13.2 % of the labelled sky.
- **Sun:** Eye2Sky (April, 53° N) has no frame with the sun above 60° elevation; zenith 50–70° holds 43.7 %, 85°+ 11.6 %.
- **Ceilometer bins (QC-clean records):** none 34 %, low 34 %, mid 20 %, high 12 % at both sites (ratio 2.9–3.1);
  layers 0 / 1 / 2 / 3 / 4: 34 / 51 / 12 / 2.4 / 0.5 %.
- Tests: 5 for this phase; `python -m pytest -q`: all pass; `ruff check .` clean.

## Exit criteria
- [x] Report: `docs/data/imbalance_report.md` (every distribution, metrics, sampling table, decision).
- [x] Sampling strategy chosen: square-root source sampling, class-balanced loss weights inside a source (beta =
  0.999), no resampling of soft labels, macro metrics by default, results by sun-zenith and cloud-fraction bin.

## Fit & data-risk notes
- **The classification sets are nearly balanced; the imbalance is between sources and in the physical targets.**
  Without source reweighting Eye2Sky is 56 % of every batch and the model's capacity goes to one camera (P029).
- **Montenegro is a low-cloud camera:** 58 % low cloud, 4 % middle, raters almost never above 1.5 km. Its height code
  cannot train a general CBH head and serves only as a soft, local evaluation (P035); the ceilometer bins are the
  CBH targets, and their "none" class (34 %) must be a first-class output ("no cloud overhead", contract v1).
- **Segmentation sets lack clear and overcast frames** (SWINySEG 1.4 % and 2.7 %): a model trained on them has barely
  seen the extremes where the legacy "85–99 % cloud" collapse happened (P074 monitor). Almería and the ceilometer
  "none" records supply the clear end; the data card says so.
- Rejected alternatives are in the report's Decision section (natural and uniform source sampling, class-uniform
  resampling).

## Deviations from plan & why
- "Per split" is reported for the published splits only (MGCD train/test, Almería train/val/test); STRATIA's own
  splits do not exist until P038, which will rerun this script's tables per split.
- The CBH-bin distribution was measured on the ceilometer records (the targets) rather than on paired frames, which
  do not exist yet (no images at the ceilometer sites, P015).

## Next phase
P031: Ceilometer QC & pairing tolerance.
