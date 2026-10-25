# P090 — Locked-test evaluation (once)

Status: IN PROGRESS     Date: 2026-10-25     Commit: (this commit)

Pre-registration first, numbers second. Everything below the line "Pre-registered protocol" was written on
2026-10-25 before any locked test row was read, and is not changed after the run; the results section is appended
afterwards. The pipeline was developed and checked on the validation rows (`--split val`, where it must reproduce
the Stage H validation numbers) and run on the locked sets exactly once (`--split test --final --reason ...`),
which `stratia.data.test_lock.require_final_reason` logs.

## Objective
Plan (Stage J, P090): all test sets, LODO, B0268, CBH sites; bootstrap 95 % CIs; results JSON hashed and
committed; done once, reason logged, no re-runs after looking. Research design §6: no decision may use test data;
any later re-evaluation is reported as such.

## Inputs / dependencies
- `test_lock.json` (P039, re-locked at P041: 23 test sets, manifest prefix `c41841f43cc488d7`), verified by
  `scripts/check_test_lock.py` before the run.
- Models (fixed before the run; nothing is selected on test):

| Role | Run | Weights |
|---|---|---|
| Global heads, in-domain, seed 0 (the deployed 1.0.0 heads) | `20261011T125754Z_global_indomain_e2s_s0_27d0d07c22` | EMA |
| Global heads, seeds 1 and 2 (seed spread) | `20261011T124825Z_global_indomain_e2s_s1_s1_bb2f2f438a`, `20261011T125312Z_global_indomain_e2s_s2_s2_9629d925c5` | EMA |
| LODO models (+ Eye2Sky étage), one per held-out source | `20261011T130211Z_global_lodo_ccsn_e2s_s0_2fca96d16c`, `20261011T130503Z_global_lodo_mgcd_e2s_s0_8428c46789`, `20261011T130753Z_global_lodo_montenegro_e2s_s0_6eab87db84`, `20261011T130934Z_global_lodo_almeria_e2s_s0_e0d03a19d5` | EMA |
| Dense heads (sky parsing + layers) | `20261011T150404Z_dense_sky_layer_s0_dcd14f4833` | best val epoch |
| CBH head, base | `20261011T101553Z_cbh_head_s0_[P075_base]_e286609cd2` (the exported one; seeds 1 and 2 kept no checkpoint) | best val epoch |
| CBH head, OLDLR-only (WESTE never seen) | `20261011T101430Z_cbh_head_s0_(OLDLR_only;_WESTE_held_out)_dcd5b03fe4` | best val epoch |
| B0268 graphs | `models/stratia-1.0.0-demo` (zero-shot, P096) and `models/stratia-1.0.1-demo` (adapted, P084) | ONNX |
| Reference baselines, refitted on train with the hyper-parameter chosen on val as in P054 / P058 | linear probes on DINOv3 CLS features: ViT-B/16 @ 224 for the LODO étage clause of C1 and ViT-L/16 @ 224 for its in-domain genus clause (the two P054 bests Gate G4 compared against; amended 2026-10-25 during the validation check, before any test row was read, from "ViT-B/16 for both"); ridge on DINOv3 ViT-S/16 @ 512 CLS features (P058) | — |

- Calibration `configs/calibration.json` (P089): CBH temperature 1.2394 and conformal radii 0.3265 / 0.4376, OOD
  statistics and threshold 1.4569 (the 1.0.0 model; the 1.0.1 graph carries its own, folded in).
- Caches: `dinov3_vits16_512_ffr1` (cls), `…-patches-all-pool4`, `…-patches-cbh-pool4`, `…-patches-seg`,
  `…-b0268` / `…-patches-b0268-pool4`, `dinov3_vitb16_224_ffr1`. Geometry from the P043–P045 calibrations and
  the manifest Sun, as in training.

## Pre-registered protocol

Script `scripts/locked_test.py`. One long results table `data/locked_test_<split>.parquet` (protocol, set, model,
metric, value, 95 % CI, n); for the test run also `results/P090_locked_test.json` with its SHA-256 next to it,
both committed, and `docs/data/locked_test.md`.

| Lock entry | Rows evaluated | Model(s) | Metrics |
|---|---|---|---|
| `in_domain/{ccsn, mgcd, swim, montenegro, almeria}/test` | the global rows of those sources with split test (SWIM = SWIMCAT genus rows; the masks have no global label) | in-domain heads seeds 0, 1, 2 (per seed and mean ± std) | `score_dataset` (P082): étage macro-F1 at 0.5, unsupported-positive rate, genus macro-F1 / accuracy (CCSN, MGCD, SWIMCAT), Montenegro oktas MAE / within ±1 / KL, étage-set agreement and the C2 ratios against the P035 ceiling |
| `cbh_pairs/{OLDLR, WESTE}/test` as étage | Eye2Sky pairs with split test, 10-min cadence, confidence ≥ 0.8 (the training cadence) | in-domain heads | lowest-étage accuracy, étage recall (partial labels) |
| `lodo/{ccsn, mgcd, montenegro, almeria}/test` | the held-out source's test rows of the fold | the fold's LODO model; the P054 ViT-B/16 @ 224 probe refitted on the fold's train rows | étage macro-F1 at 0.5 (C1); unsupported-positive rate; étage macro-F1 with thresholds chosen on the fold's val rows (secondary); the probe's macro-F1 |
| `in_domain/{ccsn, mgcd}/test` as genus (C1 second clause) | native-class rows | in-domain heads seed 0; the P054 ViT-L/16 @ 224 probe refitted | genus macro-F1 |
| `in_domain/{almeria, swim}/test` as masks | the segmentation-index images with split test (Almería; SWIMSEG / SWINSEG / SHWIMSEG / SWINySEG) | dense heads | sky mIoU and per-class IoU per dataset and overall; Almería layer mIoU and per-layer IoU; predicted-cloud fraction on the clear Eye2Sky frames of the test split (collapse check, < 0.20) |
| `cbh_pairs/{OLDLR, WESTE}/test` | pairs with split test, 10-min cadence | CBH base (with geometry; geometry zeroed as the H4 check); OLDLR-only model on WESTE; the P058 ridge refitted | MAE, bias, RMSD, skill = 1 − MAE / the training climatology's MAE on the split's cloudy pairs (both stations, as the trainer and Gate G4 computed the per-station skills; the station's own climatology MAE as a secondary row), within ±250 / 500 / 1000 m, MAE by height bin, CRPS, 80 % coverage raw and 80 / 95 % coverage with the P089 temperature and conformal radii, cloud-presence balanced accuracy |
| `few_shot_target/*/test` (the seven entries share the one locked day 2026-10-20, 120 frames) | the locked day | the 1.0.0-demo and 1.0.1-demo graphs through `scripts/eval_b0268.py --roles test` | P083 scoring against each rater, averaged: étage set exact, oktas within ±1, bias, top genus in the rater's set; cloud-fraction correlation with the raters' cover; OOD flagged share |
| Reliability | in-domain test rows (ID) against the B0268 locked day (OOD), both through the cached-feature path of the 1.0.0 heads | OOD score with the P089 statistics | AUROC; flagged share per source at the threshold |
| `lodo/swim/test`, `held_out_station/*/test` | — | none exists: no SWIM LODO model and no AURIC-only model were trained in Stage H | not evaluated; reported as such |

Confidence intervals: cluster bootstrap, 500 resamples, over the split's units (the temporal / group units of
`in_domain.parquet` for the global and LODO rows, calendar days for the Eye2Sky pairs, images for the masks,
frames for the B0268 day against the pair's majority). Seeds: three for the in-domain global heads, one for every
other model (recorded as a deviation from the research design's "3 seeds"). Thresholds: 0.5 everywhere, as at
Gate G4; the val-tuned étage thresholds are a secondary row, never the headline.

Headline table (the Gate G4 criteria, validation value from P082 next to the test value): C1 LODO étage macro-F1
mean of the four folds against the probe; C1 in-domain genus macro-F1 (CCSN, MGCD) against the probe; C2 Montenegro
étage-set and oktas ratios; C3 CBH skill on OLDLR test, skill of the OLDLR-only model on WESTE test, MAE against the
ridge (the P058 CNN kept no checkpoint: its 724 m is a validation number), calibrated 80 % coverage.

What P090 does not do: no threshold, prompt, hyper-parameter or model choice on the test rows; no second run. If
the script fails part-way on the test rows, the partial numbers are kept and the rerun is logged as a second
evaluation with its reason, and the paper says so.

## Work log
(appended after the run)

## Verification
(appended after the run)

## Exit criteria
- [ ] Every lock entry evaluated or explicitly marked not evaluated, with the model and metric pre-registered above.
- [ ] Bootstrap 95 % CIs on every headline number.
- [ ] Results JSON committed with its SHA-256; the final-evaluation reason logged.
- [ ] One run.

## Fit & data-risk notes
(appended after the run)

## Deviations from plan & why
(appended after the run)

## Next phase
P091 — Ablations (on validation data; the test numbers above are final).
