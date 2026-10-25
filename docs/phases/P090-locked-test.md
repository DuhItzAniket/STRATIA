# P090 — Locked-test evaluation (once)

Status: DONE     Date: 2026-10-25     Commit: (this commit)

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
1. Validation rehearsal (`--split val`, 500 resamples, 307 s): the pipeline reproduces every Stage H validation
   number it re-evaluates — in-domain and LODO global metrics (`data/stage1_global.parquet`), the dense mIoUs
   (P074), the CBH MAE / skills including the held-out WESTE 0.448 (P075), the ridge 699 m (P058) and the B0268
   zero-shot and adapted numbers (P083 / P084) — to the last digit (`docs/data/locked_test_val_check.md`).
   Two amendments were made during the rehearsal, before any test row was read, and are marked above: the C1
   in-domain reference is the ViT-L/16 @ 224 probe (the one Gate G4 compared against) and the per-station CBH
   skill uses the split's climatology MAE (Gate G4's definition). Pre-registration and script committed as
   `788fb4b`.
2. First evaluation, 11:36 UTC, run `20261025T113617Z_locked_test_test_f2f212f294`: stopped in the LODO section
   with a shape error — the secondary "thresholds chosen on val" row called `T.tuned_thresholds`, which returns
   predictions on its own rows, so it only breaks when val and test rows differ in number, which the rehearsal
   cannot show. The in-domain and OOD sections had completed, nothing had been saved, and only the OOD line
   had been read (flagged shares 0.054 / 0.400, AUROC 0.939; identical in the second evaluation). Fix in
   `7d4b7a4` (`val_thresholds` returns per-column thresholds; the table is saved after every section).
3. Second evaluation, 11:39 UTC, run `20261025T113929Z_locked_test_test_f2f212f294`, 288 s on the RTX 4050,
   500 cluster-bootstrap resamples: complete. Both reasons are in `runs/final_eval_log.md` (git-ignored; the
   two entries are quoted under Verification). Outputs: `results/P090_locked_test.json` (SHA-256
   `e2b4833066699d9dbbe9dc8ec37abc519ffaab798140564613758ed738df58fb`, in `results/P090_locked_test.sha256`),
   `docs/data/locked_test.md` (every number), `data/locked_test_test.parquet`.

## Verification
Final-evaluation log (two entries, lock `2026-10-22T17:48:53Z`, manifest prefix `c41841f43cc488d7`):

| When (UTC) | Run | Reason (abridged) |
|---|---|---|
| 2026-10-25 11:36 | `20261025T113617Z_locked_test_test_f2f212f294` | P090 locked-test evaluation for the CVPR 2027 paper, run once: the Gate G4 models …, the two B0268 graphs, the P054 probes / P058 ridge refitted; protocol pre-registered in 788fb4b |
| 2026-10-25 11:39 | `20261025T113929Z_locked_test_test_f2f212f294` | second evaluation: the first crashed in the LODO section on a shape bug in the secondary val-tuned row …; only the OOD line had been read; models, protocol and metrics unchanged |

### The Gate G4 criteria on the locked tests (P082 validation value next to each)

| Criterion | Quantity | Val (P082) | Test | 95 % CI | Reference on test | Required | Holds |
|---|---|---|---|---|---|---|---|
| C1 | LODO étage macro-F1, mean of the four folds | 0.548 | 0.540 | per fold below | 0.529 (ViT-B/16 @ 224 probe, refitted) | ≥ reference + 0.05 | **no** (+0.011) |
| C1 | in-domain genus macro-F1 (CCSN, MGCD) | 0.719 | 0.706 | per source below | 0.733 (ViT-L/16 @ 224 probe, refitted) | ≥ reference − 0.01 | **no** (−0.027) |
| C2 | Montenegro étage-set agreement / human ceiling | 1.047 | 0.926 | two test blocks, see below | 0.90 | ≥ 0.90 | yes |
| C2 | Montenegro oktas ±1 agreement / human ceiling | 1.027 | 0.963 | two test blocks | 0.90 | ≥ 0.90 | yes |
| C3 | CBH skill, OLDLR (CDLRA) test pairs | 0.471 | 0.536 | [0.440, 0.630] | 0.20 | ≥ 0.20 | yes |
| C3 | CBH skill, WESTE (CDLRB) test pairs, OLDLR-only training | 0.448 | 0.436 | [0.276, 0.585] | 0.10 | ≥ 0.10 | yes |
| C3 | CBH MAE (m), all test pairs | 566 | 872 | [679, 1055] | 1,096 (ridge, refitted; must be ≤ 986) | ≤ 0.9 × reference | yes |
| C3 | 80 % interval coverage with the P089 temperature and radius | 0.802 | 0.720 | — | 0.80 nominal | 0.75–0.85 | **no** |

On the locked tests C2 holds, C3 holds on three of its four components (the calibrated 80 % interval covers
72 %), and C1 fails on both clauses. The Gate G4 rule (C3 and (C1 or C2)) was the validation-time go / no-go;
these are the numbers the paper reports.

### In-domain test (seed 0 with its cluster-bootstrap CI; mean ± std over seeds 0–2)

| Source | n | Étage macro-F1 | Genus macro-F1 | Probe ViT-L/16 genus F1 | Other |
|---|---|---|---|---|---|
| CCSN | 512 | 0.719 [0.676, 0.755]; 0.730 ± 0.008 | 0.591 [0.544, 0.629]; 0.594 ± 0.003 (acc 0.600) | 0.549 [0.501, 0.591] | |
| MGCD | 1,569 | 1.000 (low / high only) | 0.821 [0.706, 0.997]; 0.828 ± 0.007 (acc 0.879) | 0.918 [0.763, 0.949] | 86 same-scene units: wide CI |
| Montenegro | 446 | 0.680 [0.629, 0.725]; 0.670 ± 0.008 | — | — | étage-set agreement 0.789 (0.784 ± 0.004) = 0.926 of the P035 ceiling; oktas within ±1 0.841 = 0.963 of the ceiling, MAE 0.71 oktas, KL 0.66; the test set is two temporal blocks (287 + 159 frames: agreement 0.786 and 0.791), so its cluster CI spans two clusters and says little |
| Almería | 164 | 0.803 [0.678, 0.862]; 0.786 ± 0.023 | — | — | |
| Eye2Sky pairs (test split, 2,672 frames, 25 days) | 2,672 | lowest-étage accuracy 0.704 [0.618, 0.778]; 0.714 ± 0.008 | — | — | étage recall 0.964 |

### LODO test (held-out source's test rows; fixed 0.5 thresholds, [CI]; val-chosen thresholds as the secondary)

| Held-out | n | Model | Model, val-chosen thresholds | Probe ViT-B/16 @ 224 |
|---|---|---|---|---|
| CCSN | 2,288 | 0.451 [0.430, 0.471] | 0.533 | 0.396 [0.376, 0.414] |
| MGCD | 7,041 | 0.709 [0.620, 0.776] | 0.822 | 0.860 [0.793, 0.917] |
| Montenegro | 2,011 (9 units) | 0.367 [0.298, 0.446] | 0.473 | 0.463 [0.392, 0.525] |
| Almería | 734 | 0.634 [0.599, 0.660] | 0.633 | 0.397 [0.361, 0.425] |
| mean | | 0.540 | 0.615 | 0.529 |

The model beats the probe on CCSN and Almería and loses on MGCD and Montenegro; the mean gap is +0.011 (val
−0.035), five points short of the pre-registered margin either way.

### Reliability (P089 statistics and threshold 1.4569, cached-feature path)

Flagged share on the in-domain test rows 0.054 overall: Eye2Sky 0.000, Montenegro 0.004, MGCD 0.022, Almería
0.055, SWIM 0.191, CCSN 0.430. B0268 locked day: 0.400 (0.508 through the 1.0.0 graph). AUROC of the score for
the B0268 day against the in-domain test rows 0.939 (the supporting criterion asked ≥ 0.80).

### Sky parsing and layers (dense heads, 1,775 test images)

| Set | n | Sky mIoU [CI] | Sky / cloud IoU | Layer mIoU [CI] (low / mid / high) |
|---|---|---|---|---|
| all | 1,775 | 0.900 [0.897, 0.903] | 0.859 / 0.878 | 0.668 [0.596, 0.743] (0.783 / 0.454 / 0.766) |
| Almería | 164 | 0.885 [0.871, 0.896] | 0.867 / 0.824 | 0.668 (the only layer set) |
| SWIMSEG / SWINSEG / SHWIMSEG / SWINySEG | 203 / 23 / 31 / 1,354 | 0.865 / 0.866 / 0.822 / 0.872 | | |

Collapse check: predicted-cloud fraction 0.281 on 200 clear Eye2Sky frames of the test split (bar < 0.20; not
met, as on val 0.282 and in P074).

### Cloud-base height (4,383 test pairs, 2,965 cloudy, 25 days per station)

The training climatology's MAE on these rows is 2,012 m (1,147 m on val): the test weeks hold far more high
cloud (model MAE in the 8–20 km bin 1,923 m).

| Model | Rows | MAE m [CI] | Bias m | Skill [CI] | ±500 m | CRPS | 80 % raw [CI] | 80 % / 95 % calibrated |
|---|---|---|---|---|---|---|---|---|
| base, with geometry | all | 872 [679, 1055] | −106 | 0.566 [0.476, 0.663] | 0.546 | 0.035 | 0.878 [0.847, 0.904] | 0.720 / 0.918 |
| base | OLDLR | 933 [744, 1127] | −46 | 0.536 [0.440, 0.630] | 0.531 | 0.038 | 0.868 | 0.698 / 0.911 |
| base | WESTE | 813 [601, 1013] | −164 | 0.596 [0.496, 0.701] | 0.561 | 0.032 | 0.887 | 0.742 / 0.925 |
| base, geometry zeroed (H4) | all | 875 [680, 1050] | −12 | 0.565 [0.478, 0.662] | 0.546 | 0.035 | 0.905 | 0.764 / 0.943 |
| OLDLR-only (WESTE never seen) | WESTE | 1,134 [835, 1456] | −575 | 0.436 [0.276, 0.585] | 0.496 | 0.049 | 0.799 | 0.605 / 0.855 |
| ridge on frozen features (P058, refitted) | all | 1,096 [852, 1365] | −285 | 0.455 | 0.470 | — | — | — |

Cloud-presence balanced accuracy 0.915 (base), 0.731 (OLDLR-only on WESTE).

### B0268, the locked day 2026-10-20 (120 frames; model against each rater, averaged; ceiling étage set 0.911, oktas ±1 0.771)

| Graph | Étage set exact | Oktas within ±1 | Okta bias | Top genus in the rater's set | Étage F1 | OOD flagged | Against the pair's majority, frame bootstrap |
|---|---|---|---|---|---|---|---|
| `stratia-1.0.0-demo` (zero-shot) | 0.000 | 0.017 | +3.46 | 0.121 | 0.217 | 0.508 | étage set 0.000, oktas ±1 0.000 |
| `stratia-1.0.1-demo` (adapted on 18 / 19 / 21 Oct) | 0.463 (0.508 of the ceiling) | 0.671 (0.870 of the ceiling) | −0.50 | 1.000 | 0.821 | 0.000 | étage set 0.458 [0.375, 0.546], oktas ±1 0.900 [0.842, 0.950] |

Sky-parse cloud fraction against the raters' mean cover: correlation 0.776, MAE 0.079 (both graphs; dense heads
unchanged). Not evaluated (no model exists): `lodo/swim/test`, `held_out_station/{same_days, disjoint_days}/test`.

`tests/test_locked_test.py` (cluster bootstrap, calibrated coverage, headline table, the refusal to read test rows
without `--final`); full suite, ruff and `scripts/check_test_lock.py` green at the commit.

## Exit criteria
- [x] Every lock entry evaluated or explicitly marked not evaluated, with the model and metric pre-registered above
      (three entries have no model: SWIM LODO, the two held-out-station variants).
- [x] Bootstrap 95 % CIs on every headline number (per fold / per source for the two C1 means; the Montenegro and
      Eye2Sky in-domain test sets hold two clusters each, so their cluster CIs are reported but not informative).
- [x] Results JSON committed with its SHA-256; the final-evaluation reasons logged (two entries).
- [ ] One run — two evaluations: the first stopped on a bug before anything was saved or read beyond one OOD
      line; recorded above and in the log, and the paper says so.

## Fit & data-risk notes
- Validation to test: the C1 in-domain clause turned from +0.003 (val) to −0.027: the MGCD genus F1 is 0.821 against
  the ViT-L probe's 0.918 (the model's CI [0.706, 0.997] is wide: 86 same-scene units), while on CCSN the model wins
  (0.591 against 0.549). The LODO mean moved from −0.035 to +0.011 against the probe. Neither reaches the
  pre-registered five points; the paper's C1 statement is a negative result with intervals.
- C2 fell from above the human ceiling on val (1.047 / 1.027) to 0.926 / 0.963 of it on test, above the 0.90 bar;
  the test set is two temporal blocks, so no honest interval exists beyond the two-block spread.
- CBH: the test weeks are a different sky (training climatology MAE 2,012 m against 1,147 m on val, high cloud
  common): the model's MAE rises from 566 to 872 m while its skill rises (0.566 against 0.506) and every skill
  criterion holds with a wide margin. The P089 conformal radius, fitted on val, under-covers on these weeks (0.720
  for the nominal 0.80; the raw 0.1–0.9 quantile interval covers 0.878): split-conformal coverage assumes
  exchangeable weeks and these are not. The 95 % interval covers 0.918.
- Geometry zeroed changes the CBH numbers by a few metres (H4 on test: the ray map carries no height
  information for this head, as P075 found on val).
- Layers: test layer mIoU 0.668 against 0.558 on val (164 Almería images; CI [0.60, 0.74]).
- The OOD threshold set on all validation rows flags 43 % of CCSN test images and 19 % of SWIM: the score reads
  camera identity (P068) and those sources are many cameras. The B0268 day scores AUROC 0.939 against them.
- B0268: the adapted heads carry to the unseen day on cover (0.671 within ±1, bias −0.5 okta) and genus (1.000),
  and half-way on exact étage sets (0.463; the held-out sweep of P084 had predicted 0.274 on average over draws);
  the zero-shot model is at the floor (0.000 / 0.017, bias +3.5 oktas) and flags half the frames as OOD.
- Seeds: three for the in-domain heads only (std ≤ 0.023 on every metric); one for the LODO, dense and CBH models.

## Deviations from plan & why
1. Two evaluations, not one: the first crashed on a bug that the validation rehearsal could not expose. Nothing
   beyond one OOD line was read from it and nothing changed between the two; both are logged.
2. "Mean ± std over 3 seeds" (research design §4) holds for the in-domain global heads only: the CBH seeds 1 and
   2 kept no checkpoint; the LODO and dense models were trained once.
3. The baseline references on test are refits (probe C and ridge alpha chosen on val, as in P054 / P058); the
   P058 ceilometer-supervised CNN kept no checkpoint, so the C3 MAE comparison uses the ridge (its 1,096 m on test
   against 724 m for the CNN on val: not comparable rows).
4. The in-domain probe reference was amended from "ViT-B/16 for both clauses" to the per-protocol bests Gate G4
   used, during the validation rehearsal and before any test row was read.
5. Three lock entries have no model to evaluate (SWIM LODO, held-out station).
6. Montenegro's and Eye2Sky's in-domain test sets hold two clusters each; their cluster-bootstrap CIs are
   reported as computed and flagged as uninformative.

## Next phase
P091 — Ablations (on validation data; the test numbers above are final).
