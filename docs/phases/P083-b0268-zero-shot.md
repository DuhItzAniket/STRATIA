# P083 — B0268 zero-shot evaluation

Status: DONE     Date: 2026-10-23     Commit: (this commit)

The method and the frames were fixed in this document before any number existed (2026-10-23, 02:30 IST, while
the machine was busy); the run followed at 12:07 UTC without a change to the method, except that the primary truth
became each rater separately (work log 1), a decision taken from a code review before the run.

## Objective
Score the exported demo model on the labelled B0268 frames of the adaptation days, before any adaptation, against
the P041 rater targets. Exit (plan, Stage I): an honest out-of-distribution number for the owner's camera, with
the raters' own agreement as the ceiling, as criterion C2 does for Montenegro.

## Inputs / dependencies
- P041 labels: `data/b0268_targets.parquet` (pair A+B), `data/b0268_targets_all.parquet` (A to D, sensitivity),
  `data/b0268_ceiling.parquet` (column `raters` = "AB" / "ABCD"); `docs/data/b0268_labels.md`. The annotators are
  language models, not the two human labellers of the plan (see the P041 section of `docs/research_log.md`).
- Few-shot protocol (P041, `stratia/data/splits.py`): `data/splits/few_shot_target.parquet` (draw, shots,
  sample_id, day, role); `test_lock.json` re-locked with the B0268 days.
- Manifest with the 393 logged B0268 rows appended (sidecars give UTC, site, Sun azimuth and zenith, pointing).
- Model: `models/stratia-1.0.0-demo/stratia.onnx` with `model_card.json` (P096; Gate G4 heads of P082 with the
  P089 calibration folded in; OOD threshold 1.4569; input 512 x 512).
- Code: `stratia/train/b0268_eval.py` (`majority_targets`, `score`, `per_day`), `scripts/eval_b0268.py`.
- Geometry: `camera_for("b0268", 4656, 3496)` (nominal fisheye), `Pose(R=I, source="declared", calib_id="nominal")`,
  `ray_map` and `meta_vector` from `stratia/geometry/raymap.py`.

## Work log
1. `stratia/train/b0268_eval.py`. Primary truth: each rater's own labels (`rater_targets` from
   `data/b0268_derived.parquet`; `score_against_raters` scores the model against every rater and averages, the way
   the ceiling is measured: one rater against the other). With two raters a "majority" at threshold 0.5 is their
   union and the argmax okta of a 1:1 vote is the lower one, which a review found biased the truth low on 130 of
   the 273 frames; scoring per rater breaks no tie. Secondary truth: `majority_targets` (étages and genera named by
   at least one rater, the lower okta on ties, the raters' mean okta / 8 as cover, clear and strong-glare majorities)
   feeds `score` / `per_day` for the sky-parse, glare and OOD columns.
2. `scripts/eval_b0268.py`: selects the manifest rows of draw `k0_d0`, role `val`; decodes the original frames,
   resizes full-frame to 512 x 512 (INTER_AREA, no crop, no mask zeroing), scales to [0, 1], ImageNet mean / std;
   builds the contract ray map (`out_size=512, patch=16`, 4 x 32 x 32, pooled to 8 x 8 inside the graph) from the
   nominal camera, the zenith pose and the manifest Sun, and `meta_vector(sun_zenith_deg)`; runs the ONNX graph on
   `CPUExecutionProvider` in batches of 8; reads the seven outputs (genus_logits, etage_logits, oktas_probs,
   sky_parse_logits [n, 4, 128, 128] = invalid / sky / cloud / sun_glare, layer_logits, cbh_probs [n, 25],
   ood_score [n, 1]); derives the cloud and glare fractions over the valid sky pixels; scores against each rater of
   the pair with the "AB" ceiling, and against all four raters with the "ABCD" ceiling as a sensitivity row.
   `--no-geometry` zeroes the ray map and the meta vector (unknown geometry, as the contract allows) for an H4 check.
3. Outputs: `data/b0268_zero_shot.parquet` (per frame: truth, étage probabilities, okta argmax, top genus, cloud
   and glare fractions, OOD score, CBH median, p(cloud overhead)), `data/b0268_zero_shot_scores.parquet` (truth
   "per-rater mean AB", "majority AB", "per-rater mean ABCD"), `docs/data/b0268_zero_shot.md`; a P083 run in
   `runs/registry.csv` (name `b0268_zero_shot`; the H4 check `b0268_zero_shot_nogeo`).
4. Runs (2026-10-23): `python scripts/eval_b0268.py` -> run `20261023T120741Z_b0268_zero_shot_6a9765a3d2`, 273
   frames, 221 ms per frame on the laptop CPU; `python scripts/eval_b0268.py --no-geometry` -> run
   `20261023T121334Z_b0268_zero_shot_nogeo_a91d5de522`. The same heads on cached features (P084 draw `k0_d0`,
   run `20261023T121336Z_adapt_b0268_k0_d0_cfebb7e772`) agree with the ONNX path within the double resize and the
   8 x 8 ray map: étage set 0.040, oktas within ±1 0.359.

### Frames evaluated
Draw `k0_d0` (zero shots): every adaptation day is `val`; the locked test day is excluded and stays for P090.

| Day | Role | Frames | Sun elevation | Sky (raters' notes) |
|---|---|---|---|---|
| 2026-10-18 | val | 120 | 15 to 29 deg | altocumulus field with cumulus puffs, 4 to 7 oktas |
| 2026-10-19 | val | 120 | 25 to 39 deg | blue sky, isolated cumulus and cirrus, 2 to 6 oktas |
| 2026-10-21 | val | 33 | 49 to 52 deg | hazy, about 1 okta; a dark towering mass after the rain pause |
| 2026-10-20 | test (locked) | 120 | 27 to 40 deg | not evaluated |

273 frames are scored. The raters call the glare "strong" on days 2 to 4. The 25 legacy frames have no labels and
are not part of the protocol.

### Metrics (`score`)

| Field | Meaning |
|---|---|
| `etage_set_agreement` | exact étage set (sigmoid > 0.5 per étage) equals the majority set |
| `etage_f1`, `etage_<e>_acc`, `_pred_rate`, `_true_rate` | macro-F1 over the three étages; per-étage accuracy and rates |
| `oktas_within1`, `oktas_mae`, `oktas_bias` | argmax okta against the majority okta |
| `genus_top1_in_union`, `genus_top1_in_majority` | top genus (first ten contract classes) named by any rater / by the majority |
| `cloud_fraction_mae_vs_cover`, `_bias_`, `_corr_` | sky-parse cloud share of the valid sky against the raters' mean cover (okta / 8) |
| `cloud_fraction_on_le1_okta`, `n_le1_okta` | cloud share on frames the raters put at 0 or 1 okta (collapse check) |
| `glare_fraction_strong_frames`, `_other_frames` | sky-parse glare share on strong-glare frames and on the others |
| `ood_mean`, `ood_p90`, `ood_flagged_share` | OOD score; share above the card's threshold 1.4569 |
| `c2_etage_ratio`, `c2_oktas_ratio` | model agreement divided by the raters' ceiling |

### Ceiling (pair A+B, `docs/data/b0268_labels.md`)

| Quantity | Raters' leave-one-out agreement | Montenegro human ceiling (P035) |
|---|---|---|
| étage set, exact | 0.911 | 0.852 |
| oktas within ±1 | 0.771 | 0.873 |

The C2 bar on Montenegro is 90 % of the ceiling. The same ratio is reported here as `c2_*_ratio`; it is not a
pre-registered criterion for B0268.

### Supporting criteria (`docs/research_design.md` §5)
- No sky-parsing collapse on B0268: predicted cloud fraction on clear frames < 20 %. Measured as
  `cloud_fraction_on_le1_okta`; only part of 2026-10-21 is at 0 or 1 okta, so the check rests on few frames.
- `ood_score` AUROC >= 0.80: not computable here, the run has no in-distribution set; `ood_flagged_share` and
  `ood_mean` are reported instead. The P096 preview flagged one of four frames; the CloudScope replay of
  2026-10-21 flagged none of 700.
- H4 (geometry inputs help transfer): an ablation, out of scope for this phase.

## Verification
Run `20261023T120741Z_b0268_zero_shot_6a9765a3d2` (`docs/data/b0268_zero_shot.md`). Model against each rater of
the pair, averaged; ceiling = the raters' agreement with each other (étage set 0.911, oktas within ±1 0.771).

| Day | Frames | Étage set exact | c2 ratio | Oktas within ±1 | c2 ratio | Okta MAE | Okta bias | Top genus in the rater's set |
|---|---|---|---|---|---|---|---|---|
| all | 273 | 0.031 (A 0.033, B 0.029) | 0.034 | 0.278 (A 0.245, B 0.311) | 0.361 | 2.61 | +2.40 | 0.300 |
| 2026-10-18 | 120 | 0.046 | 0.050 | 0.492 | 0.638 | 1.62 | +1.13 | 0.633 |
| 2026-10-19 | 120 | 0.000 | 0.000 | 0.075 | 0.097 | 3.14 | +3.14 | 0.008 |
| 2026-10-21 | 33 | 0.091 | 0.100 | 0.242 | 0.314 | 4.27 | +4.27 | 0.152 |

Against all four raters (A to D, averaged): étage set 0.037, oktas within ±1 0.291, MAE 2.53, bias +2.33, top genus
in the rater's set 0.333. The sensitivity set does not change the picture.

What the heads do (secondary table, per-frame parquet): okta argmax is 6 on 138 frames and 8 on 130 of 273 (mean
6.9 against the raters' 4.2); the low étage is predicted on 1 % of frames while a rater names it on 98 %; mid on
77 % (raters 45 %), high on 59 % (raters 50 %); top genus Ns on 152 frames, Ac 76, Cs 38, Cu 6, Cc 1. The dense
head is the healthier part: sky-parse cloud fraction against the raters' mean cover has MAE 0.107 and correlation
0.833 over the three days (0.787 on 2026-10-19, 0.830 on 2026-10-21, none within the flat 2026-10-18); mean cloud
fraction 0.69 / 0.47 / 0.43 per day against raters' cover 0.77 / 0.35 / 0.22.

| Supporting check | Value | Note |
|---|---|---|
| cloud fraction on the 27 frames at 0 or 1 okta (all on 2026-10-21) | 0.360 | above the 0.20 bar: the hazy, Sun-in-frame sky reads as cloud |
| sky-parse glare share, strong-glare frames / others | 0.000 / 0.000 | the glare class is never predicted on this camera |
| OOD score mean / p90 / share above 1.4569 | 1.127 / 1.417 / 0.084 | 0.000 on 10-18, 0.017 on 10-19, 0.636 on 10-21 |
| H4 check, geometry zeroed (`--no-geometry`) | étage set 0.081, oktas ±1 0.229, bias +2.74, top genus 0.392 | the ray map neither rescues nor breaks the heads |
| CBH head | p(cloud overhead) 0.97 to 0.99, median base 2.7 to 5.1 km | no reference at this site; not scored |

- The locked test day (2026-10-20) was not read by the script: the frames come from `few_shot_target.parquet`
  roles `val` of draw `k0_d0` only; the lock check passes (`scripts/check_test_lock.py`).

## Exit criteria
- [x] Honest OOD number on the labelled B0268 adaptation days, before any adaptation: étage set exact 0.031 (3 %
      of the raters' 0.911), oktas within ±1 0.278 (36 % of 0.771); the heads over-estimate cover by 2.4 oktas and
      miss the low étage almost always. The number is honest and it is bad.
- [x] Collapse check reported: cloud fraction 0.360 on the 27 frames at 0 or 1 okta, above the 0.20 bar.
- [x] The locked test day untouched (P090).

## Fit & data-risk notes
- No training. The frames are scored once with a fixed method; nothing in the script selects a threshold or a
  pose from the labels.
- Truth comes from language-model annotators (pair alpha: oktas 0.808, étage set 0.845, genus set 0.673). Genus is
  scored only against the raters' union and majority, never as a hard label. The four-way high-cloud alpha is 0.279,
  so any genus number on cirrus days is soft.
- Four afternoons, one site, one camera: the per-day rows matter more than the "all" row. 2026-10-21 has 33 frames.
- The previews (P096, CloudScope 2026-10-21) saw a cover over-estimate (oktas 6 to 8 on a mostly blue day) and a
  genus lean to Ns / Cs. The run confirms both through an independent implementation: the failure is the global
  heads' reading of this camera (full-frame fisheye sky with no horizon, haze and the Sun's disc in the frame),
  not the preprocessing or the geometry inputs (the H4 check with zeroed geometry gives the same picture).
- The OOD score does not flag the failure: 8 % of frames above the threshold while the heads are wrong on nearly
  all of them. The detector was fitted on source features; this camera is inside its envelope and outside the heads'.
  The "uncertain" badge of the demo will therefore stay off on most frames; the demo narrative must say so.
- The dense head transfers better than the global heads (cloud fraction correlation 0.83 with the raters' cover):
  for the demo, the sky-parse overlay is the trustworthy element, the cover and genus readouts are not, until P084.
- Night frames are not in the set (afternoon sessions). The night rule asked for by the CloudScope preview is a
  CloudScope item.

## Deviations from plan & why
- "Labelled B0268 val days" are the three adaptation days of draw `k0_d0`, not the val days of a six-day campaign:
  P041 delivered four days. The test-day rule of `configs/splits.yaml` (last day with >= 100 labelled frames) fixes
  2026-10-20 as the locked day.
- The model is the exported ONNX graph (P096) on the CPU, not the PyTorch heads: the deployed and the evaluated
  model are one object, as P090 will do.
- Two human labellers were planned; the annotators are language models (owner decision 2026-10-22).

- The metric table of the draft was written before the run; the primary truth changed from the pair's majority to
  each rater (work log 1) after a code review, before any number was seen.

## Next phase
P084, few-shot adaptation (reduced: heads only on cached features with source replay), then the AI-day demo of
2026-10-30 with the 1.0.0-demo or the 1.0.1-demo weights.
