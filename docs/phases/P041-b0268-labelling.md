# P041 — B0268 labelling protocol

Status: DONE     Date: 2026-10-22     Commit: (this commit)

## Objective
Label at least 300 frames from the owner's Arducam B0268 with two annotators: image-level genus set, étage and
oktas for all of them, and dense sky / cloud / glare / obstruction masks for 100; report the agreement.

The image-level half is done. The dense-mask half is not done (see Exit criteria).

## Inputs / dependencies
CloudScope sky-logger captures with JSON sidecars (P019 ingest, `scripts/build_manifest.py`); the WMO ontology
(P033) and the dataset mapping rules (P034); the agreement machinery of P035 (`stratia/labels/agreement.py`); the
few-shot B0268 protocol of `docs/splits.md` (P037) with the split generator (P038) and the test lock (P039).
Data: 393 logger frames under `DATA/B0268/BLR01/<day>/` with sidecars and the 25 legacy frames of
`DATA/My Sample/` (EXIF only, no labels); four label sheets under `DATA/B0268/labels/`. Manifest after this phase:
52,425 rows, sample-id SHA-256 prefix `c41841f43cc488d7`. Config: `configs/splits.yaml` version 2, seed 0.

## Work log
1. **Frames.** The B0268 (4656 x 3496, 105 deg fisheye, zenith mount, site BLR01 at 12.97 N 77.59 E, 920 m) was
   logged by CloudScope's sky logger at a 30 s cadence over four afternoons. The 2026-10-21 session was paused for
   rain; frames 390-393 are after the pause.

   | Day | Frames | Sun elevation (deg) | Sky |
   |---|---|---|---|
   | 2026-10-18 | 120 | 15-29 | altocumulus field with cumulus puffs, 4-7 oktas |
   | 2026-10-19 | 120 | 25-39 | blue with isolated cumulus and cirrus, 2-6 oktas |
   | 2026-10-20 | 120 | 27-40 | cumulus building under a cirrus / cirrostratus veil, Sun's disc in frame, 3-7 oktas |
   | 2026-10-21 | 33 | 49-52 | hazy, about 1 okta, then a dark towering mass (Cb or Cu congestus) after the pause |

2. **Sheet design.** One sheet per annotator, one row per frame, columns usable (yes/no), oktas 0-9, low_cloud
   {none, Cu, Sc, St, Cb, Cu+Sc, unsure}, mid_cloud {none, Ac, As, Ns, Ac+As, unsure}, high_cloud {none, Ci, Cs,
   Cc, Ci+Cs, contrail, unsure}, lowest_base_guess {low, mid, high, none}, sun_glare {none, mild, strong},
   obstruction {none, raindrops, dirt, object}, notes. The guide tab of `DATA/B0268/labels/B0268_labels_A.xlsx`
   is authoritative. Annotators worked from 2x2 contact sheets of the frames.
3. **Annotators.** Four sheets exist, all produced by language-model annotators from the same contact sheets,
   independently and without seeing one another: A = Claude Fable 5, B = Claude Sonnet 5.5, C = Claude Fable 5.1,
   D = Claude Opus 5.5. The owner's and a friend's human labels were lost in a disk failure (2026-10-22).
   Owner decision (2026-10-22): proceed with A and B as the P041 annotator pair; C and D are a sensitivity set.
   The plan (`docs/PLAN.md`, P041) requires two human labellers. That guard is not met. Human labels on a subset
   remain an owner item.
4. **Fixtures.** CSV copies of the sheets are committed as `tests/fixtures/b0268_labels/B0268_labels_{A,B,C,D}.csv`
   with the frame list `tests/fixtures/b0268_labels/B0268_frames.csv`, so the loader and the derivation are tested
   without the data folder.
5. **Derivation rules** (`stratia/labels/b0268_labels.py`, the sheet's guide tab is authoritative):
   `usable = no` leaves every other column empty and counts only for the usable share. A genus combination names
   co-present genera (Cu+Sc = {Cu, Sc}). `none` = no cloud of that étage. `unsure` = cloud of that étage present,
   genus unknown: the étage counts as present and the rater's genus set is unknown, so it is left out of the genus
   shares and never guessed. `contrail` in the high column is the extra class, not a genus; the high étage counts as
   present. oktas 9 = obscured (no cover estimate). oktas 0 with every column `none` = clear. Per frame the module
   writes the P035 soft targets (étage and genus shares, oktas distribution, obscured and cloud shares) plus the
   clear, contrail, glare, obstruction and usable shares, the day and `image_file` (`B0268/<file>`). `majority_sets`
   turns the shares into an étage set and a genus set at threshold 0.5.
6. **Targets and ceiling.** `scripts/b0268_labels.py` writes `data/b0268_targets.parquet` (the pair),
   `data/b0268_targets_all.parquet` (A-D), `data/b0268_ceiling.parquet` (column `raters` = `AB` / `ABCD`) and the
   report `docs/data/b0268_labels.md`. The B0268 ceiling for the C2-style comparison is the pair's agreement: étage
   set 0.911, oktas within ±1 0.771. P083 reads it as `ceiling`.
7. **Labels.** `stratia/labels/mapping.py`: `map_b0268_row` and `apply_mapping(..., b0268_majority=None)`;
   `scripts/map_labels.py` passes `majority_sets(targets)`. Result: 393 b0268 rows with `label_source`
   "rater majority" (genus set, étage set and cloud presence from the pair's majority), the 25 legacy rows "none".
   Old label rows are unchanged (`docs/data/label_mapping.md`).
8. **Manifest.** `scripts/build_manifest.py`: `b0268()` ingests `My Sample` and `B0268/BLR01`; the sidecars give
   UTC, site, Sun and pointing; width and height fall back to the ingest. `EYE2SKY_IMAGE_STATIONS = ("AURIC",
   "BARSE")` keeps the cloud-base stations OLDLR / WESTE out of the manifest as before. Reproducibility check: the
   new manifest is the 52,032 locked rows in the same order (lock-style SHA-256 prefix `464078b6d033dd39`
   reproduced) plus 393 appended rows (sample_id `b0268:B0268/BLR01/<day>/<file>.jpg`, Sun from pvlib).
   `configs/datasets.yaml` b0268: `local_paths: {legacy: "My Sample", logger: "B0268"}`, expected_images 418,
   status partial.
9. **Splits.** `stratia/data/splits.py` `few_shot_target(manifest, labelled, config)`: the locked test days are the
   last `test_days` (1) calendar days with at least `min_frames_per_test_day` (100) labelled frames, a rule fixed
   before any number was looked at. That gives 2026-10-20. The other three days (10-18, 10-19, 10-21) are
   adaptation days. For each k in `shots_days` [0, 1, 2], every combination of k adaptation days (at most `repeats`
   5) is a draw with roles `train` (the k days), `val` (the other adaptation days) and `test` (the locked day).
   k = 3 would leave nothing held out, so k = 2 replaces the plan's k = 3. `verify_few_shot` checks every draw;
   `align_target_with_few_shot` makes b0268's in-domain split mirror the protocol (test = the locked day, everything
   else train); `lodo_folds` excludes the few-shot target from LODO training; `split_hashes(..., few)` hashes the
   draws. `scripts/make_splits.py` writes `data/splits/few_shot_target.parquet` (draw, shots, sample_id, day, role)
   and a note in `docs/data/splits_report.md`. `configs/splits.yaml` `few_shot_target` carries the reason.
10. **Lock.** `stratia/data/test_lock.py` `TEST_FILES` gains `few_shot_target: ("draw", "role", "test")`.
    `test_lock.json` was re-locked with `--force`: 23 test sets, 73,376 ids, manifest prefix `c41841f43cc488d7`.
    Re-lock reason: the manifest grew by the 393 labelled frames (its hash changed) and the lock needed the new test
    sets `in_domain/b0268/test` and `few_shot_target/*/test`. Reproducibility check: every previously locked test
    hash is unchanged against the saved `data/splits/hashes_before_P041.json`. The LODO train files lost the 25
    legacy b0268 rows; the target was never a training source in practice (`stratia.train.data.GLOBAL_SOURCES`
    excludes it). `scripts/check_test_lock.py` passes.
11. **Tests.** `tests/test_b0268_labels.py` (5: derivation rules, soft targets and majority sets, the two-rater
    agreement table, the fixture sheets load and validate, bad values rejected) and `tests/test_few_shot_target.py`
    (2: the draws lock the last full day and hold the rest out; majority rows map to labels).

## Verification
- Commands (2026-10-22): `python scripts/b0268_labels.py`, `python scripts/build_manifest.py`,
  `python scripts/map_labels.py`, `python scripts/make_splits.py`, `python scripts/lock_test_sets.py --force`,
  `python scripts/check_test_lock.py`. Report: `docs/data/b0268_labels.md`. 393 frames, 2 annotators, every frame
  usable, none obscured.

| Quantity (pair A + B) | Level | alpha | Agreement |
|---|---|---|---|
| oktas 0-8, within ±1 okta | interval | 0.808 | 0.771 |
| oktas 0-8, exact | interval | 0.808 | 0.430 |
| **étage set** (exact) | nominal | 0.845 | **0.911** |
| étage low / mid / high present | nominal | 0.349 / 1.000 / 0.909 | 0.949 / 1.000 / 0.957 |
| genus set (exact, `unsure` excluded) | nominal | 0.673 | 0.789 |
| genus Ci / Ac / Cu / Cb / Cs present | nominal | 0.909 / 1.000 / 0.474 / 1.000 / -0.064 | 0.957 / 1.000 / 0.949 / 1.000 / 0.878 |
| raw low_cloud / mid_cloud / high_cloud | nominal | 0.478 / 1.000 / 0.696 | 0.949 / 1.000 / 0.835 |
| raw lowest_base_guess | nominal | 0.349 | 0.949 |
| raw sun_glare | ordinal | 0.487 | 0.679 |
| raw obstruction | nominal | 0.623 | 0.982 |

- **B0268 ceiling for the C2-style comparison:** étage set 0.911, oktas within ±1 okta 0.771 (P083 uses these as
  `ceiling`). Montenegro's human ceiling (P035) is 0.852 and 0.873.
- Four-way alpha (A-D, sensitivity, `docs/data/b0268_labels.md`, second table): oktas 0.834, étage set 0.748,
  genus set 0.357, raw low_cloud 0.491, mid_cloud 1.000, high_cloud 0.279, sun_glare 0.675 (ordinal), obstruction
  0.501. High cloud is the contested column: none against Ci on day 1, Ci against Ci+Cs against contrail on day 3.
- Lock: 23 test sets, 73,376 ids; every pre-existing test hash unchanged; `scripts/check_test_lock.py` passes.
- Tests: 7 new, passing on 2026-10-22; `ruff check` clean on the touched files.

## Exit criteria
- [x] >= 300 frames labelled by 2 people: 393 frames by 2 independent annotators. Caveat: both are language-model
  annotators (A = Claude Fable 5, B = Claude Sonnet 5.5), not people. The plan's human guard is NOT met. Human labels
  on a subset remain an owner item.
- [x] Agreement reported (Krippendorff's alpha per target with `stratia/labels/agreement.py`, as for Montenegro in
  P035): `docs/data/b0268_labels.md`.
- [ ] Dense sky / cloud / glare / obstruction masks for 100 frames (part of the objective, not a PLAN exit
  criterion): not done.

## Fit & data-risk notes
- **Single-annotator bias is partly addressed.** The pair are two independent raters, so per-frame targets are
  shares, not one opinion, and the ceiling is measured rather than assumed. It is not addressed the way the plan
  asked: no human labelled a frame.
- **Model-annotator bias is a new risk.** All four sheets come from language models of one family, labelling from
  contact sheets rather than full-resolution frames. Their shared blind spots are not measured by inter-rater
  agreement. Every B0268 number in the paper carries this caveat until a human labels a subset.
- **Four days, not six.** The plan asks for at least six distinct days. With four, one day is locked as test and
  three are adaptation days, so the few-shot draws stop at k = 2 and the held-out val set per draw is one or two
  days. Day 4 is small (33 frames, 49-52 deg Sun, one weather regime), so draws that hold it out alone
  (`k2_d0`) measure on 33 frames.
- **Glare and high cloud are the contested columns.** sun_glare alpha 0.487 for the pair (0.61 four-way), high_cloud
  0.696 (0.28 four-way), genus Cs -0.064. Glare is "strong" on days 2-4, and the Sun's disc is in the frame on
  day 3. Scores on the high étage and on the glare mask will be read against these numbers, not as hard truth.
- **Cover and the étage set are the usable signals**, as for Montenegro: oktas interval alpha 0.808, étage set
  alpha 0.845. Genus stays a soft target (share of raters) and the model is scored on genus-in-the-raters'-union.
- Targets are per-frame distributions; training and evaluation use them as such (P030, P035).

## Deviations from plan & why
- Annotators are language models, not two humans: the human labels were lost in a disk failure on 2026-10-22 and
  the owner decided to proceed with sheets A and B so that P083 and P084 can run before the AI-day demo.
- A spreadsheet with drop-down columns replaced the plan's Label Studio / CVAT set-up; the dense-mask half of the
  objective was not started.
- Four labelled afternoons instead of six or more; k = 2 replaces k = 3 in the few-shot protocol.
- Closed on 2026-10-22 instead of before Gate G2 (2026-10-10): the frames did not exist then (P042 was signed
  conditionally on this phase).
- The test lock was re-locked with `--force`; every pre-existing test hash is unchanged (`data/splits/hashes_before_P041.json`).

## Next phase
P083: B0268 zero-shot evaluation on the adaptation days (draw `k0_d0`, role `val`; the locked day is never
evaluated).
