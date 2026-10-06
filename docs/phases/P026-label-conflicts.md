# P026 — Label-conflict audit

Status: DONE     Date: 2026-10-06     Commit: (this commit)

## Objective
List every case of the same picture carrying different labels (class labels and segmentation masks), decide what
later phases do with each kind of conflict, and measure how consistent each dataset's labels are on near-identical
pictures. Flag, do not fix.

## Inputs / dependencies
P021 manifest (`source_label`, `seg_file`, Montenegro rater distributions); P024 exact-duplicate groups; P025 copy
pairs, same-scene pairs, the exhaustive SWIM-family table and the dihedral pHashes (which flip or rotation lays one
copy like the other).

## Work log
1. `stratia/data/label_conflicts.py`: pairs from duplicate groups; merging of the three kinds of "same picture"
   (exact > copy > same scene); native-label comparison with official-split crossing; recovery of the dihedral
   variant from the P025 hashes; mask alignment (flip or rotate, nearest-neighbour resize) with pixel agreement and
   class IoU; rater-agreement summary (majority share from the Montenegro distributions); mask contact sheets; report.
2. `scripts/audit_label_conflicts.py`: reads the P024/P025 tables; writes `data/label_conflicts.parquet` (every
   compared pair with both labels, kind, variant, time gap, flags), `data/mask_agreement.parquet`,
   `docs/data/label_conflicts_report.md` and the sheets `docs/data/figures/label_conflict_*.jpg`, `mask_*.jpg`.
3. Tests (`tests/test_label_conflicts.py`, 7): pair listing and kind precedence; conflict and split-crossing flags
   with their summaries; recovery of the dihedral variant and exact mask agreement after alignment (the wrong
   variant fails, another mask size works); ignored pixels; mask comparison from files; rater summary; report text.
4. Reviewed the sheets and fixed the policy (Verification).

## Verification
- Run: `python scripts/audit_label_conflicts.py` (39 s). Pairs: 390 exact, 4,217 copies, 41,671 same scene (pHash
  copies inside a camera's own time series count as same scene, see Deviations).
- **Class labels** (same-dataset pairs with a native label on both sides):

| Dataset | Kind | Pairs | Conflicts | Rate | Images involved |
|---|---|---|---|---|---|
| CCSN | copy | 129 | 105 | 81.4 % | 196 |
| CCSN | exact | 17 | 3 | 17.6 % | 6 |
| CCSN | same scene | 25 | 22 | 88.0 % | 44 |
| MGCD | same scene | 36,532 | 222 | 0.6 % | 163 |
| Montenegro | same scene | 2,077 | 595 | 28.6 % | 419 |
| SWIMCAT | exact, copy, same scene | 95 | 0 | 0 % | 0 |

- **Reviewed on the contact sheets:** the CCSN copy pairs are the same photograph, re-encoded, filed under two
  genera (Ns/St 17 pairs, Cb/Cu 10, Cc/Cs 9, Ac/Cc 7, Sc/St 6, Cs/Sc 5, ...); 242 CCSN images (9.5 %) are in some
  conflict, 200 (7.9 %) in an exact or copy conflict. The three exact pairs are ambiguous skies (Cc with a Cs veil;
  Ac/As twice) where both labels are defensible. MGCD labels flip between consecutive frames in 0.6 % of
  same-scene pairs (altocumulus/mixed 74, cumulonimbus/cumulus 42, altocumulus/stratocumulus 31): the labels were
  evidently given per sequence. Montenegro disagreements are look-alike frames of near-featureless skies, mostly
  days apart (median gap one day; 24.5 % within an hour), flipping Clear/High clouds (216 pairs) or Clear/Low
  clouds (63): thin high cloud and haze are where the raters disagree.
- **Published splits:** 701 MGCD same-scene pairs (452 images) sit on both sides of the official train/test split,
  75 of the 222 label conflicts among them; 193 Almería same-scene pairs cross train/val/test.
- **Masks** (exact duplicates and copies of segmentation images; the mask of i flipped or rotated like its picture):

| Datasets | Kind | Pairs | Median agreement | 10th percentile | Minimum | Below 0.95 |
|---|---|---|---|---|---|---|
| SWIMSEG → SWINySEG | copy | 2,213 | 0.997 | 0.991 | 0.761 | 66 |
| SWINSEG → SWINySEG | copy | 248 | 0.995 | 0.988 | 0.873 | 5 |
| SWINySEG | copy | 1,620 | 0.998 | 0.980 | 0.762 | 51 |
| SWINySEG | exact | 326 | 1.000 | 0.997 | 0.764 | 20 |
| SWIMSEG | exact | 26 | 0.949 | 0.889 | 0.764 | 13 |
| SHWIMSEG | exact / copy | 3 / 5 | 0.960 / 1.000 | | | 0 |
| Almería | exact | 1 | 0.699 | | | 1 |

  Reviewed: the augmentation that built SWINySEG carried the masks with the images (copies agree on 99.7 % of
  pixels; the resize from 600 to 300 px accounts for most of the rest). The conflicts come from SWIMSEG itself: its
  26 pixel-identical pairs were annotated twice and agree on 95 % of pixels at the median, 76 % at worst (`0782` vs
  `0792`: the cloud boundary drawn differently in haze), and every SWINySEG copy inherits one of the two masks.
  The Almería pair `asi_683_170928141100` / `asi_684_171001151700` is one frame stored under two timestamps with
  incompatible masks (detailed clouds vs all cloud): a construction error. **Rule:** a pair below 0.95 agreement is
  a mask conflict (156 pairs); 0.95 lies well below the 10th percentile of the construction copies (0.98–0.99).
- **Raters** (Montenegro, 5–9 raters per image): median majority share oktas 0.75, h 0.60, CL 0.75, CM 0.67,
  CH 0.75; no majority (<= 50 %) for 25 % of images on oktas and 39 % on h; unanimous on 13–31 %.
- Tests: 7 in `tests/test_label_conflicts.py`; `python -m pytest -q`: 66 passed; `ruff check .` clean.

## Exit criteria
- [x] Conflict list reviewed: `data/label_conflicts.parquet` (46,278 compared pairs, `conflict` flag) and
  `data/mask_agreement.parquet` (4,442 pairs, `conflict` flag); every exact and copy conflict is listed in
  `docs/data/label_conflicts_report.md`; sheets reviewed as above. Nothing auto-fixed; the policy is in the report.

## Fit & data-risk notes
- **CCSN genus labels are noisy at the whole-image level:** 7.9 % of its images carry a second genus on a copy of
  themselves. Single-label genus accuracy on CCSN therefore has a ceiling; STRATIA's `genus_set` (several genera
  per image, P033) absorbs the ambiguous skies, and the étage-level evaluation is the one to trust where the genera
  are neighbours (Cc/Cs, Ac/As, Cb/Cu, Sc/St). Some conflicts cross étages (Ci/St, Cs/Sc, Cb/Ci): one side is
  wrong, and the policy keeps those pictures out of every test split.
- **MGCD is labelled per sequence**, so a model evaluated on a random split of MGCD can score by recognising the
  sequence; the group-and-time splits (P037, P038) remove this, and the 452 images already on both sides of the
  official split are the leakage that the published numbers include.
- **Cloud height from a single webcam frame is hard for people too:** Montenegro's raters reach no majority on
  the height code for 39 % of images. A height "label" from such images is a distribution, not a class; for Paper B
  the ceilometer, not a rater, is the reference, and evaluation on Montenegro should score against the rater
  distribution (P034).
- **Segmentation ceiling:** the same SWIMSEG image annotated twice agrees on about 95 % of pixels; a result above
  that on a SWIM test set measures memorisation of one annotator, not segmentation quality. Report it in the data
  card and whenever published SWIMSEG/SWINySEG numbers are compared.
- Montenegro frames carry a burned-in timestamp (top right) and a logo (bottom left): shortcut material for P029.

## Deviations from plan & why
- pHash copies inside time-series datasets (MGCD, Montenegro, Almería, Eye2Sky) are classed as same scene, not as
  copies: P025 showed that the pHash of a smooth sky carries little information, and the sheets confirmed that
  these pairs are frames from different times (Montenegro: median gap one day). A label difference between them is
  a consistency measure, not a labelling error.
- Masks were compared as labels too (the plan says "duplicates with different labels"; for the segmentation
  datasets the label is the mask), and rater agreement was summarised where it exists, because both are direct
  measures of label noise that the data card needs.

## Next phase
P027: Temporal autocorrelation.
