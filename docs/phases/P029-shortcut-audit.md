# P029 — Shortcut audit

Status: DONE     Date: 2026-10-06     Commit: (this commit)

## Objective
Find what a model could learn instead of clouds (which camera, which dataset, what time of day, and from which
pixels), name each shortcut with its evidence, and fix the mitigation the training and evaluation recipe will carry.

## Inputs / dependencies
P022 image cache; P025 DINOv3 ViT-S/16 CLS features and near-duplicate groups (probe splits by group); P026 (label
structure); P027 (day blocks for probe splits).

## Work log
1. `stratia/data/shortcuts.py`: per-camera mean and standard-deviation images at 224 px and the fixed-structure mask
   (std below half the camera's median, plus everything outside the main field of view, so the changing digits of a
   timestamp in a corner count as not-sky); image variants re-embedded with DINOv3 (`lowres`: 96 px and back;
   `skyonly`: fixed structure painted grey; `fixedonly`: everything else grey); probe splits by day or by
   near-duplicate group; logistic-regression probes with chance and majority baselines; figure; report.
2. `scripts/shortcut_audit.py`: statistics for the seven cameras with >= 500 frames; variants (`lowres` for all 52,032
   images, `skyonly` and `fixedonly` for Eye2Sky and Montenegro); 25 probes; writes `data/shortcut_probes.parquet`,
   `docs/data/shortcuts_report.md`, `docs/data/figures/shortcut_camera_statistics.jpg`; the camera statistics are
   cached in `cache/features/camera_stats_224.npz` for the loader masks (P040).
3. Tests (`tests/test_shortcuts.py`, 7): statistics and mask find a fixed corner; the mask covers changing digits inside
   a fixed surround; the variants change the right pixels; holdouts keep blocks and groups whole; balanced subsample;
   the probe separates separable data and not noise; figure and report.
4. Reviewed the figure and the probes, wrote the shortcut list (Verification).

## Verification
- Run: `python scripts/shortcut_audit.py` on the RTX 4050: statistics 260 s for 51,815 frames, `lowres` embeddings
  180 s, `skyonly` 93 s and `fixedonly` 58 s for 31,810 frames, probes about two minutes.
- **Fixed structure** (share of the frame): Eye2Sky 36–37 % (fisheye corners, horizon objects: a pole at AURIC, trees
  at BARSE, and a text block with a changing timestamp in the top-left corner), MGCD 25 %, Almería 26 % (corners, a
  mounting arm), Montenegro 0.6 % (timestamp top right, logo bottom left; its mountains and houses change with the
  light and do not count as fixed), SWIM family and CCSN none.
- **Probes** (balanced accuracy; the test rows are days or near-duplicate groups never seen in training):

| What the probe predicts | Features | Balanced accuracy | Chance |
|---|---|---|---|
| Dataset (11, at most 1,500 images each) | original / lowres | 99.7 % / 99.0 % | 9.1 % |
| SWIMSEG vs SWINySEG-day (same camera, often the same pictures) | original / lowres | 99.7 % / 99.3 % | 50 % |
| Eye2Sky station, AURIC vs BARSE (split by day) | original / skyonly / fixedonly / lowres | 99.9 % / **100 %** / 100 % / 99.2 % | 50 % |
| Eye2Sky hour of day (15 hours) | original / skyonly / fixedonly | 28.6 % / 26.5 % / **43.5 %** | 6.7 % |
| Montenegro hour of day (14 hours) | original / skyonly / fixedonly | 31.2 % / 32.7 % / **41.5 %** | 7.1 % |
| Montenegro primary class (5) | original / skyonly / fixedonly | 53.0 % / 54.7 % / **34.9 %** | 20 % |
| MGCD sky type (7) | original / lowres | 85.1 % / 84.0 % | 14.3 % |
| CCSN genus (11) | original / lowres | 46.1 % / 45.9 % | 9.1 % |
| SWIMCAT category (5) | original / lowres | 100 % / 100 % | 20 % |

- Readings:
  1. **The source is written in the features**: 99.7 % for eleven datasets, and even two releases of the same camera
     (SWIMSEG and SWINySEG) are told apart at 99.7 %. Lowering the resolution to 96 px costs under one point:
     resolution and compression are not what carries it.
  2. **The two Eye2Sky stations are told apart perfectly from the sky alone** (fixed structure grey) on days never
     seen in training, and perfectly from the fixed structure alone. Masking corners and horizon objects does not
     remove camera identity: optics, exposure, colour response and the view itself carry it.
  3. **The hour of day is readable**: four times chance from the sky (sun position, brightness), six times chance from
     the fixed structure (illumination of horizon objects and the camera's exposure state; for Montenegro also the
     timestamp digits, which the mask includes).
  4. **Montenegro's text and logo pixels alone predict its class above chance** (34.9 % against 20 %): date and lighting
     leak through the strip. Painting the fixed structure grey slightly improves the class probe (54.7 % against
     53.0 %): the foreground was noise, not signal.
  5. For scale, frozen features with a linear probe reach 85 % on MGCD's seven sky types, 46 % on CCSN's eleven genera
     (label noise, P026) and 100 % on SWIMCAT.
- Tests: 7 in `tests/test_shortcuts.py`; `python -m pytest -q`: 79 passed; `ruff check .` clean.

## Exit criteria
- [x] Shortcut list + mitigations: the last section of `docs/data/shortcuts_report.md` (six shortcuts, each with its
  evidence and the phase that carries the mitigation); the fixed-structure masks are saved for the loader.

## Fit & data-risk notes
- **Camera identity cannot be masked away; it must be handled by protocol.** Leave-one-dataset-out and
  held-out-station evaluation (P037) are the only honest tests of cross-camera generalisation, and a pooled accuracy
  must always come with per-source numbers. Camera-balanced sampling (P030, P039) keeps the model from spending its
  capacity on the majority camera (Eye2Sky is 56 % of the images).
- **Fixed structure is 25–37 % of an all-sky frame.** The per-camera validity mask from the std image belongs in the
  loader (P040): ignore for segmentation, grey or erased for classification. Masking makes the pixels honest; it does
  not remove camera identity.
- **Burned-in text** exists in Eye2Sky (top-left block) and Montenegro (top right, plus a logo); both lie inside the
  masks and must stay masked in training and evaluation.
- **Time of day is a confound** in the sky and in the camera's exposure; the ray-map input (P044) makes the sun an
  explicit variable, and results should be reported by sun-zenith bin (P030).
- These are linear probes on a frozen backbone; a fine-tuned model can read more, not less. The list is a lower bound.

## Deviations from plan & why
- The plan lists watermarks and borders; CCSN (web photographs) shows no fixed structure in its mean and std images,
  so per-image watermarks cannot be found this way and stay unverified (a watermark tied to no class is noise, not
  a shortcut).
- Resolution was tested by re-embedding at 96 px rather than by a probe for "source resolution", which would only
  have repeated the dataset probe.

## Next phase
P030: Imbalance report.
