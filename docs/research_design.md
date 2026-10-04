# STRATIA — Research design and pre-registration

Version 1.0 · Phase P009 · Registered 2026-10-04, **before any STRATIA model training**. Changes after this date are allowed only as dated amendments at the end of this file, with the reason, and never after looking at locked test results.

## 1. Problem

Ground-based sky cameras are cheap and everywhere, but turning their images into meteorological information is unsolved in the ways that matter:
- Published cloud-type classifiers report 97–100%, yet fall to roughly 57–77% under de-duplicated, temporal or cross-dataset protocols, and expert observers themselves agree only about 55–70% of the time.
- Cloud-base height is measured by ceilometers or by networks of several calibrated cameras; **no published model estimates it from a single camera** with calibrated uncertainty.
- Current geometry foundation models (Depth Anything 3, MoGe-2, MASt3R, VGGT) explicitly mask the sky, so they offer no cloud geometry.

**STRATIA** asks: *from one image of one sky camera, with optional calibration, how much of the sky's state — cloud type, cloud layers, cover and cloud-base height — can a single geometry-aware model recover, how well does it transfer to unseen cameras, and how honest is its uncertainty?*

## 2. Hypotheses

| ID | Hypothesis | Tested by |
|---|---|---|
| H1 | A frozen-then-adapted DINOv3 backbone with a set-valued, WMO-hierarchy-aware loss transfers across cameras better than CNNs trained per dataset. | Leave-one-dataset-out (LODO) recognition at étage and genus level |
| H2 | Trained on soft labels from multiple experts, STRATIA reaches agreement with experts comparable to the agreement between experts. | Montenegro: model-as-extra-annotator agreement vs expert–expert agreement |
| H3 | A single camera, given per-patch viewing geometry, estimates zenith cloud-base height with skill over climatology and over a ceilometer-supervised CNN, with calibrated uncertainty, including at an unseen site. | Eye2Sky CDLRA (in-site temporal hold-out) and CDLRB (held-out site) |
| H4 (supporting) | Geometry inputs (ray maps) and geometry dropout help transfer to a non-fisheye consumer camera. | Ablation; B0268 out-of-distribution set |

## 3. Tasks and data (summary; details in `docs/PLAN.md` and `configs/datasets.yaml`)

| Task | Head | Main data | Locked test |
|---|---|---|---|
| Cloud genus / étage | global, multi-label | CCSN, MGCD, DeepSky, Montenegro (soft), WEBCAM, (GCD/HBMCD if obtained) | Grouped/temporal test splits per dataset; LODO folds; B0268 OOD set |
| Cloud cover (oktas) | global, ordinal | Montenegro N (soft), derived from sky parsing | Montenegro test days |
| Sky parsing | dense | SWIMSEG family, Almería, B0268 dense subset | Almería 4-camera test set; B0268 dense test |
| Cloud layer | dense | Almería, Eye2Sky ceilometer weak labels | Almería test set |
| Cloud-base height | global, ordinal distribution | Eye2Sky images paired with CHM15k CBH (CDLRA, CDLRB) | CDLRA held-out weeks; all CDLRB data (site never seen in training) |

## 4. Metrics

| Task | Primary | Secondary |
|---|---|---|
| Genus / étage | Macro-F1 (étage level primary, genus level secondary) | Accuracy, per-class F1, ECE |
| Expert agreement | Agreement of the model with a random expert vs expert–expert agreement; Krippendorff's α with the model added | KL divergence to the annotator distribution |
| Cover | Mean absolute error in oktas | Within-1-okta rate |
| Sky parsing / layers | Unweighted mIoU and per-class IoU | Predicted-cloud fraction on clear-sky frames (collapse check) |
| Cloud-base height | MAE (m) of the predicted median; **skill = 1 − MAE / MAE_climatology** | Bias, RMSD, within ±250/±500/±1000 m by height bin; CRPS; 80% and 95% interval coverage |
| Reliability | AUROC of `ood_score` for out-of-domain vs in-domain | — |

All headline numbers are reported as mean ± std over 3 seeds, with 95% bootstrap confidence intervals on test.

## 5. Pre-registered success criteria (Gate G4, PLAN P082, evaluated on validation data)

**GO for a CVPR 2027 submission** if **C3** holds **and** at least one of **C1** or **C2** holds:

| ID | Criterion |
|---|---|
| C1 | LODO étage-level macro-F1, averaged over held-out datasets, at least **5 points** above the best baseline (P059), and in-domain macro-F1 no more than 1 point below the best baseline. |
| C2 | On Montenegro, model–expert agreement at least **90%** of expert–expert agreement on étage and on total cloud cover. |
| C3 | Zenith CBH skill over climatology at least **0.20** on the CDLRA validation weeks **and** at least **0.10** on the held-out CDLRB site; at least 10% lower MAE than the ceilometer-supervised CNN baseline; 80% interval coverage within ±5 points of nominal. |

Supporting criteria, reported but not gating: H4 ablation effect; no sky-parsing collapse on B0268 (predicted cloud fraction on clear frames < 20%); `ood_score` AUROC ≥ 0.80.

If G4 is NO-GO, the plan's loop applies (at most two iterations of P073–P080), then the target moves to ICCV 2027. A negative or partial result is still written up honestly.

## 6. Locked-test policy

Test manifests are hashed into `test_lock.json` (P039). No model selection, early stopping, threshold, prompt or hyperparameter decision may use test data. The locked tests are evaluated **once** (P090) with a logged reason; any later re-evaluation is reported as such.

## 7. Threats to validity (and controls)

| Threat | Control |
|---|---|
| Leakage through near-duplicates and adjacent frames | Hash and embedding audits (P024–P025), grouped/temporal splits (P038) |
| Shortcut learning (camera identity, borders, timestamps) | Dataset-ID probe (P029, re-run in P093), masking, geometry dropout |
| Label noise and expert disagreement | Soft and set-valued labels; human ceiling (P035) |
| Ceilometer point vs camera area mismatch | Zenith region pooling; multi-layer cases reported separately |
| Time misalignment between images and ceilometer | Pairing-tolerance study (P031) |
| Over-tuning on validation | Small, logged HPO budget (P078) |
| Single annotator on own data | Two annotators for B0268 labels (P041) |

## 8. Amendments

(none)
