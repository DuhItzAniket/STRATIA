# Research log

What we learned while building STRATIA, written down when it happened, for the paper. One dated entry per phase (or
per finding that deserves its own). Every number names the file or run that produced it; negative results and
surprises are recorded with the same care as successes, because reviewers ask about those first.

Conventions
- **Claim → evidence.** A sentence that could end up in the paper carries a pointer: a path under `docs/data/`,
  `data/`, `runs/registry.csv`, or a commit.
- **Decisions carry the alternative.** What else was considered, and why not.
- **Figures** live in `docs/paper/figures/` (or `docs/data/figures/` for data-quality figures) and are generated
  by a script named in the entry.
- Related records: `EXPERIMENT_LOG.md` (one row per training or evaluation run), `docs/phases/` (what each phase
  did), `paper/outline.md` (where each finding goes in the paper).

---

## 2026-10-06 — Stage C begins: data quality and leakage audit

Context. The manifest (`data/manifest.parquet`, P021) holds 52,032 images from eleven sources: Eye2Sky (AURIC and
BARSE stations, April 2022), MGCD, SWINySEG, CCSN, Montenegro, SWIMSEG, Almería, SWIMCAT, SHWIMSEG, SWINSEG and 25
B0268 frames of our own. Stage C asks whether that material can carry a benchmark: are the files sound, how much
of it is the same picture twice, where do splits leak, and what shortcuts would a model learn instead of clouds.

Why this matters for the paper. Published sky-camera accuracies are often measured on splits that share
near-identical frames (consecutive captures, re-encoded copies, augmented datasets built from other public
datasets). Part of STRATIA's contribution is a protocol in which those effects are measured and removed, so the
audit results are not housekeeping; they are findings.

### P023 — Integrity check

Method. Every manifest image is decoded completely (not just its header) with Pillow; per image we record file
size, container format, pixel mode, decoded size, EXIF orientation, grey-level mean and standard deviation, and
flags for: missing, zero bytes, corrupt or truncated, odd pixel mode, EXIF rotation, size different from the
manifest, size far from the dataset's median, and "blank" (standard deviation below 2 grey levels).
Code `stratia/data/integrity.py`, script `scripts/integrity_check.py`, result `data/integrity.parquet`, report
`docs/data/integrity_report.md`. Tests: `tests/test_integrity_duplicates.py`.

Design choices. (1) Full decode, because a JPEG with a valid header and a truncated body passes a header check and
then fails inside a DataLoader worker at 2 a.m. (2) EXIF rotation is reported as a finding because OpenCV and
PyTorch loaders ignore it while image viewers obey it: the picture the annotator saw may be rotated against the
one the model sees. (3) "Blank" frames are expected in all-sky cameras at night; they are counted, not deleted,
because what to do with night frames is a protocol decision (P030, P037), not an integrity one.

Results. 52,032 images decoded in 318 s: **no corrupt, empty or missing file, no odd mode, no EXIF rotation, no
size mismatch, no size outlier**. 43 images are "blank" — all 125 × 125 clear-sky patches in SWIMCAT (39) and
SHWIMSEG (4), uniformly blue by nature, means 33–254: valid samples, kept. Eye2Sky April frames have no black
nights (minimum standard deviation 13.3). Per-dataset contrast differs strongly (median standard deviation 8 in
SWIMCAT, 23 in Eye2Sky, 68 in MGCD): a dataset signature to test in the shortcut audit (P029). *For the paper:*
the public datasets are mechanically clean; the quality problems are structural (duplicates, splits, labels),
not file-level. Evidence: `docs/data/integrity_report.md`, `data/integrity.parquet`.

### P024 — Exact duplicates

Method. Two SHA-256 hashes per image: of the file bytes, and of the decoded 8-bit RGB pixels (with the array
shape). The second catches the same picture re-saved (metadata stripped, PNG re-encoded, moved between
datasets), which the byte hash cannot. Images sharing a hash form a duplicate group; groups spanning datasets are
the dangerous ones. Code `stratia/data/duplicates.py`, script `scripts/find_duplicates.py`, results
`data/image_hashes.parquet` and `data/duplicate_groups.parquet`, report `docs/data/duplicates_report.md`.

Policy (resolved in this phase). All members of a duplicate group share one `group_id` when the splits are made
(P038), so an exact duplicate can never be on both sides of a split. Cross-dataset groups are kept as evidence
about how the public datasets were assembled, and reported in the data card (P032). Duplicates with different
labels go to the label-conflict audit (P026); nothing is corrected automatically.

Results. **661 images (1.3 %) in 312 exact-duplicate groups, all within one dataset, none across datasets.**
SWINySEG: 533 images (7.9 %) are identical files under different names; SWIMSEG: 52 (5.1 %), of which 46 are
re-saved copies with identical pixels and different bytes (a byte hash alone misses 88 % of them); CCSN: 34
(1.3 %), including 3 groups whose copies carry different genus labels (Cc/Cs, Ac/As, Ac/As); SWIMCAT: 34
(4.3 %); SHWIMSEG 6; Almería 2; Eye2Sky, MGCD, Montenegro, SWINSEG, B0268: none. *For the paper:* (1) the
published random splits of SWINySEG and SWIMCAT place identical images on both sides; (2) SWINySEG does not
contain exact copies of SWIMSEG/SWINSEG, so any overlap between them is through resizing or augmentation and
is a near-duplicate question (P025); (3) exact duplicates with conflicting labels exist in CCSN. Evidence:
`docs/data/duplicates_report.md`, `data/duplicate_groups.parquet`, `data/image_hashes.parquet`.

### P025 — Near-duplicates

Method. DINOv3 ViT-S/16 CLS embeddings (224 px, L2-normalised; cached for the shortcut audit) give each image its
10 nearest neighbours; dHash and pHash computed for all eight flips and rotations give every candidate pair a
"copy" distance. Thresholds were chosen from contact sheets, not assumed. Rejected alternative: cosine >= 0.97 *or*
pHash <= 10 — the pHash half matched Montenegro frames of different nights, because the pHash of a smooth sky
carries little information. Chosen rule: copy if dihedral pHash <= 2; same scene if cosine >= 0.97; either counts.
Same-station Eye2Sky pairs are a time series and go to temporal blocking (P027).

Results (`docs/data/near_duplicates_report.md`, `data/near_duplicate_groups.parquet`, `data/swinyseg_sources.parquet`):
- **34.6 % of SWINySEG (2,342 of 6,768 images) are flipped, rotated or re-encoded copies of SWIMSEG (2,096) or
  SWINSEG (246) images**, found by an exhaustive dihedral-pHash pass over every pair. Exact hashing saw none of
  them. *For the paper:* the three datasets are one source; any protocol that trains on one and tests on another
  measures memorisation.
- **MGCD's 8,000 images are a few hundred scenes**: 97.2 % of them sit in 227 same-scene groups (largest 291)
  of consecutive frames seconds apart. Almería (43.5 %) and Montenegro (35.0 %) behave the same way at lower
  cadence. *For the paper:* published random-split results on these datasets are inflated by design; our splits go
  by group and time.
- Overall: 14,112 images (27 %) in 1,656 near-duplicate groups; CCSN has 149 small groups of re-posted photos
  (with the label conflicts P024 found); B0268 none; Eye2Sky cross-station pairs show that overcast skies look alike
  15 km apart.
- Negative result worth keeping: a perceptual hash alone cannot audit sky images (too many spurious matches on
  smooth skies), and an embedding alone cannot see flips. Two measures were needed.

### P026 — Label-conflict audit

Method. Every pair that P024 (exact duplicates) or P025 (copies, same scene) marked as the same picture was
compared on the label its dataset ships with: the native class (CCSN, MGCD, SWIMCAT, Montenegro étage set) or the
mask (SWIM family, SHWIMSEG, Almería; the copy's mask flipped or rotated like its picture, then pixel agreement).
Montenegro's multi-rater distributions were summarised as majority shares. Decision: pHash copies inside a camera's
own time series count as same scene (frames days apart hash alike, P025), so a label difference there is a
consistency rate, not an error; alternative rejected: treating them as copies, which would have called 512
Montenegro pairs "mislabelled". Nothing was corrected; the policy (report, section Policy) keeps conflicting
pictures out of every test split and leaves training labels to the harmonisation (P033, P034).

Results (`docs/data/label_conflicts_report.md`, `data/label_conflicts.parquet`, `data/mask_agreement.parquet`,
sheets `docs/data/figures/label_conflict_*.jpg`, `mask_*.jpg`):
- **CCSN files the same photograph under two genera**: 105 of its 129 copy pairs and 22 of 25 same-scene pairs
  disagree; 200 images (7.9 %) are in an exact or copy conflict (Ns/St, Cb/Cu, Cc/Cs, Ac/Cc, Sc/St most often, some
  across étages). *For the paper:* a ceiling on single-label genus accuracy on CCSN, and the reason STRATIA predicts
  a genus set and evaluates at étage level.
- **MGCD labels are per sequence**: consecutive frames disagree in 0.6 % of 36,532 same-scene pairs; 701 same-scene
  pairs (452 images) straddle the official train/test split.
- **People cannot agree on cloud height from a webcam frame**: Montenegro's 5–9 raters reach no majority on the
  height code for 39 % of images (median majority share 0.60; oktas 0.75). Look-alike frames days apart flip
  Clear/High clouds in 216 pairs. *For the paper:* image-only height labels are distributions; the ceilometer is
  the reference (Paper B); evaluation on Montenegro scores against the rater distribution.
- **Segmentation annotation noise is about 5 % of pixels**: SWIMSEG's 26 pixel-identical pairs were annotated twice
  and agree on 94.9 % of pixels at the median (76 % at worst); SWINySEG copies inherit either mask (the construction
  copies themselves agree on 99.7 %). One Almería frame is stored under two timestamps with incompatible masks.
  *For the paper:* SWIM segmentation scores above roughly 95 % pixel accuracy are not meaningful.
- SWIMCAT: no conflict of any kind. Montenegro frames carry a burned-in timestamp and a logo (shortcuts for P029).

### P027 — Temporal autocorrelation

Method. For each camera with timestamps, pairs of frames sampled at time gaps in doubling bins (30 s to 23 days)
and measured with the P025 instruments (DINOv3 cosine, pHash distance), against two baselines from the same camera:
*different day* (pairs more than a day apart) and *same hour, other day* (the sun alone). Rule for the minimum gap:
the mean-cosine excess over the different-day level falls to 10 % of the adjacent-frame excess and the same-scene
share comes within one point of the baseline; the alternative, the first crossing of the 10th percentile, was not
used because a few alike pairs matter more than the typical pair for leakage. Script
`scripts/temporal_autocorrelation.py`, figure `docs/data/figures/temporal_autocorrelation.png`, table
`data/temporal_curves.parquet`, report `docs/data/temporal_report.md`.

Results.
- **Eye2Sky (30 s cadence): 82–92 % of adjacent frames are the same scene; the share is below 1 % after 2–4 h; the
  mean cosine reaches the different-day level between 8.5 and 17 h, and the same hour next day is no more alike than
  the same hour on any day.** Decision: the block is one calendar day, no buffer. *For the paper:* a random split of
  an all-sky time series is leakage by construction; day blocks are the smallest honest unit.
- **Montenegro (20 min cadence, Oct–Dec): correlated across days** (excess 0.30 at one day, 0.09–0.17 up to six
  days, baseline at 11+ days): weather and season persist. Decision: contiguous blocks of at least 7 days; residual
  cross-boundary correlation reported rather than bought off with an 11-day buffer.
- **Almería** comes in bursts (100 % same scene below 2 min); excess 0.05 at one day: day blocks.
- **The two Eye2Sky stations 15 km apart are correlated through the weather but are not duplicates**: same-moment
  cosine 0.81 against 0.70, same-scene share 0.4 %. A held-out station is a legitimate out-of-camera test; holding
  out its days too removes the shared-weather term.
- **The sun is a confound no temporal split removes**: the same hour on another day adds 0.02 cosine (0.04 for
  Montenegro's fixed-foreground camera). *For the paper:* motivates the sun-position (ray-map) input and a P029 probe
  for time of day.
- Housekeeping found: the manifest holds only 9 Eye2Sky days (the download has about 120 per station) and no Almería
  timestamps (they are in the file names); both to be fixed before the splits (P037/P038).

### P028 — MGCD ≟ GRSCD (deferred)

GRSCD cannot be fetched from here (Baidu link, like HBMCD). Every number the publications allow us to compare matches
MGCD exactly (8,000 all-sky images of 1,024 px from the same Tianjin fisheye camera, the same seven sky types, the
same 4,000/4,000 split). Decision: MGCD and GRSCD count as one source in every protocol and in the literature
comparison until a hash run (`scripts/find_duplicates.py`, `scripts/find_near_duplicates.py`) on the GRSCD files
says otherwise. `docs/phases/P028-mgcd-grscd.md`.

### P029 — Shortcut audit

Method. Per-camera mean and std images at 224 px give a fixed-structure mask (pixels that never change, plus
everything outside the main field of view). The same images were re-embedded with DINOv3 after a controlled change
(`lowres`: 96 px and back; `skyonly`: fixed structure grey; `fixedonly`: everything else grey), and linear probes on
the frozen features, trained and tested on disjoint days or near-duplicate groups, predicted what a cloud model
should not need: the dataset, the camera, the hour. Script `scripts/shortcut_audit.py`; report
`docs/data/shortcuts_report.md`; table `data/shortcut_probes.parquet`; figure
`docs/data/figures/shortcut_camera_statistics.jpg`.

Results.
- **The source is in the features**: a linear probe names the dataset at 99.7 % (chance 9 %) and tells SWIMSEG from
  SWINySEG, two releases of one camera, at 99.7 %. Down-sampling to 96 px costs under one point. *For the paper:* any
  pooled accuracy over mixed sources is partly "which camera"; leave-one-dataset-out and held-out-station results are
  the ones that count, and per-source numbers must accompany every pooled one.
- **The two Eye2Sky stations are told apart at 100 % from the sky alone** on unseen days (fisheye corners, horizon
  objects and the timestamp block painted grey), and at 100 % from the fixed structure alone. *For the paper:*
  camera identity is not a border artefact that masking removes; it is in the optics, exposure and colour response.
  Masking is still right (25–37 % of an all-sky frame is not sky), but for pixel honesty, not for identity removal.
- **The hour of day is readable**: 4× chance from the sky (sun position, brightness) and 6× chance from the fixed
  structure (lit horizon objects, exposure state, and Montenegro's timestamp digits). *For the paper:* the sun is a
  confound; STRATIA's ray-map input makes it an explicit variable, and results are reported by sun-zenith bin.
- **Burned-in text leaks labels**: Montenegro's timestamp and logo pixels alone predict its class at 35 % balanced
  accuracy (chance 20 %). Eye2Sky also carries a text block (top left). Both are masked from here on.
- For scale: linear probes on frozen DINOv3 reach 85 % on MGCD (7 types), 46 % on CCSN (11 genera; the label noise
  of P026), 100 % on SWIMCAT (5). Resolution changes these by at most two points.
- Negative result worth keeping: masking the fixed structure *helps* the Montenegro class probe (54.7 % against
  53.0 %); the foreground is noise for the cloud task.

## 2026-10-08 — Stage C closes, Stage D begins

### P030 — Imbalance report

Method. Every label distribution we will train on (native classes per dataset and published split, Montenegro
majority oktas and height codes, mask cloud fraction, sun-zenith bins, ceilometer cloud-base bins), each with the
largest/smallest ratio, the normalised entropy and the effective number of classes. `scripts/imbalance_report.py`,
`docs/data/imbalance_report.md`, `docs/data/figures/imbalance.png`, `data/imbalance_tables.parquet`.

Results and decision.
- **The class sets are nearly balanced (ratios 2–3); the imbalance is between sources and in the physical targets.**
  Eye2Sky is 56 % of all images; Montenegro is 58 % low cloud with raters putting nearly every base below 1.5 km;
  Montenegro's oktas are U-shaped (0 and 8 oktas 47 % together, 4 oktas 2.4 %); the segmentation sets have almost no
  clear (1.4 %) or overcast (2.7 %) frames; the ceilometer records are 34 % "no cloud overhead".
- **Decision:** sources sampled in proportion to the square root of their size (Eye2Sky 56 % → 35 %, B0268 25 frames
  → 1 %), class-balanced loss weights inside a source (beta = 0.999), soft labels never resampled, macro metrics by
  default, results by sun-zenith and cloud-fraction bin. Rejected: natural sampling (majority-camera collapse, the
  P029 shortcut) and uniform sampling (tiny sources repeated thousands of times). *For the paper:* the "none" class
  of the CBH head and the clear/overcast extremes need explicit attention; they are where the legacy model failed.

### P031 — Ceilometer QC & pairing tolerance

Method. All 1.36 M CHM15k records of CDLRA and CDLRB checked for completeness, range, layer order and flags; then the
instrument's own time consistency measured (agreement of two records Δ apart; unanimity of all records inside ±w)
and the two sites compared at the same instants. `scripts/ceilometer_qc.py`, `docs/data/ceilometer_qc.md`,
`docs/data/figures/ceilometer_pairing.png`.

Results and decision.
- The streams are complete (99.998 %, no duplicates, no out-of-range or mis-ordered layers); flags remove 6.1 %
  (CDLRA) and 3.5 % (CDLRB) of records (rain 3 %, window particles and optics on CDLRA 3 % each).
- **Pairing tolerance ±30 s:** two records 30 s apart agree on presence / étage in 96.4 / 96.1 % of cases against
  97.7 / 97.3 % at the instrument's 15 s cadence; ±2 min doubles the disagreement, ±5 min triples it, for one to two
  points more coverage. Label = median lowest base of the clean records in the window, cloudy share as confidence.
- **A cloud base is local:** the two sites 15 km apart agree on presence 82 % of the time and differ by 255 m at the
  median when both see cloud. *For the paper:* CDLRB is a genuine held-out site (criterion C3), and the time-mismatch
  part of the label noise is about 4 %, to be quoted next to any CBH error.

### P033 — WMO ontology

`configs/ontology.yaml` (validated by `stratia/labels/ontology.py`): ten genera with WMO étages, the contract's
twelve output classes, code tables 0513 / 0515 / 0509 / 1600 / 2700 with the genera each code names, and every
dataset's native classes mapped to genus sets or explicitly excluded (MGCD merged types → sets with alternatives,
"mixed" → cloud of unknown genus; SWIMCAT → clear / cloud only; CCSN Ct → contrail). Decision: merged classes become
sets, never one forced genus (a forced choice would import a 50 % error rate). Open item: the code-table wording was
written from the tables as known to the author and must be checked against the current WMO edition before the
appendix is final.

### P032 — Gate G1: data card and sign-off

`docs/data/data_card.md` collects the Stage C audit in the Datasheets-for-Datasets structure: eleven image sources
(52,032 images, six camera types) and two ceilometers (1.36 M records); what is in the frames besides sky; every
cleaning step with its numbers; intended and unintended uses (no pooled accuracy without per-source numbers, no
genera from SWIMCAT, no general height model from Montenegro's code, CC-BY-NC inheritance from the SWIM family);
seven open items with owners, none blocking Paper A. Gate G1 signed 2026-10-08. Housekeeping: Almería timestamps
parsed into the manifest (row order unchanged, hash `464078b6d033dd39`). *For the paper:* the data card is the
benchmark's datasheet; its open-items table is the honest limitations section.

### P034 — Dataset → ontology mapping

`data/labels.parquet` (one row per manifest sample) carries set-valued labels for every image whose dataset names
them: CCSN 2,543 genus sets (200 conflict merges as alternatives, 199 contrails), MGCD 6,980 (4,204 sets of
alternatives from its merged classes, 1,020 "mixed" = cloud of unknown genus, 1,338 clear), Montenegro 2,363 from the
majority codes, SWIMCAT clear / cloud only, the segmentation sets cloud presence and (Almería) étages from the layer
masks; Eye2Sky and B0268 wait for P036 and P041. Decision: merged classes become sets of alternatives, never a forced
genus; conflicting copies take the union and are kept out of test splits. `docs/data/label_mapping.md`.

### P035 — Montenegro soft labels & human ceiling

Krippendorff's alpha and leave-one-rater-out agreement over 2,270 images with at least two raters, on the raw codes
and on the derived STRATIA labels (`docs/data/montenegro_ceiling.md`, targets in `data/montenegro_targets.parquet`).
**Human ceiling for criterion C2: étage set 85.2 % (alpha 0.54), total cover within ±1 okta 87.3 % (alpha 0.91);
the model must reach 90 % of these on the same images against the rater majority.** Genus from the codes is weak
(exact set alpha 0.28; Cc 0.07, Cb 0.09) and the height code has no consensus (alpha 0.36). *For the paper:*
Montenegro tests cover and étage; genus and height on it are soft, exploratory comparisons; the ceilometer stays
the CBH reference.

### P036 — Ceilometer → targets

`data/ceilometer_targets.parquet`: for both sites and every 30 s of every day, the label a frame would get from the
clean records within ±30 s: no cloud overhead (30 %), or the étage of the median lowest base (daytime: low 38-40 %,
mid 20 %, high 10 %), "mixed" when fewer than half of the records see cloud (2 %), with the cloudy share as
confidence, the base spread inside the window (33-35 m at the median), the mean layer count (16 % of windows have
two or more layers) and the second layer's height. Both étage thresholds (weak 2 / 6 km, WMO 2 / 7 km) are stored
for the ablation. 94-97 % of daytime windows are labelled; the rest are flagged records. *For the paper:* the CBH
target distribution and its confidence are now explicit; pairing with frames (P047) waits for the images at the
ceilometer sites.

### P037 — Split design

`docs/splits.md` and `configs/splits.yaml`: the unit a split may not cut is a connected component of "same group"
(P024 exact and P025 near-duplicate groups) and "same block" (P027: Eye2Sky days shared by both stations, Montenegro
weeks, Almería and B0268 days). Protocols: in-domain blocked (70/10/20 by units, stratified on native labels, conflict
merges never in test), leave-one-dataset-out over CCSN, MGCD, Montenegro, SWIM family and Almería (the main
protocol for H1 / C1), held-out Eye2Sky station in same-day and disjoint-day variants, few-shot B0268 by days, and
the published MGCD / Almería splits for comparison only. Seven guarantees are listed for P038 to prove. Decision:
no buffer days between blocks (P027), no image-level random split anywhere.

### P038 — Split generator

`scripts/make_splits.py` implements the design with proofs: 8,661 units over 52,032 images, every source split
70/10/20 by units with native-label shares within a point across splits (Montenegro's eleven weeks and Eye2Sky's
nine days coarser), five LODO folds, two station variants, hashes per file (`data/splits/hashes.json`,
`docs/data/splits_report.md`). **Decision forced by the real data:** near-duplicate links are not applied inside a
time series. Taken literally, P025's groups fused eight of nine Eye2Sky days and Montenegro's ten weeks into three
units, because pHash matches between smooth-sky frames days apart (P026) and same-scene links across days and
stations (P027) are coincidences and weather look-alikes, not copies; inside a time series the block is the unit
and only exact duplicates add links. *For the paper:* the split rule is a published, hashed artefact; the published
random splits of MGCD, SWINySEG, SWIMCAT and Almería are reported next to it as the leaky baseline protocol.

### P039 — Lock test sets

`test_lock.json` (committed) holds the SHA-256 of every test set of every protocol (13 sets, 53,252 ids), bound to
the manifest hash and the split config; a final evaluation on them must log a reason (`runs/final_eval_log.md`), and
CI refuses a training config that names a test file (`scripts/check_test_lock.py`). Re-locking needs `--force` and an
explanation in the commit. *For the paper:* test-set integrity is a verifiable artefact, not a promise.

### P040 — Label-noise estimation

Confident-learning flags from out-of-fold probes on frozen DINOv3 features (folds by split unit; C = 0.1 after a
C = 1.0 run saturated every probability). Strong flags (suggested class >= 0.5 and >= 0.25 ahead): CCSN 29 %, MGCD
7 %, Montenegro 11 %, SWIMCAT 0 (`docs/data/label_noise_report.md`, `data/label_noise_flags.parquet`). The sheets
confirm real errors in CCSN (a lone cumulus labelled Ac, cumulonimbus labelled Ns), boundary noise between MGCD's
merged classes, and Montenegro raters using "clouds of vertical development" for dark overcast skies. Decisions:
nothing deleted or relabelled; CCSN genus results reported with and without strongly flagged test images;
Montenegro's vertical-development class folded into the low étage for scoring; MGCD scored on the set. Negative
result worth keeping: the plain confident-joint flag rate is not interpretable when the probe is weak or a class
is rare (CCSN 46 %, Montenegro 55 % plain against 29 % and 11 % strong).

### P041 / P042 — B0268 labelling (blocked) and Gate G2

No B0268 logger frames exist (owner item), so P041 is BLOCKED and Gate G2 is signed **conditionally** on 2026-10-10: every other Stage D criterion is met with evidence (`docs/phases/P042-gate-g2.md`; 149 tests, lock check, lint). *For the paper:* C1/C2 do not depend on B0268; H4 (geometry helps transfer to a consumer camera) and the B0268 OOD numbers wait for P041/P083. Stage E starts on fixed labels and splits.

### P043 — Camera models

Every camera now answers pixel <-> direction through one interface: OCamCalib for Eye2Sky, OpenCV fisheye/pinhole (NumPy, matching cv2 to 1e-3 px) for the B0268 through CloudScope's calibration file, and an explicit unknown camera (all-zero ray maps) for CCSN, MGCD, Montenegro, SWIM and Almería. The B0268 model is a **nominal datasheet model** (105 deg equidistant, no distortion) flagged `provisional` until the checkerboard calibration exists; no geometry number about the B0268 may come from it. Shared test vectors (39 samples, 3 cameras) are the contract's executable part for CloudScope. *For the paper:* the geometry pipeline is camera-agnostic by construction; the single-camera claim (ADR-003) does not depend on a particular projection.

### P044 — Sun position & calibration validation

The Sun validates the whole geometry chain on real frames: the OCamCalib centre convention of P013 is confirmed (the swapped reading fits at 0.26–0.33° instead of 0.07–0.10°), file-name times are UTC (a ±1 h offset tilts the fitted axis by 8.6°), and all four stations with images fit at **0.08–0.12° median** (AURIC 80 inliers over 18 days, BARSE 40, OLDLR 51 and WESTE 30 over 1 and 8 April), optical axes 0.4–2.6° from the zenith (`configs/camera_poses.yaml`, `docs/data/sun_validation.md`). The declared `external_orientation` reads as Ry(pitch)·Rx(−roll)·Rz(yaw) camera-to-world on the OCamCalib frame (768 readings tested; four names are one matrix here); it reproduces the Sun to 0.1–0.35° on three stations and is **2.9° off on WESTE's March file** (the camera was adjusted; its June file changes roll by 3.3°), so declared orientations are a fallback, never the reference for pairing. Negative results worth keeping: a 250 threshold on a camera that clips at 240 produced a 5.6° 'calibration error' that was a detector error; bright cumulus edges clip too (peak gate); hazy glows bias centroids by up to 1.3° (area limit); BARSE shows whole-day ghost tracks 12–33° off on three days (dome reflections). *For the paper:* the Sun is the free, exact calibration target for ground cameras; pose accuracy of ~0.1° is far below the 6° patch and the ceilometer footprint.

### P045 — Ray-map generator

Contract ray maps from camera + pose + Sun + mask through one affine map (`stratia/geometry/raymap.py`); near-horizon floor 5° for Eye2Sky (`configs/near_horizon.csv`). Overlays on four stations put the projected Sun on the visible disc; the patch holding it looks within 0.7–3.5° of the Sun (patch quantisation; a patch is ~6° wide) with Sun-relative azimuth within 2.4° (`docs/data/ray_maps.md`). About 54 % of patches are valid on an all-sky frame. *For the paper:* the geometry input is physically defined and camera-agnostic; the Sun-patch check is its executable sanity test.

### P046 — Geometry-consistent transforms

ADR-004: native camera view, no re-projection; every geometric augmentation is a pixel remapping tracked in a `GeoState` (affine current → source pixels + coverage), and the ray map is recomputed through it, never resampled (`stratia/geometry/transforms.py`). Tests prove the composed map against a coordinate-encoding image (< 1 px), that the Sun's patch still looks at the Sun after random crop/rotate/flip chains, that flips mirror and rotations rotate the zenith map, and that padding is invalid. Decision: flips carry directions rather than mirroring the world (left-handed layout, which no target depends on). *For the paper:* augmentation cannot break geometry by construction, which is the premise of the ray-map ablation (P091).

### P048 — Augmentation policy

Augmentations are what cameras do, not what image libraries offer: exposure and white balance in linear space within physical ranges, JPEG, read + shot noise, a bloom at the Sun's own pixel with the dome ghost seen in P044, dirt and raindrops, and dark obstructions that also mark the dense label invalid and clear the ray-map patches (`stratia/augment`, `configs/augment.yaml`). Geometric policies differ by camera family: all-sky imagers rotate about the centre and never flip, pointed cameras flip and crop. No hue shift exists anywhere (sky colour is signal). `strength` and per-augmentation switches are the P080/P091 hooks. Gallery reviewed (`docs/data/augmentation_gallery.md`). *For the paper:* the augmentation section can be stated as a physical model of the acquisition chain, with one knob for the regularisation study.

### P049 — Consumer-view synthesis

1063 synthetic 105 deg views (AURIC 320, BARSE 316, OLDLR 211, WESTE 216) rendered from Eye2Sky frames at random pointings (elevation 15-70 deg) with the nominal B0268 lens, each with an exact ray map from its own virtual camera and pose (`stratia/geometry/synthesis.py`, `docs/data/synthetic_views.md`); mean valid share 82%. 365 views at the ceilometer sites carry an inherited cloud-base label ({'low': 161, 'mid': 87, 'none': 60, 'high': 42, 'mixed': 15}) under the stated **flat-layer assumption** (a base height, not a slant range) with the provisional nearest-grid-point pairing that P047 will fix; views at AURIC/BARSE are unlabelled and serve the appearance gap only. Colour-coded direction tests prove the re-projection samples the right sky directions to ~1 deg. *For the paper:* this is the bridge from zenith fisheyes to a pointed consumer camera; its value is measured in the P091 ablation and the B0268 zero-shot number (P083), never assumed.

### P050 — Metadata dropout

Geometry dropout masks the ray map and the Sun vector to the contract's null state with p = 0.3 each, on calibrated samples too, so that the presence of geometry is not a dataset signature (P029); the SkyRay encoder turns the null state into learned null tokens and the meta FiLM into learned null parameters (`stratia/augment/dropout.py`, `stratia/model/skyray.py`). A stub model trains on mixed batches and runs with everything or nothing known. *For the paper:* the geometry ablation (SkyRay on/off, dropout rate) is wired at the input; the model cannot refuse an uncalibrated photograph.

### P047 — Image–ceilometer pairing

The cloud-base-height table exists: 104,683 daytime pairs at OLDLR and WESTE (one frame every 2 min, label from the ceilometer's 30 s grid within 15 s, Sun-fitted pose required, zenith pixel inside the mask), 104,683 of them in calendar-day splits shared across the two stations (test 21,859 pairs / 25 days, train 71,867 pairs / 81 days, val 10,957 pairs / 12 days); label mix {'low': 40850, 'none': 30518, 'mid': 21011, 'high': 10034, 'mixed': 2270}; low cloud is 39% of the pairs, which is the overcast-day mass the plan warned about (`day_label_share` travels with every pair for P075's weighting). The test days are locked (`test_lock.json` re-locked with the two new sets; nothing previously locked changed). Prerequisite finding (P044 update): the dome ghost produced whole-week false Sun tracks in summer; a 15° gate around the declared prediction fixed it, and the fitted zenith pixel is stable to 1-5 px across each station's calibration windows, so the cameras never moved while their declared orientations differ by up to 11°. *For the paper:* the CBH supervision set is reproducible from public data with measured, not assumed, geometry; the synthetic consumer views (P049) now inherit labels from this table.

### P051 — Feature cache

Every frozen backbone the baselines use is cached once: pooled embeddings for 72,974 rows (52,032 manifest images + 20,942 cloud-base-height frames) at 224 px for DINOv3 S/B/L, DINOv2 S/B, CLIP B/16 and SigLIP 2 B/16, ViT-S also at 512 px, and the ViT-S 512 px patch grid for the 8,870 dense-label images (6.5 GB). Lessons: DINOv3 ViT-L produces NaN for every image in fp16 and needs bfloat16; OpenAI CLIP checkpoints are pickle-only and transformers 5 refuses them on torch 2.5, so they are converted to safetensors once. *For the paper:* frozen-feature stages cost minutes, which is what makes the baseline table and the ablations affordable on one laptop.

### P052 — Legacy CNN

The prototype's MobileNetV3-L recipe on CCSN, random versus grouped split with the same pool and sizes (3 seeds): random val macro-F1 0.511, grouped val 0.588. Inside the random val the leaked images (24 of 255 have a near-duplicate in train) score 0.146 against 0.543 for the clean ones. *For the paper:* the leakage effect is measured per image, not inferred from a split-level gap; the prototype's 70/15/15 split over the whole dataset is not reproduced because it would read the locked test rows.

### P053 — Fine-tuned CNNs

ResNet-50 and ConvNeXt-T, 3 seeds, grouped splits (`docs/data/baseline_cnn.md`); largest seed std 0.026. Fit verdicts are logged per run.

### P054 — Frozen probes

Linear probes on frozen features are the strongest recognition baselines: best in-domain mean macro-F1 0.765 (('dinov3_vitb16', 224)), best LODO étage 0.583 (('dinov3_vitb16', 224)) over the CCSN, MGCD, Montenegro and Almería folds (`docs/data/baseline_probes.md`). Amendment: the LODO folds now carry a val role (the held-out source's in-domain val units) so criterion C1 can be judged on validation data; the five LODO test entries were re-locked, nothing else changed (research design §8).

### P055 — CLIP / SigLIP 2

Zero-shot with fixed prompts: best 0.413 mean macro-F1 (('clip_vitb16', 'definition')); linear probes on the same embeddings in `docs/data/baseline_probes.md`.

### P056 — Small VLM

Qwen3-VL-2B zero-shot sanity check (`docs/data/baseline_vlm.md`), one prompt, no 4-bit on Windows: a sanity check, not a tuned baseline.

### P057 — Segmentation baselines

dinov3_vits16_patch_probe: val mIoU 0.875, clear-sky cloud fraction 0.323. small_unet: val mIoU 0.867, clear-sky cloud fraction 0.237. The collapse check is the number the legacy prototype failed (`docs/data/baseline_segmentation.md`).

### P058 — CBH baselines

climatology_median: MAE 1147 m, skill +0.000. climatology_month_hour: MAE 1131 m, skill +0.014. etage_lookup: MAE 1124 m, skill +0.020. ridge_dinov3_vits16_512: MAE 699 m, skill +0.390. cnn_resnet18_logheight: MAE 724 m, skill +0.368. The ceilometer-supervised CNN is the C3 reference (`docs/data/baseline_cbh.md`).

### P059 — Baseline leaderboard

`docs/data/baseline_leaderboard.md` is generated from the result files and names the best baseline per criterion; every number is validation-only.

### P060 — Gate G3

Baselines reproduced with acceptable seed variance and plausible against the literature; signed (`docs/phases/P060-gate-g3.md`).

### P061 — Architecture spec

The v1 design is fixed in `docs/architecture.md` with ADR-005 (heads) and ADR-006 (losses). Two choices matter for the paper: the CBH output is a distribution over 24 étage-aligned log bins (so the layer-consistency term is a sum over bins), and every loss is masked per sample so a batch can mix label types.

### P062 — SkyRay encoder

Geometry enters as tokens added to the backbone tokens with a learned strength α (initial 0.1) and as FiLM from the Sun vector; one traced graph serves known and unknown geometry. α is the first number to read in the geometry ablation.

### P063 — Global heads

Attention pool ⊕ CLS → genus / étage / oktas; oktas through CORAL (rank-consistent by construction, which the calibration phase relies on).

### P064 — Losses

Set-valued genus BCE with partial negatives (single-label datasets are not exhaustive), expert shares as soft targets, CORAL with soft cumulative targets, hierarchy consistency by noisy-OR, CBH ordinal CE + CRPS. Each has a toy test with a known gradient direction.

### P065 — Sky-parsing head

A 0.6 M-parameter DPT-lite decoder on frozen DINOv3 tokens reaches val mIoU 0.919 against the token probe's 0.875 and labels 22% of ceilometer-clear Eye2Sky sky as cloud (probe 32 %, prototype 85–99 %). *For the paper:* the dense head on frozen features already beats a from-scratch U-Net and the probe on every dataset.

### P066 — Layer head

Same decoder trunk, three layer classes on cloud pixels, trained on Almería: layer mIoU 0.559 with a bootstrap 95 % interval [0.467, 0.661] on the small val set; sky parsing stays at 0.894 in the joint run.

### P067 — CBH head

Zenith-prior attention over an 8 × 8 pooled token grid: val MAE 566 m, skill +0.506 against climatology, 22 % below the ceilometer-supervised CNN (724 m) and below the ridge (699 m); no mode collapse (prediction histogram ≈ label histogram). *Finding:* with geometry dropout 0.3 the head learns a content-only solution (zeroing the geometry changes nothing, 566 m); trained without dropout it uses the geometry (551 m, and 765 m when it is removed). The dropout rate is a real trade between robustness to uncalibrated hosts and the use of calibration, not a free regulariser; P075 / P091 set it.

### P068 — OOD score

Energy + Mahalanobis on frozen features: leave-one-source-out AUROC 0.961–1.000. The Mahalanobis term reads camera identity (P029 again), the energy term adds nothing; a new camera will be called 'uncertain' until adaptation adds its features to the fit.

### P069 — Sampler

Per-task batches, square-root dataset balance (τ = 0.5), label masks; observed shares match the config.

### P070 — Balancing

Kendall weighting over present tasks and a task-gradient cosine monitor logged per step.

### P071 — Sanity suite

Contract shapes, overfit one batch, shuffled labels at chance, constant input, determinism: the checks that would have caught the legacy pipeline bugs, run in CI on a fake backbone and before every Stage H run.

### P072 — Training engine

AMP, accumulation, clipping, EMA, atomic checkpoints with RNG and epoch position, NaN guard, early stopping on val; a resumed run is bit-identical to the uninterrupted one. Stage G complete.

### P073 — Stage-1 global heads

On cached DINOv3 features the global heads reach val mean étage macro-F1 0.817 in-domain and 0.548 averaged over the four LODO folds (P054 probe reference 0.583). Montenegro: the model agrees with the majority étage set and the majority cover (±1 okta) at 1.047 and 1.027 of the leave-one-out human ceiling. *For the paper:* the LODO gain comes with the Eye2Sky étage rows (ceilometer lowest-base étage as a partial label), without them the LODO mean is 0.485.

### P074 — Stage-2 dense heads

Kendall weighting settles at sky 6.2 / layer 1.6 and recovers most of the sky-parsing loss of the fixed-weight joint run (mIoU 0.905, clear-sky cloud fraction 0.299); the layer head stays at 0.559.

### P075 — Stage-3 CBH head

Seed std 10 m; trained on OLDLR alone the head keeps skill +0.448 on the never-seen WESTE site (C3 asks 0.10); the P049 synthetic views give the best single number (538 m); the overcast-day weighting does not help; the geometry remains unused at dropout 0.15 and 0.3.

### P076 — Stage-4 joint

All heads together: étage 0.799, dense 0.881, CBH 545 m (`docs/data/stage4_joint.md` lists the head-by-head deltas and the task-gradient cosines).

### P077 — Stage-5 LoRA

LoRA on the last blocks with augmentation: étage 0.805 → 0.810, dense 0.869 → 0.877 (frozen → LoRA on the same val images).

### P078 — HPO

Random search, 20 trials: scores 0.775–0.819; the sensitivity table says which knobs matter (`docs/data/hpo_global.md`).

### P079 — Learning curves

Global heads 0.662 / 0.698 / 0.774 / 0.817 at 10 / 25 / 50 / 100 % of the training units; CBH 756 / 661 / 580 / 557 m at 10 / 25 / 50 / 100 % of the days.

### P080 — Regularisation

Seed std n/a; the table in the phase doc.

### P081 — Fit sign-off

`docs/data/fit_report.md`: gaps, verdicts, ECE, CBH coverage after a cross-fitted temperature.

### P082 — Gate G4

GO: see `docs/data/gate_g4.md` and the phase doc.

### P089 — Calibration

Cross-fitted on validation units: the étage and oktas heads are calibrated as trained (ECE 0.023 / 0.030) and temperature scaling would worsen them; the genus native-class confidence needs T ≈ 4.5 (0.145 → 0.092); the CBH distributions are too wide and a conformal radius brings 80 % / 95 % coverage to 0.798 / 0.952 with the median unchanged.

### P088 — Robustness

Corruption table in `docs/data/robustness.md`: the weakest spots are sensor noise for étage, sun glare for sky parsing and sensor noise for cloud-base height; the P048 augmentations cover the photometric ones, which the LoRA stage can train against.

### P096 — ONNX export (pulled forward)

The Gate G4 model exported as one FP32 ONNX graph with the P089 calibration folded in (`docs/data/onnx_export.md`): parity with PyTorch ≤ 1.6e-4 absolute, 172 ms per 512 px frame on the laptop CPU, ray-map pooling in the graph within 0.05° of the training ray maps. First look at the owner's B0268 camera (four frames of 19 Oct 2026, nominal camera, zenith pose, no labels): finite and well-formed outputs, one of four frames above the OOD threshold, oktas 6–8 against 31–61 % cloud pixels from the sky-parse head, genus leaning to Ns / Cs. A cover over-estimate on an unseen consumer fisheye is the first thing P083 should quantify once the P041 labels exist (393 frames over four days, 18–21 Oct, two annotators in progress).

### B0268 through CloudScope (zero-shot preview, 2026-10-21)

CloudScope Stage F ran the exported `stratia-1.0.0-demo` on replayed B0268 frames of 19 Oct 2026 (nominal fisheye model, zenith pose, Sun from the clock and site) and on the live B0268 at night. Daytime replay, 700 frames at 4656 x 3496: reliability score 0.9-1.2 against the threshold 1.457, so **no frame was flagged out-of-domain** (the four-frame preview of P096 had flagged one of four); top genus Ns (5) or Cs (2) on every frame, oktas 6 on nearly all, mean cover 0.66-0.69, mean cloud base about 5.1 km, p(cloud overhead) high. The frames of that day were mostly blue sky with cumulus and cirrus, so the cover and the genus are over-estimated on this camera: the first quantified item for P083 once the P041 labels exist. A night run (live camera, 42 frames) gave Cb on every frame with one frame flagged: the night flag (mean luminance below 8 %) did not fire on indoor-lit frames; P083 should add an explicit night rule.

### P041 — B0268 labelling

393 frames of the owner's Arducam B0268 (4656 x 3496, 105 deg fisheye, zenith mount, site BLR01) were logged by the CloudScope sky logger over four afternoons at 30 s cadence and labelled per frame with the sheet of `DATA/B0268/labels/` (usable, oktas 0-9, low / mid / high cloud genus, lowest base guess, sun glare, obstruction, notes; the sheet's `guide` tab is authoritative). The plan asked for two human labellers. That guard is not met. The owner's and a friend's label sheets were lost in a disk failure on 2026-10-22, and all four sheets that exist were produced by language-model annotators, each from the same 2 x 2 contact sheets, independently and without seeing another sheet. Owner decision of 2026-10-22: sheets A and B are the P041 annotator pair, C and D a sensitivity set; human labels on a subset remain an owner item, and every B0268 number carries this caveat.

| Sheet | Annotator | Role |
|---|---|---|
| A | Claude Fable 5 | protocol pair |
| B | Claude Sonnet 5.5 | protocol pair |
| C | Claude Fable 5.1 | sensitivity |
| D | Claude Opus 5.5 | sensitivity |

| Day | Frames | Sun elevation | Sky (raters' notes) |
|---|---|---|---|
| 2026-10-18 | 120 | 15 to 29 deg | altocumulus field with cumulus puffs, 4 to 7 oktas |
| 2026-10-19 | 120 | 25 to 39 deg | blue sky, isolated cumulus and cirrus, 2 to 6 oktas |
| 2026-10-20 | 120 | 27 to 40 deg | cumulus building under a cirrus / cirrostratus veil, Sun's disc in the frame, 3 to 7 oktas |
| 2026-10-21 | 33 | 49 to 52 deg | hazy, about 1 okta; paused for rain; a dark towering mass after the pause |

Every frame is usable and none is obscured. Agreement of the pair (`docs/data/b0268_labels.md`, `stratia/labels/b0268_labels.py`, `scripts/b0268_labels.py`; alpha = Krippendorff's alpha, agreement = leave-one-out against the other rater, 393 units):

| Quantity | alpha | Agreement |
|---|---|---|
| oktas 0-8 (interval), within ±1 okta / exact | 0.808 | 0.771 / 0.430 |
| étage set, exact | 0.845 | 0.911 |
| étage low / mid / high present | 0.349 / 1.000 / 0.909 | 0.949 / 1.000 / 0.957 |
| genus set, exact ('unsure' rows excluded) | 0.673 | 0.789 |
| genus Ci / Ac / Cu / Cb / Cs present | 0.909 / 1.000 / 0.474 / 1.000 / -0.064 | 0.957 / 1.000 / 0.949 / 1.000 / 0.878 |
| raw low_cloud / mid_cloud / high_cloud | 0.478 / 1.000 / 0.696 | 0.949 / 1.000 / 0.835 |
| raw lowest_base_guess | 0.349 | 0.949 |
| raw sun_glare (ordinal) | 0.487 | 0.679 |
| raw obstruction | 0.623 | 0.982 |

Four-way alpha over A to D, for reference: oktas 0.834, étage set 0.748, genus set 0.357, raw low_cloud 0.491, mid_cloud 1.000, high_cloud 0.279, sun_glare 0.675, obstruction 0.501. High cloud is the contested column (none against Ci on day 1; Ci against Ci+Cs against contrail on day 3). The B0268 ceiling for the C2-style comparison of P083 is the pair's agreement: étage set 0.911, oktas within ±1 0.771 (Montenegro's human ceiling: 0.852 and 0.873). Genus stays a soft target (share of raters), never a hard label; a model is scored on the étage set, total cover and genus in the raters' union. Targets: `data/b0268_targets.parquet` (pair), `data/b0268_targets_all.parquet` (A to D), `data/b0268_ceiling.parquet`; `scripts/map_labels.py` gives the 393 rows label_source "rater majority" and the 25 legacy frames "none".

Protocol and re-lock. `stratia/data/splits.py` `few_shot_target` (`configs/splits.yaml`): the locked test day is the last calendar day with at least 100 labelled frames, 2026-10-20 (the 33-frame 2026-10-21 session is too small to be the test day); the adaptation days are 10-18, 10-19 and 10-21; k in {0, 1, 2} (k = 3 would leave nothing held out, so k = 2 replaces it); draws `k0_d0`, `k1_d0..2`, `k2_d0..2` with roles train / val / test in `data/splits/few_shot_target.parquet`. The manifest keeps the 52,032 locked rows in the same order (hash prefix `464078b6d033dd39` reproduced) and appends the 393 logged frames with the sidecar Sun (`scripts/build_manifest.py`; OLDLR / WESTE stay out). `test_lock.json` was re-locked with `--force`: 23 test sets, 73,376 ids; every previously locked test hash is unchanged (checked against `data/splits/hashes_before_P041.json`); new entries `in_domain/b0268/test` and `few_shot_target/*/test`; the LODO train files lost the 25 legacy b0268 rows, which `GLOBAL_SOURCES` never used; `scripts/check_test_lock.py` passes. Tests: `tests/test_b0268_labels.py`, `tests/test_few_shot_target.py` on the CSV copies of the sheets in `tests/fixtures/b0268_labels/`.

### P083 — B0268 zero-shot evaluation

The exported demo graph (`stratia-1.0.0-demo`, Gate G4 heads with the P089 calibration folded in) on the 273 labelled frames of the three adaptation days (draw `k0_d0`, role val; the locked 2026-10-20 untouched), nominal B0268 camera, zenith pose, manifest Sun, ONNX Runtime on the CPU at 221 ms per frame (`docs/data/b0268_zero_shot.md`, run `20261023T120741Z_b0268_zero_shot_6a9765a3d2`). The model is scored against each rater of the P041 pair and averaged, the way the ceiling (rater against rater: étage set 0.911, oktas within ±1 0.771) is measured, so no two-rater tie is broken.

| Day | Frames | Étage set exact | Oktas within ±1 | Okta bias | Top genus in the rater's set |
|---|---|---|---|---|---|
| all | 273 | 0.031 (3 % of the ceiling) | 0.278 (36 %) | +2.40 | 0.300 |
| 2026-10-18 | 120 | 0.046 | 0.492 | +1.13 | 0.633 |
| 2026-10-19 | 120 | 0.000 | 0.075 | +3.14 | 0.008 |
| 2026-10-21 | 33 | 0.091 | 0.242 | +4.27 | 0.152 |

The honest out-of-distribution number is bad, and it is the global heads: the okta argmax is 6 or 8 on 268 of 273 frames (mean 6.9 against the raters' 4.2), the low étage is predicted on 1 % of frames where a rater names it on 98 %, the top genus is Ns on 152 frames and Cs on 38. The dense head transfers: the sky-parse cloud fraction has correlation 0.83 with the raters' mean cover (MAE 0.107), though it reads the hazy Sun-in-frame sky of 2026-10-21 as 36 % cloud on the 27 frames the raters put at 0 or 1 okta (the no-collapse bar is 20 %), and the glare class is never predicted. The OOD score flags 8 % of the frames (64 % on 2026-10-21, none on 2026-10-18): the detector sees this camera as inside its envelope while the heads are wrong on nearly every frame. Zeroing the geometry inputs (`--no-geometry`, run `20261023T121334Z_b0268_zero_shot_nogeo_a91d5de522`) gives the same picture (étage set 0.081, bias +2.74), so the gap is not in the ray map; the same heads on cached features (P084 draw `k0_d0`) give 0.040 / 0.359. All four raters as truth: 0.037 / 0.291. Consequences: the demo's cover and genus readouts are not trustworthy on this camera until P084 adapts the heads, the sky-parse overlay is, and the "uncertain" badge will stay off on most frames; the paper reports this number with the annotator caveat of P041.

Open for later phases. P041 is done as far as the labels allow: the few-shot B0268 protocol is instantiated and the test sets are re-locked with the B0268 days; what remains of it is human labels on a subset of the frames (owner item) and the 100-frame dense masks the plan names; P042 reviews Stage D; P044 makes the sun explicit; P065's genus loss must handle `genus_alternatives` and may down-weight `label_noise_flags`; P047 pairs frames with `data/ceilometer_targets.parquet`; the evaluation CLI wires `require_final_reason`.

### P084 — Few-shot adaptation on B0268 (reduced: heads only)

Heads-only LP-FT of the Gate G4 global heads (`GeometryEncoder` + `GlobalHeads` on frozen DINOv3 features cached for the 393 labelled frames) with source replay, on the day-blocked draws of P041 (`scripts/adapt_b0268.py`, `docs/data/b0268_adaptation.md`). Truth and scoring as in P083 (each rater of the pair, averaged); the source check is the in-domain validation split (mean étage F1 over the sources, mean genus F1); the rule rejects a draw's model when either drops by more than 0.02. Seven configurations ran on the six draws (42 runs of 20–45 s on the RTX 4050): replay 4 / 8 / 16, LP learning rate 1e-3 / 3e-4, FT learning rate 1e-4 / 3e-5, LP only. Configuration `r16_lp3` (replay 16, LP 5 epochs at 3e-4 on the three output layers, FT 10 epochs at 1e-4 on every head parameter) kept the source best (worst drop 0.030, on one draw) with the second-largest gains; the choice used the held-out days, so its numbers carry the optimism of a selection among seven.

| Draw (train -> held out) | Étage set | Oktas ±1 | Okta bias | Top genus in the rater's set | Source étage / genus F1 change | Rule |
|---|---|---|---|---|---|---|
| k1_d0 (18 -> 19, 21) | 0.026 -> 0.157 | 0.085 -> 0.170 | +3.59 -> +2.66 | 0.007 -> 0.739 | −0.006 / −0.030 | broken |
| k1_d1 (19 -> 18, 21) | 0.072 -> 0.307 | 0.601 -> 0.892 | +1.38 -> −0.23 | 0.542 -> 0.938 | −0.009 / −0.011 | ok |
| k1_d2 (21 -> 18, 19) | 0.029 -> 0.029 | 0.379 -> 0.713 | +1.91 -> +0.42 | 0.350 -> 0.479 | −0.013 / −0.015 | ok |
| k2_d0 (18, 19 -> 21) | 0.121 -> 0.879 | 0.212 -> 0.924 | +4.58 -> +0.91 | 0.000 -> 0.788 | −0.019 / −0.014 | ok |
| k2_d1 (18, 21 -> 19) | 0.000 -> 0.033 | 0.050 -> 0.554 | +3.33 -> +0.64 | 0.008 -> 0.775 | −0.006 / −0.016 | ok |
| k2_d2 (19, 21 -> 18) | 0.058 -> 0.238 | 0.708 -> 0.975 | +0.50 -> −0.24 | 0.692 -> 0.983 | −0.005 / −0.004 | ok |
| mean over the six | 0.051 -> 0.274 | 0.340 -> 0.705 | | 0.267 -> 0.784 | −0.010 / −0.015 | |

What the table says. The okta bias is the transferable part: +2.4 oktas zero-shot, within ±1 on 55–98 % of the held-out frames after one or two days, on every draw, because cover varies within an afternoon. The étage set is not: 2026-10-19 (120 frames whose étage sets, by the raters, no training on the other days matches) stays at 0.00–0.03, 2026-10-18 reaches 0.15–0.24, 2026-10-21 reaches 0.73–0.88 once another afternoon is in training. A day is one afternoon from one camera, so the held-out étage number is a property of the day pair and bimodal; the mean over draws hides it. That is the memorisation guard working. Forgetting grew with less replay and a larger LP step (`base`, replay 4, LP 1e-3: worst drop 0.043; LP only at 1e-3: 0.052) and shrank with 16x replay; the legacy "10x over-biases" did not repeat: at 16x the held-out gains were within 0.03 of 4x on oktas and larger on the étage set.

Final model and export. Run `20261023T123549Z_adapt_b0268_final_r16_lp3_76502c7936` (all three adaptation days, 273 frames, same configuration, seed 0): source étage F1 0.805 (0.817), genus F1 0.701 (0.719), regression 0.018, inside the rule. One of the six draws (`k1_d0`) was outside it by 0.010 on genus F1, so the pre-registered condition "no draw breaks the rule" is not met and the export is a recorded deviation (every configuration had at least one such draw). The P089 calibration was refitted on the adapted heads (`configs/calibration_1.0.1.json`, `docs/data/calibration_1.0.1.md`: genus T 5.00 with ECE 0.156 -> 0.073; étage and oktas keep T = 1; OOD threshold 1.497, 21.9 % of SWIM patches flagged against 25.3 %) and folded into `models/stratia-1.0.1-demo/stratia.onnx` (94.9 MB, SHA-256 `66e22ed1…`, parity ≤ 1.5e-4, CPU 163 ms; card `1.0.1-demo`; dense and CBH heads unchanged). Through the exported graph, on the 273 frames it was adapted on (a fit check, not an evaluation; run `20261023T123919Z_b0268_adapted_adapted_fit_c47e02b737`): étage set 0.905 (the pair's ceiling is 0.911), oktas within ±1 0.681 (0.771), bias +0.32, top genus in the rater's set 0.989, 1.5 % of frames flagged OOD (8.4 % before), sky-parse cover correlation 0.83 as before. The gap between 0.905 fitted and 0.274 held out is the size of the day effect: the demo model is fitted to the demo camera's three afternoons, and its only unbiased B0268 number will be the locked 2026-10-20 at P090, through this graph, once.

### P090 — Locked-test evaluation (once)

The locked test sets were read for the first time on 2026-10-25 (`scripts/locked_test.py --split test --final`, reasons in `runs/final_eval_log.md`, numbers in `results/P090_locked_test.json` with its SHA-256 and `docs/data/locked_test.md`). The protocol, the models (the Gate G4 heads, the four LODO models, the two B0268 graphs, the P054 probes and the P058 ridge refitted on train) and the metrics were pre-registered in `docs/phases/P090-locked-test.md` and the pipeline was first made to reproduce every Stage H validation number exactly. It took two evaluations: the first stopped in the LODO section on a shape bug (a helper that returns predictions where thresholds were needed; invisible while val and test rows coincide in number), before anything was saved or read beyond one OOD line; the second, three minutes later, is the one reported, and the paper says so.

| Criterion | Validation (P082) | Locked test | Reference on test | Holds |
|---|---|---|---|---|
| C1 LODO étage macro-F1, mean of four folds | 0.548 | 0.540 (CCSN 0.451, MGCD 0.709, Montenegro 0.367, Almería 0.634) | probe ViT-B/16 @ 224: 0.529 | no (+0.011, needs +0.05) |
| C1 in-domain genus macro-F1 (CCSN, MGCD) | 0.719 | 0.706 (0.591, 0.821) | probe ViT-L/16 @ 224: 0.733 (0.549, 0.918) | no (−0.027, needs ≥ −0.01) |
| C2 Montenegro étage-set / oktas ±1 agreement over the human ceiling | 1.047 / 1.027 | 0.926 / 0.963 | 0.90 | yes |
| C3 CBH skill OLDLR / held-out WESTE (OLDLR-only training) | 0.471 / 0.448 | 0.536 [0.44, 0.63] / 0.436 [0.28, 0.59] | 0.20 / 0.10 | yes |
| C3 CBH MAE | 566 m | 872 m [679, 1055] | ridge 1,096 m (≤ 986 required) | yes |
| C3 calibrated 80 % interval coverage | 0.802 | 0.720 (raw 0.1–0.9 quantiles 0.878; 95 % calibrated 0.918) | 0.75–0.85 | no |

What the test adds to the validation story. C1 stays negative in both directions: against the strongest probe the model is eleven points better on CCSN and Almería transfer and worse on MGCD and Montenegro, a mean of +0.011 (val −0.035); in-domain the probe on ViT-L features wins MGCD genus (0.918 against 0.821, the model's interval [0.71, 1.00] on 86 same-scene units) and loses CCSN (0.549 against 0.591). C2 comes down from above the expert ceiling to 0.93 / 0.96 of it, above the pre-registered bar, on a test set that is two temporal blocks. C3 is the robust result: the test weeks are a different sky (the training climatology's MAE is 2,012 m on them against 1,147 m on val; high cloud is common), the MAE rises to 872 m and the skill to 0.566 overall, 0.536 on OLDLR and 0.436 at the never-seen site, with the ridge at 1,096 m and no help from the ray map (geometry zeroed 875 m); the split-conformal radius fitted on val under-covers these non-exchangeable weeks (0.72 for 0.80; the 95 % interval still covers 0.92), which the paper reports as the limit of conformal calibration across seasons. Sky parsing holds at 0.900 mIoU (Almería 0.885, SWIMSEG 0.865, SWINySEG 0.872) and the layer head gives 0.668 on the Almería test images (val 0.558); the clear-sky cloud fraction stays at 0.28, above the 0.20 bar, as on val. The OOD score flags 5.4 % of in-domain test rows at the val threshold but 43 % of CCSN and 19 % of SWIM (many cameras per source) and scores AUROC 0.939 for the B0268 day. The B0268 locked day 2026-10-20, never seen by anything: the zero-shot graph is at the floor (exact étage set 0.000, oktas within ±1 0.017, bias +3.5 oktas, half the frames flagged OOD); the heads adapted on the three other afternoons (P084) reach oktas within ±1 0.671 (0.87 of the pair's ceiling; against the majority 0.900 [0.84, 0.95]), the top genus in the rater's set on every frame, and the exact étage set on 0.463 (0.51 of the ceiling; the held-out draws had predicted 0.274), with no frame flagged OOD. Three lock entries have no model (SWIM LODO, the two held-out-station variants) and stay unevaluated; the CBH seeds 1–2 kept no checkpoint, so only the in-domain heads carry a seed spread (std ≤ 0.023).

