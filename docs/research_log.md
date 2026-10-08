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

Open for later phases. P037/P038 take the blocks decided in P027, the sampling of P030 and the `label_conflict` flag of P034; P039 locks the test sets; P040 takes the per-camera masks and the frozen features; P044 makes the sun explicit; P065's genus loss must handle `genus_alternatives`.
