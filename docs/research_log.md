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

Open for later phases. P027 measures how fast frames decorrelate in the time-series datasets (Eye2Sky 30 s, Montenegro 20 min); P029 takes Montenegro's burned-in timestamp and logo as shortcut candidates; P033 and P034 take the conflict lists (`data/label_conflicts.parquet`, `data/mask_agreement.parquet`).
