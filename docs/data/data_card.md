# STRATIA data card (Gate G1, P032)

Written in the spirit of *Datasheets for Datasets* (Gebru et al.) for the collection of public sky-camera datasets
and our own captures that STRATIA trains and evaluates on. Every number below comes from a Stage C report under
`docs/data/`; the card is the sign-off of that audit. Date: 2026-10-08. Manifest: `data/manifest.parquet`, 52,032
rows, `sample_id` content hash `464078b6d033dd39` (first 16 hex digits of SHA-256 over the ordered ids).

## 1. Motivation

STRATIA is a single-camera sky-understanding model (cloud genus and étage, total cover, sky parsing, cloud-base
height) meant to transfer across cameras. The collection exists to (a) train and evaluate that model with a protocol
in which leakage, shortcuts and label noise are measured rather than assumed away, and (b) serve as a
leakage-audited, WMO-harmonised benchmark (Paper A). It was assembled by the STRATIA author (undergraduate,
Bengaluru) with the AI assistant that wrote this repository; no funding.

## 2. Composition

| Source | Images | Camera | Labels shipped | Time span | Site | Licence |
|---|---|---|---|---|---|---|
| Eye2Sky (AURIC, BARSE) | 29,288 | fisheye all-sky, 2,112 px, 30 s | none (ceilometer targets at other stations) | 1–9 Apr 2022 in the manifest (Apr–Jul on disk) | Oldenburg region, DE | CDLA-Sharing-1.0 |
| SWIM family: SWIMSEG 1,013, SWINSEG 115, SWINySEG 6,768, SWIMCAT 784, SHWIMSEG 156 | 8,836 | WSI crops and patches, 125–600 px | binary sky/cloud masks; SWIMCAT 5 categories | undated | Singapore | CC-BY-NC-4.0 |
| MGCD | 8,000 | fisheye all-sky, 1,024 px | 7 sky types, official 4,000/4,000 split | undated | Tianjin, CN | research use by agreement |
| CCSN | 2,543 | consumer photographs, 256–400 px | 11 classes (10 genera + contrail) | undated | web | CC0-1.0 |
| Montenegro | 2,522 | fixed low-cost camera, 640 px, 20 min | 5–9 raters: altitude class, N, Nh, h, CL, CM, CH | 3 Oct – 17 Dec 2025 | Montenegro | CC-BY-4.0 |
| Almería (DLR) | 818 | fisheye all-sky, 512 px (Kontas + 3 test cameras) | 5-class masks (camera, sky, low, mid, high); official train/val/test | 2017–2021 (file names) | PSA Almería, ES | CC-BY-4.0 |
| B0268 (ours) | 25 | consumer wide-angle USB, legacy frames | none yet (P041) | 27 Sep 2026 | Bengaluru, IN | owner |
| Ceilometers CDLRA, CDLRB | 1,359,329 records | Lufft CHM15k, 15 s | lowest base, layers, flags | Apr–Jul 2022, 118 days each | Oldenburg region, DE | CDLA-Sharing-1.0 |

Totals: 11 image sources, 6 camera types, 52,032 images, 1.36 M ceilometer records. Not obtained: DeepSky and WEBCAM
(owners asked, no reply), HBMCD and GRSCD (Baidu links not reachable), LenghuSky-8 labels; GRSCD is treated as the
same source as MGCD (P028).

**What is in the images besides sky.** Fixed structure (fisheye corners, horizon objects, mounts) covers 36–37 %
of an Eye2Sky frame, 25 % of MGCD, 26 % of Almería; Eye2Sky carries a burned-in text block (top left), Montenegro a
timestamp and a logo (`docs/data/shortcuts_report.md`, masks in `cache/features/camera_stats_224.npz`).

**Label distributions** (`docs/data/imbalance_report.md`): class sets are nearly balanced (largest/smallest 2–3);
Montenegro is 58 % low cloud with raters almost never above 1.5 km and U-shaped oktas; the segmentation sets hold
1.4 % clear and 2.7 % overcast frames; ceilometer records are 34 % "no cloud overhead", 34 % low, 20 % mid, 12 % high.

## 3. Collection process

Public sources were downloaded from their official records (Zenodo, GitHub, institutional pages; URLs and licence
checks in `configs/datasets.yaml`, `docs/data/acquisition.md`). Eye2Sky images for stations AURIC and BARSE were
downloaded for April–July 2022; the manifest ingests 1–9 April (nine days per station) to keep the audit and the
image cache tractable; the two ceilometer stations (CDLRA at OLDLR, CDLRB at WESTE) have **no images on disk**, so no
image–ceilometer pair exists yet (open item, Paper B). B0268 frames are the 25 legacy captures of the CloudScope
project; the sky-logger campaign for the labelled B0268 set (P041) has not started. Timestamps: Eye2Sky from file
names (UTC), Montenegro from local time stated as UTC+1 (daylight saving not verified), Almería from file names
(time zone not stated, taken as UTC; no sun position derived), B0268 from EXIF (assumed UTC+5.5 h).

## 4. Preprocessing, cleaning, labelling

- **Cache:** every image decoded once and stored at a longest side of 768 px (`cache/img768`, 3.16 GB); models read
  the cache (P022).
- **Integrity (P023):** all 52,032 images decode; no corrupt, empty, mis-sized, rotated or odd-mode file; 43
  featureless clear-sky patches flagged "blank" and kept. `docs/data/integrity_report.md`.
- **Exact duplicates (P024):** 661 images in 312 within-dataset groups (SWINySEG 533, SWIMSEG 52, CCSN 34, SWIMCAT 34,
  SHWIMSEG 6, Almería 2), none across datasets; one `group_id` per group at split time. `docs/data/duplicates_report.md`.
- **Near-duplicates (P025):** copies (dihedral pHash <= 2) and same scenes (DINOv3 cosine >= 0.97): 14,112 images in
  1,656 groups; **34.6 % of SWINySEG are flipped, rotated or re-encoded copies of SWIMSEG/SWINSEG**; MGCD's 8,000
  images are about 227 scenes. The SWIM family is one source. `docs/data/near_duplicates_report.md`.
- **Label conflicts (P026):** 7.9 % of CCSN images carry a second genus on a copy of themselves; MGCD labels are per
  sequence; Montenegro raters reach no majority on cloud height for 39 % of images; SWIMSEG images annotated twice
  agree on 95 % of pixels. Policy: flag, never auto-correct; conflicting pictures never in a test split; training
  labels decided by the harmonisation. `docs/data/label_conflicts_report.md`.
- **Temporal structure (P027):** Eye2Sky frames are the same scene 82–92 % of the time at 30 s and reach the
  different-day level after 8.5–17 h (block = calendar day); Montenegro stays correlated across days (block = 7-day
  run); Almería day blocks; the two Eye2Sky stations are correlated but not duplicates. `docs/data/temporal_report.md`.
- **Shortcuts (P029):** the dataset is readable from frozen features at 99.7 %, the Eye2Sky station at 100 % from the
  sky alone, the hour of day at 4–6× chance; Montenegro's text strip predicts its class above chance. Mitigations:
  out-of-camera protocol, per-camera masks, resolution normalisation, explicit sun input. `docs/data/shortcuts_report.md`.
- **Ceilometer (P014, P031):** streams 99.998 % complete; 3.5–6.1 % of records flagged (rain, window, optics, errors);
  CDLRA's laser-ageing bit on 58 % of records is a warning, not an error; pairing tolerance ±30 s. `docs/data/ceilometer_qc.md`.
- **Labels (Stage D):** `configs/ontology.yaml` maps every native class to WMO genus sets or excludes it (P033);
  mapping, soft labels, ceilometer targets, splits and the test lock follow in P034–P039.

## 5. Uses

Intended: training and evaluating single-camera sky understanding with the protocol of `docs/research_design.md`:
leave-one-dataset-out and held-out-station tests, day- or week-blocked splits, square-root source sampling and
class-balanced losses (P030), macro metrics with per-source and per-sun-zenith reporting, CBH scored against the
ceilometer at the same site only (a cloud base 15 km away differs by 255 m at the median and disagrees on presence
18 % of the time). Not intended: pooled accuracies over mixed sources without per-source numbers (they measure
"which camera"); any claim about genera for SWIMCAT (appearance categories only); training a general cloud-height
model on Montenegro's height code (local, low-cloud, no majority for 39 % of images); any use of the SWIM family
beyond non-commercial research (CC-BY-NC); commercial use of MGCD without its agreement.

## 6. Distribution

STRATIA redistributes no images. The repository holds code, manifests of file paths, derived tables (hashes, groups,
flags, splits) and figures; the figures derived from Eye2Sky are shared under CDLA-Sharing-1.0 terms. B0268 frames
are the owner's; their EXIF GPS is rounded to 0.01° in any published form (open decision on the legacy public
frames). Model weights trained on this collection inherit the most restrictive licence of their training sources
(CC-BY-NC for anything that saw the SWIM family).

## 7. Maintenance

Maintained in the STRATIA repository by its author; versioned by the manifest content hash above, the per-phase
documents under `docs/phases/`, and the test-set lock (P039). Changes to the collection are new phases with new
reports, never silent edits of labels or files.

## 8. Known limitations and open items

| Item | Effect | Owner / phase | Blocks G1? |
|---|---|---|---|
| No images at the ceilometer sites (OLDLR, WESTE) | no image–CBH pairs; Paper B training impossible until downloaded | owner download; P036 (targets ready), P047 (pairing) | no (Paper A unaffected) |
| Manifest holds 9 of ~120 Eye2Sky days per station | few day blocks for Eye2Sky splits | P038 decides whether to ingest more days at a lower cadence | no |
| DeepSky, WEBCAM, HBMCD, GRSCD not obtained | fewer sources; GRSCD counted as MGCD | P015 cut-off 15 Oct | no |
| Almería and Montenegro time zones not verified; no Almería sun position | day blocks safe, hour-level analyses approximate | data card note; P044 ray maps for Almería unavailable | no |
| WMO code-table wording written from memory | appendix must be checked against WMO-No. 306 | P033 open item, before the paper | no |
| B0268 labelled set does not exist | OOD evaluation waits | owner logging now; P041 | no |
| CCSN label noise and cross-étage conflicts | genus scores on CCSN capped | P034 merges sets; test exclusion | no |

## 9. Gate G1 sign-off

| Check | Status |
|---|---|
| Every image decodes and matches the manifest (P023) | passed |
| Exact and near-duplicate groups defined for the splits (P024, P025) | passed |
| Label conflicts listed with a policy (P026) | passed |
| Temporal blocks decided per dataset (P027) | passed |
| Double counting across sources documented (P028) | passed (deferred measurement, policy in place) |
| Shortcut list with mitigations (P029) | passed |
| Imbalance measured, sampling chosen (P030) | passed |
| Ceilometer QC and pairing tolerance (P031) | passed |
| Blockers for Paper A | **0** |

Signed: STRATIA author, 2026-10-08 (commit of P032). Paper B's data dependency (images at the ceilometer sites) is
recorded above as an open item, not a G1 blocker.
