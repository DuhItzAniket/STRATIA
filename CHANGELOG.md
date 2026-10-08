# Changelog

All notable changes, grouped by phase.

## [Unreleased]

### P037 — Split design (2026-10-08)
- `docs/splits.md` and `configs/splits.yaml`: units (groups ∪ temporal blocks), five protocols (in-domain blocked, LODO, held-out station, few-shot target, published for comparison), guarantees for the generator.

### P036 — Ceilometer → targets (2026-10-08)
- Ceilometer target table (`stratia/labels/ceilometer_targets.py`, `scripts/ceilometer_targets.py` → `data/ceilometer_targets.parquet`): per site and 30 s grid point, no-cloud / étage of the median lowest base within ±30 s under weak and WMO thresholds, confidence, base spread, layer count, second layer, daytime flag. Report `docs/data/ceilometer_targets.md`.

### P035 — Montenegro soft labels & human ceiling (2026-10-08)
- Krippendorff's alpha, pairwise and leave-one-rater-out agreement (`stratia/labels/agreement.py`, `scripts/montenegro_ceiling.py`) on raw codes and derived labels; human ceiling for C2: étage set 85.2 %, total cover ±1 okta 87.3 %. Per-image soft targets `data/montenegro_targets.parquet`; report `docs/data/montenegro_ceiling.md`.

### P034 — Dataset → ontology mapping (2026-10-08)
- Set-valued labels for every manifest sample through the ontology (`stratia/labels/mapping.py`, `scripts/map_labels.py` → `data/labels.parquet`): merged classes as alternatives, unknown genera excluded with reasons, P026 conflicts merged and flagged. Report `docs/data/label_mapping.md`. Ontology keys MGCD classes by the manifest's labels.

### P032 — Gate G1: data card (2026-10-08)
- `docs/data/data_card.md` (Datasheets-for-Datasets style) collecting the Stage C audit; gate G1 signed with zero blockers for Paper A, seven open items with owners. Manifest builder parses Almería timestamps from file names (row order unchanged).

### P033 — WMO ontology (2026-10-08)
- `configs/ontology.yaml` with loader and validator (`stratia/labels/ontology.py`): ten genera and étages, the contract's output classes, WMO code tables 0513/0515/0509/1600/2700, every dataset's native classes mapped to genus sets or explicitly excluded.

### P031 — Ceilometer QC & pairing tolerance (2026-10-08)
- Ceilometer QC and pairing study (`stratia/data/ceilometer_qc.py`, `scripts/ceilometer_qc.py`): streams 99.998 % complete, 3.5–6.1 % flagged; pairing tolerance ±30 s chosen from the instrument's own time consistency; the two sites 15 km apart agree on cloud presence only 82 % of the time. Report `docs/data/ceilometer_qc.md`.

### P030 — Imbalance report (2026-10-08)
- Imbalance report (`stratia/data/imbalance.py`, `scripts/imbalance_report.py`): 30 label distributions with imbalance metrics; sampling decided (square-root source sampling, class-balanced loss weights, no resampling of soft labels, macro metrics). `stratia/data/ceilometer.py::load_all` caches all ceilometer records. Report `docs/data/imbalance_report.md`.

### P029 — Shortcut audit (2026-10-06)
- Shortcut audit (`stratia/data/shortcuts.py`, `scripts/shortcut_audit.py`): per-camera mean/std images and fixed-structure masks (25–37 % of all-sky frames; Eye2Sky and Montenegro carry burned-in text), DINOv3 re-embeddings of controlled image variants (low resolution, sky only, fixed structure only) and 25 linear probes split by day or near-duplicate group. Findings: the dataset is readable at 99.7 % and the Eye2Sky station at 100 % from the sky alone; the hour of day at 4–6× chance; Montenegro's text strip predicts its class above chance. Shortcut list with mitigations in `docs/data/shortcuts_report.md`; probes in `data/shortcut_probes.parquet`.

### P028 — MGCD ≟ GRSCD (2026-10-06, PARTIAL)
- Deferred (plan marking D): GRSCD is not reachable; the publications match MGCD in every checkable number, so the two count as one source until a hash comparison can run (`docs/phases/P028-mgcd-grscd.md`).

### P027 — Temporal autocorrelation (2026-10-06)
- Similarity-against-time-gap curves (`stratia/data/temporal.py`, `scripts/temporal_autocorrelation.py`) for Eye2Sky (both stations and across them), Montenegro and Almería, with different-day and same-hour-other-day baselines. Decisions: Eye2Sky and Almería split by calendar day without buffer; Montenegro by contiguous blocks of at least 7 days; a held-out Eye2Sky station is a valid out-of-camera test (same-moment cosine 0.81 vs 0.70, 0.4 % same scene). Report `docs/data/temporal_report.md`, figure `docs/data/figures/temporal_autocorrelation.png`, table `data/temporal_curves.parquet`.

### P026 — Label-conflict audit (2026-10-06)
- Label-conflict audit (`stratia/data/label_conflicts.py`, `scripts/audit_label_conflicts.py`): every exact-duplicate, copy and same-scene pair compared on its dataset's native label and, for segmentation data, on the aligned mask; rater agreement for Montenegro. Findings: 7.9 % of CCSN images carry a second genus label on a copy of themselves; MGCD labels flip in 0.6 % of consecutive-frame pairs and 452 MGCD images straddle the official split; Montenegro raters reach no majority on cloud height for 39 % of images; SWIMSEG images annotated twice agree on 95 % of pixels. Nothing auto-corrected; policy in `docs/data/label_conflicts_report.md`; tables `data/label_conflicts.parquet`, `data/mask_agreement.parquet`.
- `contact_sheet()` accepts a ready-made `caption` column.

### P025 — Near-duplicates (2026-10-06)
- Near-duplicate search (`stratia/data/near_duplicates.py`, `scripts/find_near_duplicates.py`): DINOv3 ViT-S/16 embeddings of all 52,032 images (cached under `cache/features/`, reused by P029), dHash and pHash over all eight flips and rotations, k-nearest-neighbour pairs, thresholds chosen from contact sheets, union-find groups. Same-station Eye2Sky pairs are left to temporal blocking. Report `docs/data/near_duplicates_report.md`, sheets in `docs/data/figures/`.

### P024 — Exact duplicates (2026-10-06)
- Exact-duplicate search by file bytes and by decoded pixels (`stratia/data/duplicates.py`, `scripts/find_duplicates.py`): 661 images (1.3 %) in 312 groups, all within one dataset (SWINySEG 533, SWIMSEG 52, CCSN 34 with 3 label conflicts, SWIMCAT 34, SHWIMSEG 6, Almería 2), none across datasets. Policy: one `group_id` per duplicate group at split time. Tables `data/duplicate_groups.parquet`, `data/image_hashes.parquet`; report `docs/data/duplicates_report.md`.

### P023 — Integrity check (2026-10-06)
- Integrity check of all 52,032 manifest images (`stratia/data/integrity.py`, `scripts/integrity_check.py`): full decode, format, mode, EXIF orientation, size against the manifest, size outliers, contrast. Result: no corrupt, empty, missing, odd-mode, rotated or mis-sized image; 43 featureless clear-sky patches (SWIMCAT, SHWIMSEG) flagged as "blank" and explained. Report `docs/data/integrity_report.md`.
- Added `docs/research_log.md`: the paper's working notes (decisions, numbers with evidence, negative results), now part of the phase protocol.

### P022 — Loader performance (2026-10-04)
- Resized image cache (52,032 images, 3.16 GB, 0 errors) and DataLoader benchmark: 817 img/s to the GPU with 8 workers (3.3× faster than original files).

### P021 — Unified sample schema (2026-10-04)
- Unified 31-column manifest schema with validation (`stratia/data/manifest.py`) and builder: 52,032 images, 11 datasets, 6 camera types; Sun positions where time and location are known.

### P020 — External weak references (METAR) (2026-10-04, PARTIAL)
- Deferred (plan marking D): METAR references for the B0268 site to be implemented with the first real B0268 campaign.

### P019 — B0268 ingest (2026-10-04, PARTIAL)
- B0268 ingest (`stratia/data/b0268.py`): sky-logger sidecars with checksum verification, EXIF fallback for the 25 legacy frames (UTC from IST), locations rounded to 0.01° for privacy.

### P018 — Segmentation datasets loader (2026-10-04)
- Segmentation index and unified label mapping for SWIMSEG, SWINSEG, SWINySEG, SHWIMSEG and Almería (`stratia/data/segmentation.py`); encodings verified from the data (JPEG masks thresholded), QA sheets per dataset.

### P017 — MGCD parser + weather (2026-10-04)
- MGCD reader with official split, classes and weather joined for all 8,000 images (`stratia/data/mgcd.py`).

### P016 — Montenegro parser (2026-10-04)
- Montenegro parser and per-image soft labels (`stratia/data/montenegro.py`), all README counts reproduced, archive image-name prefix handled; experts' majority share on cloud-base height only 0.58.

### P015 — Acquire missing public data (2026-10-04, PARTIAL)
- Verified licences from official records (CCSN CC0, Almería and Montenegro CC BY 4.0, Eye2Sky CDLA-Sharing 1.0, LenghuSky-8 Apache-2.0); found that DeepSky images are only available on request (Zenodo holds the paper only) and WEBCAM has no public link; owner actions in `docs/data/acquisition.md`.

### P014 — Ceilometer reader (2026-10-04)
- CHM15k ceilometer reader with QC flags (`stratia/data/ceilometer.py`), summary over all 236 days and time-height figures; matches the vendor plot; CDLRB file latitude error and CDLRA 0x8000 laser-ageing warning identified and handled.

### P013 — Eye2Sky readers (2026-10-04)
- Eye2Sky readers (`stratia/data/eye2sky.py`) and the OCamCalib fisheye model (`stratia/geometry/ocam.py`); axis convention measured on 38 calibrations (xc = row), mask-name mismatch resolved, round trip ≈ 4e-7 px on OLDLR, WESTE and AURIC.

### P012 — Eye2Sky inventory (2026-10-04)
- `scripts/eye2sky_inventory.py` and `docs/data/eye2sky_inventory.md`: images per station/day, calibration validity windows, ceilometer coverage (118 days each), stations within 1 km of the ceilometers (OLDLR, OLUOL, OLWIN; WESTE); 0 pairable image-days on disk — escalated.

### P011 — Inventory local data (2026-10-04)
- `scripts/inventory.py`: per-file inventory of the data root (dataset, format, size, resolution, mode, junk, unreadable) and `docs/data/inventory.md`; 64,406 files, 15.0 GB, 0 unreadable images; CCSN mixed-resolution shortcut risk recorded.

### P010 — Hardware budget profiling (2026-10-04)
- Measured DINOv3 ViT-S/B/L memory and throughput at 224/384/512 px (inference, full fine-tune, checkpointing, LoRA); `docs/compute.md` with defaults (ViT-S @512, batch 16 or 64 with checkpointing) and Windows sysmem-fallback countermeasures (allocator cap, throughput-collapse guard).

### P009 — Research design & pre-registration (2026-10-04)
- Pre-registered research design (`docs/research_design.md`): hypotheses H1–H4, metrics, Gate G4 criteria C1–C3, locked-test policy, threats to validity; ADR-002 (DINOv3 ViT-S/16 backbone), ADR-003 (single-camera scope); paper outline.

### P008 — Interface contract v1 (2026-10-04)
- `stratia-contract` v1.0 (`docs/contract.md`), class-order constants (`stratia/contract.py`), model-card JSON Schema, ADR-001; Sun-aligned unit-vector ray maps and a 3-value Sun metadata input (no time/date/location shortcuts).

### P007 — Data registry spec (2026-10-04)
- Data registry `configs/datasets.yaml` (12 datasets with source, licence and verification status, paths, image globs, expected counts), machine paths template, registry loader/validator and `scripts/check_registry.py`; all present datasets match their expected image counts.

### P006 — Tests & CI (2026-10-04)
- Unit tests for config, seeding and run bookkeeping; ruff and pytest configuration; GitHub Actions CI (CPU-only PyTorch, lint, tests, gitleaks); pre-commit configuration.

### P005 — Experiment tracking (2026-10-04)
- Metrics log (JSONL + optional TensorBoard), run summary, and `runs/registry.csv` row per run (git SHA, dirty flag, config hash, data hash, status, metrics); robust when no GPU is visible.

### P004 — Config & run system (2026-10-04)
- Config composition with a stable config hash, global seeding/determinism, per-run folders with config and environment snapshot; reproducibility check: two GPU runs with one config gave bit-identical metrics.

### P003 — Project environment (2026-10-04)
- Clean project virtual environment; `requirements.in` (top level) and `requirements.lock` (97 pinned packages, PyTorch 2.5.1 + CUDA 12.1).
- `python -m stratia.env_check`: versions, GPU/driver, CUDA matmul accuracy, fp16/bf16 autocast, NumPy < 2 guard, gated DINOv3 access — all passing.

### P002 — Phase protocol & logs (2026-10-04)
- Added `docs/PLAN.md` (100-phase plan with 2026-10-04 corrections: exact DINOv3 model IDs, Bengaluru airports VOBL/VOBG for METAR, HDD archive + SSD cache storage layout).
- Added phase template, `CONTRIBUTING.md` (phase workflow and research honesty rules), `PROJECT_STATE.md`, this changelog and `EXPERIMENT_LOG.md`.

### P001 — Repo bootstrap (2026-10-04)
- README, Apache-2.0 licence, ignore rules, line-ending rules, `stratia` package skeleton.
