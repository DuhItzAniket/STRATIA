# Changelog

All notable changes, grouped by phase.

## [Unreleased]

### P078 — Budgeted HPO (2026-10-11)
- Budgeted random search for the global heads (`scripts/hpo_global.py`): 20 trials, scores 0.775–0.819, sensitivity table (`docs/data/hpo_global.md`).

### P077 — Stage-5: partial unfreeze (2026-10-11)
- Stage-5 LoRA on the backbone (`scripts/train_lora.py`, peft): étage 0.805 → 0.810, dense mIoU 0.869 → 0.877 frozen → LoRA (`docs/data/stage5_lora.md`).

### P076 — Stage-4: joint multi-task (frozen) (2026-10-11)
- Stage-4 joint training (`scripts/train_joint.py`): étage 0.799, dense mIoU 0.881, CBH MAE 545 m against the single-task runs; task-gradient cosines logged (`docs/data/stage4_joint.md`).

### P075 — Stage-3: CBH head (2026-10-11)
- Stage-3 CBH head regime (`--train-stations`, `--synthetic`, `--day-weight`, checkpoints, `stratia/train/cbh_data.py`): base MAE 557 m skill +0.514 over 3 seeds; held-out WESTE skill +0.448 with OLDLR-only training; + synthetic views 538 m.

### P074 — Stage-2: dense heads (frozen backbone) (2026-10-11)
- Stage-2 dense heads with Kendall weighting and the class-fraction monitor (`--kendall`, `stratia/train/dense_data.py`): val mIoU 0.905, layer mIoU 0.559, clear-sky cloud fraction 0.299.

### P073 — Stage-1: global heads on cached features (2026-10-11)
- Stage-1 global heads on cached features (`stratia/train/data.py`, `global_eval.py`, `scripts/train_global_heads.py`): in-domain val mean étage macro-F1 0.817 (genus 0.626 CCSN, 0.812 MGCD), Montenegro agreement 1.047 / 1.027 of the human ceiling; LODO étage mean 0.485 (sources only) and 0.548 (+ Eye2Sky étage); `docs/data/stage1_global.md`.

### P072 — Training engine (2026-10-11)
- Training engine (`stratia/train/engine.py`): AMP, accumulation, clipping, EMA, atomic checkpoint / resume with RNG and epoch position, NaN guard, early stopping; resume reproduces the uninterrupted run bit for bit (3 tests).

### P071 — Model sanity suite (2026-10-11)
- Full model (`stratia/model/stratia.py`: backbone + geometry + heads, `forward` and `forward_tokens`) and the sanity suite (`stratia/model/sanity.py`): contract shapes, overfit one batch, shuffled labels at chance, constant input, determinism (5 tests on a fake backbone).

### P070 — Multi-task balancing (2026-10-11)
- Multi-task balancing (`stratia/model/balance.py`): Kendall uncertainty weighting over present tasks, task-gradient cosine monitor (2 tests).

### P069 — Multi-dataset sampler (2026-10-11)
- Multi-dataset sampler (`stratia/data/sampler.py`): per-task batches, p_d ∝ n_d^τ with an exact floor, label masks per head; observed shares match the config within 0.02 (3 tests).

### P068 — OOD / quality score (2026-10-11)
- OOD score (`stratia/model/ood.py`, `scripts/ood_eval.py`): energy + Mahalanobis; leave-one-source-out AUROC 0.961–1.000 (mean 0.978), the Mahalanobis term carries it; night and glare flags; `docs/data/ood_eval.md`.

### P067 — CBH head (2026-10-11)
- CBH head (`stratia/model/cbh.py`, `scripts/train_cbh_head.py`; pooled 8 × 8 token cache via `FeatureCache.build(pool=4)`): val MAE 566 m, skill +0.506 (P058 best 699 m / +0.390); with geometry dropout 0.3 the head ignores the geometry (zeroed: 566 m), without dropout it uses it (551 m vs 765 m zeroed); `docs/data/cbh_head.md`.

### P066 — Layer head (2026-10-11)
- Layer head on the shared decoder trunk (`--layers`): Almería layer mIoU 0.559 [0.467, 0.661] (bootstrap 95 %), sky parsing 0.894 jointly.

### P065 — Sky-parsing head (2026-10-11)
- Dense decoder + sky-parsing head (`stratia/model/dense.py`, `scripts/train_dense_heads.py`): val mIoU 0.919 on the grouped segmentation val (P057 token probe 0.875), clear-sky cloud fraction 0.220 (probe 0.323); `docs/data/dense_heads.md`. P057's data helpers moved into `stratia/baselines/segmentation.py`.

### P064 — Taxonomy & soft-label losses (2026-10-11)
- Losses (`stratia/model/losses.py`) and CBH bins (`stratia/model/cbh_bins.py`): set-valued BCE, soft BCE, hierarchy consistency, CORAL, KL, dense CE, CBH CE + CRPS, CBH–étage consistency, all masked with an exact zero for empty masks (6 toy-gradient tests).

### P063 — Global heads (2026-10-11)
- Global heads (`stratia/model/heads.py`): `AttentionPool`, `CoralHead` (rank-consistent oktas), `GlobalHeads` → genus / étage logits and oktas probabilities in the contract's shapes (4 tests).

### P062 — SkyRay encoder (2026-10-11)
- `GeometryEncoder` (`stratia/model/skyray.py`): tokens + α · SkyRay tokens, FiLM on CLS and tokens, `zenith_cos`; one traced graph serves known and unknown geometry (4 tests).

### P061 — Architecture spec (2026-10-11)
- Architecture spec: `docs/architecture.md` (diagram, shape table, contract → module → loss map, frozen-feature training path), ADR-005 (heads: attention pool ⊕ CLS, CORAL oktas, light DPT-style dense decoder, zenith-prior CBH attention over 24 étage-aligned log bins + no-cloud, energy + Mahalanobis OOD), ADR-006 (losses: set-valued genus BCE with partial negatives, soft BCE, CORAL with soft cumulative targets, hierarchy consistency, dense CE, CBH ordinal CE + CRPS, layer consistency, Kendall weighting).

### P060 — Gate G3 — Baselines reproduced (2026-10-11)
- Gate G3 signed (`docs/phases/P060-gate-g3.md`): seed variance and plausibility against the literature reviewed.

### P059 — Baseline leaderboard (2026-10-11)
- `scripts/baseline_leaderboard.py` → `docs/data/baseline_leaderboard.md`: recognition (in-domain, LODO), protocol gap, segmentation, CBH, and the best-baseline references for criteria C1 and C3.

### P058 — B7: CBH baselines (2026-10-11)
- Cloud-base-height baselines on the P047 pairs (`stratia/baselines/cbh.py`, `scripts/baseline_cbh.py`): climatology, étage lookup, ridge on frozen features, ceilometer-supervised ResNet-18; MAE / bias / skill by height bin (`docs/data/baseline_cbh.md`).

### P057 — B6: segmentation baselines (2026-10-11)
- Segmentation baselines (`stratia/baselines/segmentation.py`, `scripts/baseline_segmentation.py`): small U-Net from scratch and a linear probe on DINOv3 patch tokens, per-dataset IoU and the clear-sky collapse check; SAM 3 deferred.

### P056 — B5: small VLM (2026-10-11)
- Qwen3-VL-2B zero-shot sanity check in fp16 on 100 val images of CCSN and MGCD (`scripts/baseline_vlm.py`); PARTIAL: no 4-bit, one prompt.

### P055 — B4: CLIP / SigLIP 2 (2026-10-11)
- CLIP ViT-B/16 and SigLIP 2 zero-shot with name and WMO-definition prompts fixed in `configs/prompts.yaml` (`scripts/baseline_zeroshot.py` → `docs/data/baseline_zeroshot.md`; best 0.413) plus linear probes on their embeddings (P055 rows of `docs/data/baseline_probes.md`).

### P054 — B3: frozen probes (2026-10-11)
- Linear and k-NN probes on the frozen caches, in-domain and LODO étage (`stratia/baselines/tasks.py`, `probes.py`, `scripts/baseline_probes.py` → `docs/data/baseline_probes.md`); best in-domain mean 0.765 (('dinov3_vitb16', 224)), best LODO étage mean 0.583 (('dinov3_vitb16', 224)). LODO amendment: folds carry a val role (config version 2; research design §8); test lock re-locked for the five LODO entries.

### P053 — B2: fine-tuned CNNs (2026-10-11)
- ResNet-50 and ConvNeXt-T fine-tuned on CCSN, MGCD, SWIMCAT and Montenegro, 3 seeds each (`stratia/baselines/train_cnn.py`, `scripts/baseline_cnn.py`), with per-epoch fit curves and verdicts; largest seed std 0.026.

### P052 — B1: legacy CNN (2026-10-11)
- Legacy MobileNetV3-L recipe on CCSN under a per-image random split and the grouped split, same pool and sizes, 3 seeds (`scripts/baseline_cnn.py --recipe legacy`): random val macro-F1 0.511 vs grouped val 0.588; the random val's leaked images (a near-duplicate in train, 24 of 255) score 0.146 against 0.543 for the clean ones (`docs/data/baseline_cnn.md`). The locked test rows are not used.

### P051 — Feature cache (2026-10-11)
- Frozen-feature caches (`stratia/features/backbones.py`, `cache.py`, `scripts/build_feature_cache.py`, `scripts/cache_cbh_frames.py`): pooled fp16 embeddings for every manifest image plus the 10-minute cloud-base-height frames for DINOv3 S/B/L, DINOv2 S/B, CLIP ViT-B/16 and SigLIP 2 B/16 at 224 px (ViT-S also at 512 px) and the ViT-S patch grid for the dense subsets, keyed by backbone, size and transform version with a fingerprint and a resumable build; `docs/data/feature_cache.md`. DINOv3 ViT-L runs in bfloat16 (fp16 overflowed); OpenAI CLIP is converted to safetensors once.

### P047 — Image–ceilometer pairing (2026-10-11)
- Pair table at the ceilometer sites (`stratia/labels/cbh_pairs.py`, `scripts/make_cbh_pairs.py` → `data/cbh_pairs.parquet`): frames every 2 min joined to the P036 targets within 15 s, Sun-fitted pose required, zenith pixel and 10° ROI per pair; 104,683 daytime pairs, 104,683 in the day splits (test 21,859, train 71,867, val 10,957); new protocol file `data/splits/cbh_pairs.parquet` and lock entries (`test_lock.json` re-locked); report `docs/data/cbh_pairs.md`, figures. P044 refit of OLDLR/WESTE with a 15° prior gate (dome ghost tracks) and daily sampling; P049's `label_from_pairs` joins views to this table.

### P050 — Metadata dropout (2026-10-10)
- `GeometryDropout` (ray map / meta masked to the contract's null state, p = 0.3 each; `stratia/augment/dropout.py`) and the null-token path of the SkyRay encoder with meta FiLM (`stratia/model/skyray.py`); a stub model runs with and without metadata.

### P049 — Consumer-view synthesis (2026-10-10)
- Re-projection of Eye2Sky frames into B0268-like 105° views at random pointings with exact ray maps and ceilometer labels inherited under the flat-layer assumption (`stratia/geometry/synthesis.py`, `scripts/synthesize_views.py` → `data/synthetic_views.parquet`, images under the cache); report `docs/data/synthetic_views.md`, figure `docs/data/figures/synthetic_views.jpg`.

### P048 — Augmentation policy (2026-10-10)
- Physical photometric augmentations (`stratia/augment/photometric.py`: exposure, white balance on the colour-temperature axis, JPEG, sensor noise, Sun-aware glare with ghost, dirt and raindrops, obstruction cut-outs), per-camera-type geometric policies through the P046 transforms with dense labels carried along (`configs/augment.yaml`, `stratia/augment/policy.py`), `strength` and per-augmentation `enabled` for the ablations; gallery `docs/data/figures/augmentation_gallery.jpg`, `docs/data/augmentation_gallery.md`. No hue shift anywhere.

### P046 — Geometry-consistent transforms (2026-10-10)
- ADR-004 (native view + ray maps; augmentations are pixel remappings); `GeoState` and the transforms Resize, Crop, RandomResizedCrop, HorizontalFlip, Rotate, Compose with `geometry_for` recomputing the ray map through the composed map (`stratia/geometry/transforms.py`); 8 consistency tests (coordinate image, Sun stays at the Sun, flip/rotation of the zenith map, padding invalid).

### P045 — Ray-map generator (2026-10-10)
- Contract ray maps and meta vector from camera, pose, Sun, mask and a near-horizon floor through one affine (`stratia/geometry/raymap.py`, `configs/near_horizon.csv`); overlays and Sun-patch check on four stations (`scripts/ray_map_overlays.py` → `docs/data/ray_maps.md`, figures).

### P044 — Sun position & calibration validation (2026-10-10)
- Sun position and ENU/Sun-aligned frames (`stratia/geometry/sun.py`), camera poses with Kabsch fitting and 768 readings of Eye2Sky's declared orientation (`stratia/geometry/pose.py`), a Sun-disc detector built on measured facts (clip at 240, peak gate, glow limit; `stratia/geometry/sun_detect.py`), `scripts/sun_validation.py` → `configs/camera_poses.yaml` (4 fitted poses, median 0.08–0.12°, all ok), `docs/data/sun_validation.md`, figures. Declared orientation = Ry(pitch)·Rx(−roll)·Rz(yaw) camera-to-world on the OCamCalib frame; WESTE's March file is 2.9° off.

### P043 — Camera models (2026-10-10)
- One camera interface (`stratia/geometry/cameras.py`): OCamCalib, OpenCV fisheye/pinhole in NumPy (cross-checked against cv2), CloudScope camera-model file loader (`schemas/camera_model.schema.json`), unknown-camera placeholder, registry `configs/cameras.yaml`; provisional B0268 datasheet model (`configs/cameras/b0268_nominal.camera.json`); shared test vectors `tests/vectors/camera_test_vectors.json` (`scripts/make_camera_test_vectors.py`).

### P042 — Gate G2: splits & labels sign-off (2026-10-10)
- Review of P033–P041 with evidence (`docs/phases/P042-gate-g2.md`); mechanical checks rerun (149 tests, test-set lock, lint); gate signed conditionally with P041 deferred (no B0268 frames exist; latest before P083).

### P041 — B0268 labelling protocol (2026-10-10)
- BLOCKED on the owner's sky-logger frames and a second labeller; requirements and the day-coverage rule recorded (`docs/phases/P041-b0268-labelling.md`).

### P040 — Label-noise estimation (2026-10-08)
- Confident-learning flags on frozen features with a strong tier (`stratia/labels/noise.py`, `scripts/label_noise.py` → `data/label_noise_flags.parquet`); CCSN 29 %, MGCD 7 %, Montenegro 11 %, SWIMCAT 0 strong flags; sheets reviewed; nothing deleted. Report `docs/data/label_noise_report.md`.

### P039 — Lock test sets (2026-10-08)
- `test_lock.json` with the SHA-256 of 13 test sets (`stratia/data/test_lock.py`, `scripts/lock_test_sets.py`); final evaluations need a logged reason; CI step `scripts/check_test_lock.py` refuses training configs that reference a test set.

### P038 — Split generator (2026-10-08)
- Split generator with proven guarantees (`stratia/data/splits.py`, `scripts/make_splits.py`): units from exact groups and temporal blocks (near-duplicate links only where no block exists), greedy stratified assignment, LODO folds, held-out station, hashes. Outputs `data/splits/*.parquet`, `data/splits/hashes.json`; report `docs/data/splits_report.md`.

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
