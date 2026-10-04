# STRATIA — Training & Research Implementation Plan (100 phases)

**STRATIA** — *Spatio-Temporal Representation for Atmospheric Intelligence and Analysis*
Repo: https://github.com/DuhItzAniket/STRATIA · Plan date: 2026-10-04 · Compute: 1× RTX 4050 Laptop (6 GB), 16 GB RAM

---

## 0. What STRATIA is (scope lock)

STRATIA is **a model, not a system**. Given **one image from one sky camera** (plus optional time, location and calibration metadata), it outputs:

| Head | Output | Supervised by |
|---|---|---|
| Sky parsing (dense) | invalid/obstruction · sky · cloud · sun-glare | SWIMSEG family, Almería, (LenghuSky-8), labelled B0268 subset |
| Cloud layer (dense) | low · mid · high on cloud pixels | Almería layer masks; Eye2Sky ceilometer weak labels |
| Cloud genus (global, multi-label) | 10 WMO genera + clear + contrail | CCSN, MGCD, DeepSky, WEBCAM, HBMCD, Montenegro (soft labels) |
| Étage (global) | low / mid / high present | ontology mapping of all genus labels; Montenegro CL/CM/CH |
| Cloud cover (global, ordinal) | 0–8 oktas | Montenegro N (annotator distribution); derived from sky parsing |
| Cloud-base height (global, ordinal distribution) | K log-spaced bins 0–12 km + "no cloud overhead" | **Eye2Sky CHM15k ceilometers** (CDLRA, CDLRB); Montenegro `h` codes |
| Reliability | OOD / quality score | unsupervised (energy + feature-distance) |

**Single-camera by design.** Multi-camera stereo is a *future* extension (Section 9), not part of v1. Everything trains on your laptop; a rented GPU is optional for multi-seed sweeps only.

**Scientific claim we are aiming for (pre-registered in P009):** a single, geometry-aware, foundation-model-based network can (1) recognise clouds across cameras better than published-style CNNs once leakage is removed, (2) reach close to the human expert-agreement ceiling, and (3) estimate cloud-base height from one image with calibrated uncertainty, with skill over climatology — the thing current geometry foundation models (Depth Anything 3, VGGT, MoGe-2, MASt3R) explicitly refuse to do because they mask the sky.

---

## 1. Architecture (v1 design — finalized in P061)

```
image (512×512) ──► DINOv3 ViT-S/16 backbone (frozen → LoRA later) ──► patch tokens (32×32×384) + CLS
ray_map (32×32×4) ──► SkyRay MLP ──────────────────────────────────────► + added to patch tokens
meta (sun pos, solar hour) ──► Meta MLP ──► FiLM on CLS                  (null tokens when unknown)
                                         │
     ┌──────────────┬───────────────┬────┴──────────┬───────────────┬──────────────┐
  sky-parse head  layer head     genus/étage head  oktas head     CBH head        OOD score
  (light DPT       (same decoder,  (attention-pool   (CORAL         (attention-pool  (energy +
   decoder, ¼ res)  3 classes)      + MLP, sigmoid)   ordinal)        over zenith-     Mahalanobis)
                                                                     weighted tokens,
                                                                     ordinal bins)
```

Key design choices (each gets an ADR):
- **Backbone:** DINOv3 ViT-S/16 (21 M, `facebook/dinov3-vits16-pretrain-lvd1689m`) primary — fits 6 GB for fine-tuning; ViT-B/16 (86 M, `facebook/dinov3-vitb16-pretrain-lvd1689m`) as ablation; ViT-L/16 (300 M, `facebook/dinov3-vitl16-pretrain-lvd1689m`) only for frozen feature extraction. The 7B model (26.9 GB of weights) does not fit the 6 GB GPU and is not used. Loaded through `transformers` 5.5.4 (gated access granted 2026-10-04).
- **SkyRay geometry encoding:** per-patch zenith angle, azimuth relative to the sun, angular distance to the sun, valid flag. Lets one model serve fisheye all-sky imagers *and* the 105° B0268, and tells the CBH head where "zenith" is. **Geometry dropout** (p≈0.3) during training so the model still works on datasets with no calibration (CCSN, WEBCAM).
- **Set-valued taxonomy loss:** `L = −log Σ_{c∈S(y)} p(c|x)` so coarse labels (e.g. MGCD "stratocumulus" or "mixed") train the fine head without being forced to one genus.
- **Soft labels from experts:** Montenegro annotator distributions → KL loss; lets us measure the model against the human ceiling.
- **CBH as ordinal distribution**, not a single number: honest about ambiguity (multi-layer, overlapping étage ranges). Loss = ordinal cross-entropy + CRPS; consistency term with layer head.
- **Edge student** (P095): distilled small model for Raspberry Pi 5 / CloudScope CPU mode.

---

## 2. Phase protocol (every phase, no exceptions)

1. Create `docs/phases/P###-<slug>.md` from the template below **at the start** of the phase.
2. Do the work. Record commands, configs (with hash), outputs, plots.
3. Verify **exit criteria**. Add evidence to the doc.
4. Update `PROJECT_STATE.md` (current phase, last good result, blockers) and `CHANGELOG.md`.
5. `git add` (code + docs only; data/weights are ignored) → `git commit -m "P###: <title>"` with body + trailer `Phase-Status: DONE|PARTIAL|BLOCKED` → `git push origin main`.
6. End of stage: `git tag stage-<X>-complete && git push --tags`.

**Phase doc template**
```markdown
# P### — <title>
Status: DONE | PARTIAL | BLOCKED     Date: YYYY-MM-DD     Commit: <sha>
## Objective
## Inputs / dependencies (phases, data, configs)
## Work log (what was done, decisions, ADR links)
## Verification (commands, outputs, metrics, figures)
## Exit criteria — checklist with evidence
## Fit & data-risk notes (over/underfitting, leakage, shortcuts observed)
## Deviations from plan & why
## Next phase
```

**Honesty rules (inherited from CloudScope's good practice):** no fabricated numbers or labels; every metric traceable to a run directory; the locked test set is never used for decisions; negative results are written up, not deleted.

---

## 3. The 100 phases

Legend — **CP** = on the CVPR critical path (must be done by Gate G4); **D** = deferrable until after 16 Nov submission. Guard column = the specific over/underfitting or data risk the phase must check.

### Stage A — Foundation (P001–P010)

| ID | Phase | Work | Exit criteria | Guard | |
|---|---|---|---|---|---|
| P001 | Repo bootstrap | README, LICENSE, `.gitignore` (data/, runs/, caches/, *.pth, *.onnx>50MB), skeleton: `stratia/` (package), `configs/`, `scripts/`, `tests/`, `docs/{phases,adr,data}`, `paper/` | First push to GitHub succeeds; CI badge placeholder | secrets never staged | CP |
| P002 | Phase protocol & logs | Phase template, `PROJECT_STATE.md`, `CHANGELOG.md`, `EXPERIMENT_LOG.md`, commit convention, honesty rules | Template used by P001–P002 retroactively | — | CP |
| P003 | Project environment | Project-local `.venv` (Py 3.11), pinned `requirements.txt`: torch 2.5.1+cu121, torchvision 0.20.1, transformers 5.5.4, timm (upgrade), albumentations, pandas, pyarrow, xarray, netCDF4, h5py, zarr, scipy, scikit-learn, pvlib, imagehash, opencv, onnx, onnxruntime-gpu, tensorboard, omegaconf/hydra, pytest, ruff; `stratia.env_check` | env_check prints GPU, driver, CUDA, versions; CUDA matmul + AMP test pass | numpy<2 pin (CloudScope broke on NumPy 2) | CP |
| P004 | Config & run system | Hydra/OmegaConf configs; run dir `runs/<date>_<name>_<cfghash>/`; global seeding; config hash logged | Two runs with same config produce identical first-epoch metrics | non-determinism documented | CP |
| P005 | Experiment tracking | TensorBoard + JSONL metrics + `runs/registry.csv` (run id, git sha, cfg hash, data hash, metrics) | Registry row auto-written by a dummy run | — | CP |
| P006 | Tests & CI | pytest, ruff, pre-commit (incl. gitleaks); GitHub Actions CPU-only tests on synthetic tiny data | CI green on push | — | CP |
| P007 | Data registry spec | `configs/datasets.yaml`: source URL, licence, version, local path, file count, SHA manifest path | All local datasets registered | licence recorded per dataset | CP |
| P008 | Interface contract v1 | `docs/contract.md` + `model_card.schema.json` (see plans/README) ; ADR-001 | CloudScope maintainers (you) sign off; mirrored in CloudScope P065 | — | CP |
| P009 | Research design & pre-registration | Problem statement, hypotheses H1–H3, tasks, metrics, **pre-registered success criteria**, locked-test policy, venue gates; ADR-002 backbone, ADR-003 single-camera scope; paper outline started | Doc committed **before** any model training | prevents p-hacking / test peeking | CP |
| P010 | Hardware budget profiling | VRAM & throughput for DINOv3 ViT-S/B/L at 224/384/512: inference, fwd+bwd with AMP, grad-checkpointing, LoRA; WDDM overhead | Table in `docs/compute.md`; chosen default batch sizes | OOM risks known before training | CP |

### Stage B — Data acquisition & inventory (P011–P022)

| ID | Phase | Work | Exit criteria | Guard | |
|---|---|---|---|---|---|
| P011 | Inventory local data | Scan `DATA/`: counts, sizes, formats, resolutions → `data/inventory.parquet` | Report matches known counts (CCSN 2,543; MGCD 8,000; Montenegro 2,522; …) | junk files (`__MACOSX`, `._*`) flagged | CP |
| P012 | Eye2Sky inventory (incl. new download) | Stations × dates × image counts; calibration validity windows; ceilometer coverage; **list frames co-located with CDLRA/CDLRB** | Coverage matrix; number of usable (image, CBH) pairs known | if no co-located images ⇒ escalate to you before P047 | CP |
| P013 | Eye2Sky readers | Calibration YAML parser (OCamCalib poly, centre, affine, extrinsics; handle 0-byte YAMLs), `.mat` mask loader, filename→UTC + exposure parser | Unit tests on 3 stations; pixel/ray round-trip error < 0.1 px | wrong calib period per date | CP |
| P014 | Ceilometer reader | CHM15k NetCDF3 via scipy: `cbh`, `cbe`, `cdp`, `tcc`, `bcc`, `sci`, `time` (s since 1904); QC flags; daily plots | 1 week plotted; matches ceilometer `plots/` PNGs visually | time-zone/epoch bugs | CP |
| P015 | Acquire missing public data | DeepSky images (Zenodo 8208505), WEBCAM images (GitHub), HBMCD (GitHub), GCD (if TJNU grants), verify checksums | Registered in P007 registry | licences recorded | CP (DeepSky), D (others) |
| P016 | Montenegro parser | `dtype=str`; per-image annotator distributions for N, Nh, h, CL, CM, CH, altitude class; merge "Low"/"Low clouds" | Per-image soft labels table; counts match README | annotator 72–75 sparsity noted | CP |
| P017 | MGCD parser + weather | Class folders + xlsx covariates (temp, RH, pressure, wind) joined by image | Join rate 100% or documented | weather used only as analysis, not input (single-camera scope) | D |
| P018 | Segmentation datasets loader | SWIMSEG/SWINSEG/SWINySEG/SHWIMSEG, Almería (5 classes), optional LenghuSky-8 labels → unified label IDs + ignore=255 | Visual QA sheet per dataset | label-ID mismatches (CloudScope Exp-A bug class) | CP |
| P019 | B0268 ingest | Read CloudScope captures + JSON sidecars (UTC, exposure, gain, pose); fallback: EXIF/filename | Ingests the 25 legacy frames + first logger batch | timestamps must be UTC | CP |
| P020 | External weak references | METAR archive for Bengaluru airports VOBL (Kempegowda) and VOBG (HAL) from the Iowa Environmental Mesonet, ERA5 notes (deferred) | METAR parser + 1 month downloaded | METAR is coarse — evaluation only | D |
| P021 | Unified sample schema | `data/manifest.parquet`: sample_id, path, dataset, camera_id, site, utc, lat/lon, sun_zen/az, calib_id, genus_set, etage_set, oktas_dist, h_dist, seg_path, layer_path, cbh_layers, licence, group_id, split | Manifest built for all datasets; schema validated in tests | group_id defined for leakage control | CP |
| P022 | Loader performance | Pre-resized 512-px cache for Eye2Sky 2112×2048 JPEGs; Windows DataLoader workers; decode benchmark | ≥ 300 img/s to GPU at 512 px or documented bottleneck | — | CP |

### Stage C — Data quality & leakage audit (P023–P032)

| ID | Phase | Work | Exit criteria | Guard | |
|---|---|---|---|---|---|
| P023 | Integrity check | Corrupt/zero-byte/odd-mode images, EXIF rotation, size outliers | 0 unexplained failures | — | CP |
| P024 | Exact duplicates | SHA-256 within and across datasets | Duplicate table; resolved policy | train/test duplicates = leakage | CP |
| P025 | Near-duplicates | pHash/dHash + DINOv3 embedding cosine clustering; contact sheets for manual review | Threshold chosen from review; clusters become `group_id`s | near-dup leakage inflates accuracy 20–40 pts | CP |
| P026 | Label-conflict audit | Duplicates with different labels; flag, don't auto-fix | Conflict list reviewed | noisy labels | CP |
| P027 | Temporal autocorrelation | Similarity vs time-gap curves for Eye2Sky (30 s), Montenegro (20 min), DeepSky | Minimum block size / gap per dataset decided | adjacent frames across splits | CP |
| P028 | MGCD ≟ GRSCD | Hash comparison vs published GRSCD (if obtainable) | Documented | double counting | D |
| P029 | Shortcut audit | Linear probe predicting *dataset ID* and *camera ID* from frozen features; check borders, watermarks, timestamp text, fisheye corners, resolution | Shortcut list + mitigations (mask corners, resolution normalization, text inpainting) | model learning "which camera" instead of "which cloud" | CP |
| P030 | Imbalance report | Class/étage/oktas/CBH-bin distributions per dataset and split | Report + sampling strategy chosen | minority collapse | CP |
| P031 | Ceilometer QC & pairing tolerance | Invalid ranges, rain/fog (`sci`), layer counts; agreement of CBH within ±30 s / ±2 / ±5 min windows | Pairing tolerance chosen with justification | label noise from time mismatch | CP |
| P032 | **Gate G1 — Data audit sign-off** | Data card (Datasheets for Datasets style) | Signed; blockers = 0 | — | CP |

### Stage D — Label harmonization & splits (P033–P042)

| ID | Phase | Work | Exit criteria | Guard | |
|---|---|---|---|---|---|
| P033 | WMO ontology | `configs/ontology.yaml`: genus ↔ étage ↔ CL/CM/CH codes ↔ each dataset's classes | Reviewed against WMO Cloud Atlas & WMO-No. 306 tables | — | CP |
| P034 | Dataset → ontology mapping | Set-valued labels (e.g. MGCD "mixed" → multi; SWIMCAT only cloudy/clear level; WEBCAM fog → obscured) | Every label in every dataset mapped or explicitly excluded | forced single-genus mapping = label noise | CP |
| P035 | Montenegro soft labels & human ceiling | CL/CM/CH → genus sets; `h` → height bins; N → oktas; per-code Krippendorff α; leave-one-annotator-out agreement | Ceiling table (the bar STRATIA is compared to) | — | CP |
| P036 | Ceilometer → targets | Zenith CBH (lowest layer), layer count, étage of lowest layer, "no cloud overhead"; weak layer thresholds (low <2 km, mid 2–6 km, high >6 km; ablate) | Target table for all paired frames | multi-layer ambiguity recorded, not discarded | CP |
| P037 | Split design | Grouped / temporal / site / camera splits; **leave-one-dataset-out (LODO)**; few-shot target protocol | `docs/splits.md` reviewed | — | CP |
| P038 | Split generator | Implementation with guarantees: no `group_id` crosses splits; temporal gaps; manifests hashed; unit tests | Tests prove no overlap (hash + group + time) | leakage | CP |
| P039 | **Lock test sets** | Test manifests + SHA → `test_lock.json`; `eval --final` requires a logged reason; CI check that training configs never reference test manifests | Lock committed | test peeking | CP |
| P040 | Label-noise estimation | Confident-learning style scores on frozen features; manual review of top flagged; **no automatic deletion** | Flag list + decisions | over-cleaning biases data | D |
| P041 | B0268 labelling protocol | Guide with WMO examples; Label Studio/CVAT local; 2 annotators; image-level (genus set, étage, oktas) + 100-frame dense (sky/cloud/glare/obstruction) | ≥300 frames labelled by 2 people; agreement reported | single-annotator bias | CP (image-level), D (dense) |
| P042 | **Gate G2 — Splits & labels sign-off** | Review of P033–P041 | Signed | — | CP |

### Stage E — Geometry & metadata (P043–P050)

| ID | Phase | Work | Exit criteria | Guard | |
|---|---|---|---|---|---|
| P043 | Camera models | OCamCalib (Eye2Sky), OpenCV fisheye/pinhole (B0268), "unknown" placeholder; pixel↔ray | Round-trip tests pass | — | CP |
| P044 | Sun position & calibration validation | pvlib SPA per sample; project sun into image; compare with detected sun blob on Eye2Sky | Median sun-projection error reported per station; bad calibs excluded | wrong extrinsics silently corrupt CBH labels | CP |
| P045 | Ray-map generator | Per-patch zenith angle, azimuth-from-sun, sun distance, valid flag; camera mask + `near_horizon.csv` | Visual check overlays | — | CP |
| P046 | Geometry-consistent transforms | Native view + ray maps (ADR-004); crops/resizes/rotations transform ray maps identically | Unit test: transform(image, ray) consistency | augmentation that breaks geometry = silent label noise | CP |
| P047 | Image–ceilometer pairing | Zenith ROI, pairs at OLDLR/OLUOL↔CDLRA and WESTE↔CDLRB using P031 tolerance; subsample (e.g. 1 per 2–5 min) to limit autocorrelation | Pair table with counts per height bin | over-representation of slow-changing overcast days | CP |
| P048 | Augmentation policy | Photometric: exposure, white balance within physical range, JPEG, noise, synthetic glare/flare, dirt/raindrops; geometric: rotation about zenith (all-sky), horizontal flip (consumer), crop; obstruction cut-outs. **No hue shifts that change sky colour** | Visual gallery approved; per-aug ablation slot in P080 | aug that destroys cloud texture ⇒ underfit; too weak ⇒ overfit | CP |
| P049 | Consumer-view synthesis | Re-project Eye2Sky fisheye into 105° perspective views at random tilt/azimuth (B0268-like) with correct ray maps and inherited CBH labels | Synthetic set + visual check | bridges all-sky → consumer camera domain gap | CP |
| P050 | Metadata dropout | Train-time masking of ray map / meta (p≈0.3) with null tokens | Model runs with and without metadata | over-reliance on metadata shortcut (e.g. time-of-day ⇒ class) | CP |

### Stage F — Baselines (P051–P060)

| ID | Phase | Work | Exit criteria | Guard | |
|---|---|---|---|---|---|
| P051 | Feature cache | Frozen backbone features → fp16 memmap/zarr keyed by (sample, transform-version, backbone) | Cache reproducible; disk budget logged | stale cache after transform change ⇒ version keys | CP |
| P052 | B1: legacy CNN | Re-run CloudScope MobileNetV3-L on random split *and* new grouped split | Protocol-gap number (random vs honest) | — | CP |
| P053 | B2: fine-tuned CNNs | ResNet-50, ConvNeXt-T; 3 seeds | Mean ± std | — | CP |
| P054 | B3: frozen probes | DINOv2 & DINOv3 S/B/(L frozen) linear + kNN | Table | — | CP |
| P055 | B4: CLIP/SigLIP 2 | Zero-shot (names vs WMO-definition prompts) + linear probe | Table | prompt overfitting to val — fix prompts before test | CP |
| P056 | B5: small VLM | Qwen3-VL-2B 4-bit zero-shot, sanity check | Table | — | D |
| P057 | B6: segmentation baselines | Legacy U-Net, DINOv3 linear seg probe, SAM 3 (separate Py3.12/torch≥2.7 env) | Per-dataset IoU | all-cloud collapse metric (pred-cloud fraction on clear-sky frames) | CP (U-Net, probe), D (SAM 3) |
| P058 | B7: CBH baselines | Climatology, genus→étage lookup, ceilometer-only CNN regressor, zero-shot DA3/MoGe-2 (show sky masked) | MAE/bias/skill table by height bin | — | CP |
| P059 | Baseline leaderboard | Random vs grouped vs LODO protocol-gap table; all heads | Auto-generated from run registry | — | CP |
| P060 | **Gate G3 — Baselines reproduced** | Variance across seeds acceptable; numbers plausible vs literature | Signed | unstable baselines invalidate comparisons | CP |

### Stage G — STRATIA model core (P061–P072)

| ID | Phase | Work | Exit criteria | Guard | |
|---|---|---|---|---|---|
| P061 | Architecture spec | Final diagram, tensor shapes, contract mapping; ADR-005 heads, ADR-006 losses | Reviewed | — | CP |
| P062 | SkyRay encoder | Ray-map MLP + null token; meta FiLM | Unit tests; with/without metadata both run | — | CP |
| P063 | Global heads | Attention-pool; genus (sigmoid, hierarchical), étage, oktas (CORAL ordinal) | Shape/contract tests | — | CP |
| P064 | Taxonomy & soft-label losses | Set-valued loss, KL to annotator distributions, hierarchy consistency | Toy-case tests (gradients correct) | — | CP |
| P065 | Sky-parsing head | Light DPT-style decoder, 4 classes, ignore=255 | Trains on one dataset; IoU > probe | — | CP |
| P066 | Layer head | Same decoder trunk, 3 classes on cloud pixels | Trains on Almería | tiny test sets (36+12) — report CIs | CP |
| P067 | CBH head | Zenith-weighted token pooling + ordinal bins + no-cloud; ordinal CE + CRPS; layer-consistency term | Trains on paired data; beats climatology on val | predicting the prior (mode collapse to common bin) | CP |
| P068 | OOD / quality score | Energy score + Mahalanobis on pooled features; glare/night flags | AUROC on held-out-domain vs in-domain | — | D |
| P069 | Multi-dataset sampler | Per-task batches, temperature-balanced dataset sampling, missing-label masks | Sampling stats match config | big datasets drowning small ones | CP |
| P070 | Multi-task balancing | Uncertainty weighting (Kendall) + gradient-conflict monitor (task-grad cosine) | Logged per step | negative transfer | CP |
| P071 | Model sanity suite | Overfit-one-batch (→ ~0 loss), shuffled labels (→ chance), constant input, contract/shape tests, determinism | All pass in CI (tiny) | catches pipeline bugs before real training | CP |
| P072 | Training engine | AMP, grad accumulation, clipping, EMA, checkpoint/resume, NaN guard, early stopping on val | Resume reproduces metrics | — | CP |

### Stage H — Training regime & fit diagnostics (P073–P082)

| ID | Phase | Work | Exit criteria | Guard | |
|---|---|---|---|---|---|
| P073 | Stage-1: global heads on cached features | Frozen backbone | Fit report: train/val curves per dataset | under/overfit diagnosis (see §4) | CP |
| P074 | Stage-2: dense heads (frozen backbone) | Sky parsing + layer | Per-class IoU; **predicted-class-fraction monitor on clear-sky & B0268 frames** | the legacy "85–99% cloud" collapse | CP |
| P075 | Stage-3: CBH head | Eye2Sky pairs; synthetic consumer views (P049) | Skill vs climatology on val, by height bin | mode collapse; overcast-day dominance | CP |
| P076 | Stage-4: joint multi-task (frozen) | All heads together | Joint ≥ single-task on ≥ most heads, else documented | negative transfer | CP |
| P077 | Stage-5: partial unfreeze | LoRA (r=8–16) on last N blocks, layer-wise LR decay, grad checkpointing | Gain over frozen on val without forgetting on other datasets | overfitting small datasets; forgetting | CP |
| P078 | Budgeted HPO | Optuna on **val only**, ≤ 30 trials, mostly on cached features | Best config + sensitivity plot | val overfitting through HPO — keep budget small, report it | D |
| P079 | Learning curves | Performance vs 10/25/50/100% data per dataset/task | Curves + regime diagnosis (data-limited vs capacity-limited) | — | CP |
| P080 | Regularization study | Weight decay, drop-path, aug strength, label smoothing, EMA | Table; choice justified | — | D |
| P081 | Fit sign-off | Train/val gaps, calibration, curves for final candidate | Report signed | — | CP |
| P082 | **Gate G4 — Model vs baselines (CVPR go/no-go, target 3 Nov)** | Compare to pre-registered criteria (P009) on **val** | GO ⇒ P089-P093 then paper; NO-GO ⇒ loop P073–P080 (max 2) and retarget ICCV 2027 | — | CP |

### Stage I — Domain adaptation & B0268 (P083–P088)

| ID | Phase | Work | Exit criteria | Guard | |
|---|---|---|---|---|---|
| P083 | B0268 zero-shot eval | On labelled B0268 val days, before any adaptation | Honest OOD number | — | CP |
| P084 | Few-shot adaptation | LP-FT with source replay (legacy lesson: tune replay ratio; 4× worked, 10× over-biased) on day-blocked splits | Gain on held-out B0268 days without source regression >2 pts | over-fitting a handful of days | D |
| P085 | Self-training on unlabelled captures | EMA teacher, class-balanced confidence thresholds | Gain on held-out days; pseudo-label precision audited on labelled subset | confirmation bias | D |
| P086 | Test-time adaptation | LayerNorm-affine entropy minimisation (TENT-style) | Accept only if helps on held-out days | collapse to one class | D |
| P087 | Non-zenith CBH & METAR check | Synthetic tilted views; B0268 vs METAR cloud bases (coarse, ordinal) | Report with caveats | METAR ≠ ground truth at your site | D |
| P088 | Robustness suite | Glare, haze, dirty lens, JPEG, exposure shift corruptions | Corruption-robustness table | — | D |

### Stage J — Evaluation & analysis (P089–P094)

| ID | Phase | Work | Exit criteria | Guard | |
|---|---|---|---|---|---|
| P089 | Calibration | Per-head temperature scaling on val; conformal intervals for CBH | ECE, interval coverage within ±5 pts of nominal on val | over-confident CBH | CP |
| P090 | **Locked-test evaluation (once)** | All test sets, LODO, B0268, CBH sites; bootstrap 95% CIs; results JSON hashed & committed | Done once; reason logged | no re-runs after looking | CP |
| P091 | Ablations | SkyRay on/off, metadata dropout, set-valued vs hard labels, soft labels, backbone size, multi- vs single-task, consumer-view synthesis, augmentation | Ablation table (on val; final rows on test once) | — | CP |
| P092 | Human-ceiling comparison | STRATIA as an extra Montenegro annotator; α with/without model | Figure | — | CP |
| P093 | Error analysis & shortcut re-check | Confusions (St/Sc, Ci/Cs, As/Ns), glare cases, failure gallery, attention maps; re-run dataset-ID probe on final features | Gallery + notes | — | CP |
| P094 | Efficiency | Params, FLOPs, latency on RTX 4050, laptop CPU, Pi 5 (via CloudScope bench) | Table | — | D |

### Stage K — Export & release (P095–P097)

| ID | Phase | Work | Exit criteria | Guard | |
|---|---|---|---|---|---|
| P095 | Edge student | Distil to DINOv3-ConvNeXt-T or MobileNetV4 for Pi 5 | ≤3 pt drop vs teacher on val | student overfitting teacher errors | D |
| P096 | ONNX export & quantization | Opset ≥17, dynamic batch; PyTorch↔ORT parity ≤1e-4; INT8 for Pi; `model_card.json` | Parity + accuracy-drop report; contract test passes in CloudScope | — | CP (FP32 ONNX), D (INT8) |
| P097 | Release | Reproduction scripts (`scripts/reproduce_table*.py`), data cards, licences, weights on GitHub Release/HF | Fresh-clone reproduction of one table works | — | D |

### Stage L — Paper (P098–P100)

| ID | Phase | Work | Exit criteria | Guard | |
|---|---|---|---|---|---|
| P098 | Paper draft | CVPR template; all tables/figures generated from results JSON (no hand-typed numbers) | Full draft 6 Nov | — | CP |
| P099 | Internal review | Mentor read; **verify every citation exists** (CVPR rejects fabricated citations); anonymity; supplementary (code, splits, data card) | Checklist complete | — | CP |
| P100 | Submission | OpenReview registration (10 Nov), PDF (16 Nov AoE), supplementary (23 Nov); arXiv decision; rebuttal kit | Submitted | — | CP |

> **Paper track runs in parallel**, not only at the end: outline in P009, related work during Stage B–C, method section at P061, experiments section filled as results land.

---

## 4. Over/underfitting & training-failure playbook

Every training phase (P073–P088) must fill a **Fit report** in its phase doc: train vs val loss curves, per-dataset/per-class metrics, train–val gap, calibration (ECE), predicted-class distribution, and a verdict from this table.

| Problem | Symptoms | Diagnostics | Remedies (in order) |
|---|---|---|---|
| **Overfitting** | Train ↑, val ↓/flat after early epochs; gap > ~15 pts | Learning curves (P079); per-dataset gap; seed variance | Freeze more / smaller LoRA rank; more aug (P048); weight decay, drop-path, EMA; early stopping on val; more data (P049 synthetic views, B0268 captures) |
| **Underfitting** | Train and val both low; loss plateau high | Overfit-one-batch test (P071); capacity probe (ViT-B vs S); LR finder | Unfreeze more blocks; higher LR for heads; longer schedule; weaker aug; check labels/loader first |
| **Leakage (fake good results)** | Val ≫ literature-honest numbers; random split ≫ grouped split | P024–P027 audits; group/time overlap tests (P038) | Fix splits; never tune on test; report both protocols |
| **Shortcut learning** | Dataset/camera ID predictable; model fails LODO | P029 probe; saliency on borders/text | Mask corners/text; resolution normalization; domain-balanced sampling; geometry dropout |
| **Class imbalance / minority collapse** | Minority recall ~0; macro-F1 ≪ accuracy | Per-class confusion; prediction histogram | Balanced sampling, logit adjustment, focal loss (ablate), set-valued labels |
| **Label noise / expert disagreement** | Plateau near human ceiling; noisy confusions | P035 ceiling; P040 flags | Soft labels; set-valued loss; don't chase accuracy above the ceiling |
| **Segmentation collapse** (legacy bug) | Predicts "cloud" everywhere on new cameras | Pred-class-fraction monitor on clear-sky + B0268 (P074) | Sun-glare & obstruction classes; consumer-view synthesis; B0268 dense labels; photometric aug |
| **CBH mode collapse** | Predicts the most common height bin | Prediction histogram vs label histogram; skill vs climatology | Ordinal loss + CRPS; bin re-weighting; zenith ROI pooling; more multi-layer/high-cloud samples |
| **Negative transfer (multi-task)** | Joint < single-task on some head | Task-gradient cosine (P070); single vs joint ablation | Loss re-weighting; task-specific adapters; drop the hurting task from joint training |
| **Catastrophic forgetting** | Source datasets regress after adaptation | Track all-dataset val during P077/P084 | Source replay (tuned ratio), lower LR, LoRA only |
| **Pseudo-label confirmation bias** | Self-training improves train confidence, not held-out accuracy | Pseudo-label precision on labelled subset | Higher thresholds, class-balanced selection, EMA teacher, stop early |
| **Miscalibration** | High confidence, wrong answers; poor interval coverage | ECE, reliability diagrams, PIT for CBH | Temperature scaling, conformal intervals, ensembles (rented GPU) |
| **Validation overfitting via HPO** | Val keeps improving, test (later) disappoints | Track #trials; holdout-of-val | Small HPO budget; report it; one locked test eval |
| **Pipeline bugs** (legacy: `ImageFolder` mapped Cu→Ac; NumPy 2 broke torch) | Impossible metrics (0/5 or 100%) | Contract tests, `class_to_idx` assertions, env pins | Assertions in every loader; CI on tiny data |
| **Numerical instability** | NaN/inf losses with AMP | NaN guard; grad-norm logs | bf16/fp16 switch, clipping, lower LR |
| **Non-reproducibility** | Same config, different results | Seed tests (P004), cuDNN flags | Fix seeds; report mean ± std over 3 seeds |

---

## 5. CVPR 2027 fast-track (today → 16 Nov 2026)

| Target date | Milestone | Phases |
|---|---|---|
| Sun 11 Oct | Foundation + inventory done | P001–P014, P016, P018–P019, P021–P022 |
| Wed 14 Oct | **G1** data audit | P023–P027, P029–P032 |
| Sun 18 Oct | **G2** splits/labels locked; geometry ready | P033–P039, P041 (image-level), P042–P047 |
| Fri 23 Oct | **G3** baselines | P048–P055, P057–P060 |
| Sun 1 Nov | Model trained (frozen + LoRA) | P061–P067, P069–P077, P079, P081 |
| **Tue 3 Nov** | **G4 go/no-go** | P082–P083 |
| Thu 5 Nov | Calibration, locked test, ablations | P089–P093, P096 (FP32) |
| Fri 6 Nov | Full draft | P098 |
| **Tue 10 Nov** | OpenReview registration (author list frozen) | P099 |
| **Mon 16 Nov AoE** | Submit | P100 |

Everything marked **D** moves after 16 Nov (and feeds an ICCV 2027 / NeurIPS 2027 version if needed). If **G4 = NO-GO**, we do not submit a weak paper: the plan continues at P073 with the ICCV 2027 deadline (~early March 2027, estimate) as the new target.

Honest note: this schedule is tight but feasible because most work is frozen-feature training on your GPU. The riskiest items are (1) whether the new Eye2Sky download contains images at the ceilometer sites (P012) and (2) CBH skill over climatology (P075).

---

## 6. Compute & storage plan (RTX 4050, 6 GB)

| Workload | Where | Notes |
|---|---|---|
| Feature extraction ViT-S/B/L @224–512 | Laptop GPU | bf16 inference; ViT-L weights ≈0.6 GB |
| Heads on cached features | Laptop GPU/CPU | minutes |
| ViT-S LoRA fine-tune @512 | Laptop GPU | AMP + grad checkpointing + accumulation; batch tuned in P010 |
| CNN baselines @224 | Laptop GPU | 3–5 GB |
| Multi-seed sweeps, ViT-B/L fine-tune ablations | Rented RTX 4090 (optional) | upload cached features, not raw images |
| SAM 3 | Separate env (Python ≥3.12, torch ≥2.7, CUDA 12.6 wheels — your driver supports 12.7) | deferred |

Storage: Eye2Sky full-res images ≈1 GB per station-day. **Raw archive on the 1 TB external HDD** (read once); **training caches on the internal SSD** — 512-px image cache (≈10× smaller) and fp16 feature memmaps (ViT-S @512 ≈0.8 MB/frame), packed into few large files because hard disks are slow at random small reads. Both locations are set in `configs/paths.yaml` (not committed).

---

## 7. Requirements checklist (STRATIA-specific)

- [ ] Eye2Sky: images at OLDLR / OLUOL / OLWIN (for CDLRA) and/or WESTE (for CDLRB) on ceilometer days (Apr–Jul 2022). Inventory in P012 tells us if more downloading is needed.
- [ ] DeepSky images (Zenodo 8208505); WEBCAM images; HBMCD — download links verified in P015.
- [ ] Hugging Face DINOv3 access (gated) — you accept the licence and log in.
- [ ] ≥300 B0268 frames labelled by 2 people (P041) — needs CloudScope P003 logger running ASAP.
- [ ] Your site's lat/lon (airports: VOBL Kempegowda, VOBG HAL — METAR, P020).
- [ ] Storage: raw archive (Eye2Sky originals, B0268 captures) on the 1 TB external HDD; training caches (512-px images, features) on the internal SSD (≈30–80 GB). Paths in `configs/paths.yaml`.
- [ ] OpenReview profile with college email (CVPR) — create this week.
- [ ] Optional: rented GPU budget; mentor.

---

## 8. Repository layout (target)

```
STRATIA/
├── stratia/            # package: data/, geometry/, models/, losses/, train/, eval/, export/
├── configs/            # hydra configs: datasets.yaml, ontology.yaml, splits/, model/, train/
├── scripts/            # inventory, audit, cache_features, train, eval, export, reproduce_*
├── tests/              # unit + contract + sanity tests
├── docs/
│   ├── phases/         # P001-…-P100 phase docs
│   ├── adr/            # architecture decision records
│   ├── data/           # data cards, audit reports
│   ├── contract.md     # STRATIA ↔ CloudScope interface
│   └── compute.md
├── paper/              # LaTeX (CVPR template), generated tables/figures
├── PROJECT_STATE.md  CHANGELOG.md  EXPERIMENT_LOG.md  README.md  LICENSE
└── (ignored) data/ runs/ caches/ weights/
```

## 9. After v1 (not in the 100 phases)

Multi-camera teacher (Eye2Sky network stereo → dense per-pixel CBH pseudo-labels for the single-camera student), temporal two-frame input with ERA5 wind scaling, nowcasting. These are the natural "STRATIA v2" / follow-up paper.
