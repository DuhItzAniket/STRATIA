# Project State

| Field | Value |
|---|---|
| Current stage | E — Geometry & metadata (P043–P050); Stages A–D complete (partial P015, P019, P020, P028; P041 blocked) |
| Last completed phase | P046 — Geometry-consistent transforms |
| Next phase | P047 — Image–ceilometer pairing (once the OLDLR/WESTE download is complete; rerun scripts/sun_validation.py first for the 22 Apr OLDLR and 27 Jun WESTE calibrations) or P048 — Augmentation policy; P041 (B0268 labelling) runs whenever the logger frames exist, latest before P083 |
| Branch | `main` |
| Target venue | CVPR 2027 (registration 10 Nov, paper 16 Nov 2026 AoE); go/no-go at Gate G4 (target 3 Nov) |
| Blockers | None for Paper A (G1 signed). **Paper B:** 0 pairable Eye2Sky images on disk; OLDLR and WESTE images Apr–Jul 2022 needed before P047: download running since 2026-10-10 (`tools/download_eye2sky.py`). B0268 logging for P041 to start now (P041 BLOCKED). Missing public datasets need owner action (P015, cut-off 15 Oct). |
| Compute | RTX 4050 Laptop 6 GB, 16 GB RAM; DINOv3 S/B/L access verified 2026-10-04 |

## Gates

| Gate | Phase | Target date | Status |
|---|---|---|---|
| G1 Data audit | P032 | 14 Oct | **Signed 2026-10-08** (`docs/data/data_card.md`) |
| G2 Splits & labels | P042 | 18 Oct | **Signed 2026-10-10, conditional** (P041 deferred, latest before P083) |
| G3 Baselines | P060 | 23 Oct | — |
| G4 Model vs baselines (CVPR go/no-go) | P082 | 3 Nov | — |

## Phase log

| Phase | Title | Status | Date |
|---|---|---|---|
| P001 | Repo bootstrap | DONE | 2026-10-04 |
| P002 | Phase protocol & logs | DONE | 2026-10-04 |
| P003 | Project environment | DONE | 2026-10-04 |
| P004 | Config & run system | DONE | 2026-10-04 |
| P005 | Experiment tracking | DONE | 2026-10-04 |
| P006 | Tests & CI | DONE | 2026-10-04 |
| P007 | Data registry spec | DONE | 2026-10-04 |
| P008 | Interface contract v1 | DONE | 2026-10-04 |
| P009 | Research design & pre-registration | DONE | 2026-10-04 |
| P010 | Hardware budget profiling | DONE | 2026-10-04 |
| P011 | Inventory local data | DONE | 2026-10-04 |
| P012 | Eye2Sky inventory | DONE | 2026-10-04 |
| P013 | Eye2Sky readers | DONE | 2026-10-04 |
| P014 | Ceilometer reader | DONE | 2026-10-04 |
| P015 | Acquire missing public data | PARTIAL | 2026-10-04 |
| P016 | Montenegro parser | DONE | 2026-10-04 |
| P017 | MGCD parser + weather | DONE | 2026-10-04 |
| P018 | Segmentation datasets loader | DONE | 2026-10-04 |
| P019 | B0268 ingest | PARTIAL | 2026-10-04 |
| P020 | External weak references (METAR) | PARTIAL | 2026-10-04 |
| P021 | Unified sample schema | DONE | 2026-10-04 |
| P022 | Loader performance | DONE | 2026-10-04 |
| P023 | Integrity check | DONE | 2026-10-06 |
| P024 | Exact duplicates | DONE | 2026-10-06 |
| P025 | Near-duplicates | DONE | 2026-10-06 |
| P026 | Label-conflict audit | DONE | 2026-10-06 |
| P027 | Temporal autocorrelation | DONE | 2026-10-06 |
| P028 | MGCD ≟ GRSCD | PARTIAL | 2026-10-06 |
| P029 | Shortcut audit | DONE | 2026-10-06 |
| P030 | Imbalance report | DONE | 2026-10-08 |
| P031 | Ceilometer QC & pairing tolerance | DONE | 2026-10-08 |
| P032 | Gate G1 — Data audit sign-off | DONE | 2026-10-08 |
| P033 | WMO ontology | DONE | 2026-10-08 |
| P034 | Dataset → ontology mapping | DONE | 2026-10-08 |
| P035 | Montenegro soft labels & human ceiling | DONE | 2026-10-08 |
| P036 | Ceilometer → targets | DONE | 2026-10-08 |
| P037 | Split design | DONE | 2026-10-08 |
| P038 | Split generator | DONE | 2026-10-08 |
| P039 | Lock test sets | DONE | 2026-10-08 |
| P040 | Label-noise estimation | DONE | 2026-10-08 |
| P041 | B0268 labelling protocol | BLOCKED | 2026-10-10 |
| P042 | Gate G2 — Splits & labels sign-off | DONE (conditional) | 2026-10-10 |
| P043 | Camera models | DONE | 2026-10-10 |
| P044 | Sun position & calibration validation | DONE | 2026-10-10 |
| P045 | Ray-map generator | DONE | 2026-10-10 |
| P046 | Geometry-consistent transforms | DONE | 2026-10-10 |
