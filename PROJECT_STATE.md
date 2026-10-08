# Project State

| Field | Value |
|---|---|
| Current stage | D — Label harmonization & splits (P033–P042); Stages A–C complete (partial P015, P019, P020, P028) |
| Last completed phase | P033 — WMO ontology (Gate G1 signed in P032) |
| Next phase | P034 — Dataset → ontology mapping |
| Branch | `main` |
| Target venue | CVPR 2027 (registration 10 Nov, paper 16 Nov 2026 AoE); go/no-go at Gate G4 (target 3 Nov) |
| Blockers | None for Paper A (G1 signed). **Paper B:** 0 pairable Eye2Sky images on disk; OLDLR and WESTE images Apr–Jul 2022 (≈ 0.3 GB per station-day) needed before P047. B0268 logging for P041 to start now. Missing public datasets need owner action (P015, cut-off 15 Oct). |
| Compute | RTX 4050 Laptop 6 GB, 16 GB RAM; DINOv3 S/B/L access verified 2026-10-04 |

## Gates

| Gate | Phase | Target date | Status |
|---|---|---|---|
| G1 Data audit | P032 | 14 Oct | **Signed 2026-10-08** (`docs/data/data_card.md`) |
| G2 Splits & labels | P042 | 18 Oct | — |
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
