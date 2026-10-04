# P009 — Research design & pre-registration

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Register the problem, hypotheses, metrics, success criteria and locked-test policy **before** any model training, so results cannot move the goalposts.

## Inputs / dependencies
Deep-research report (`STRATIA/reports/STRATIA sky vision research landscape.md`), plan, contract (P008).

## Work log
1. `docs/research_design.md`: problem, hypotheses H1–H4, tasks and locked tests, metrics, **pre-registered Gate G4 criteria C1–C3**, locked-test policy, threats to validity, amendment log.
2. ADR-002 (DINOv3 ViT-S/16 primary backbone; ViT-B ablation; ViT-L frozen; 7B excluded).
3. ADR-003 (single-camera scope: multi-camera data only as future training supervision).
4. `paper/outline.md` started (paper track runs in parallel with the phases).

## Verification
- Registered on 2026-10-04 at this commit; no STRATIA model has been trained yet (`runs/registry.csv` contains only the P004/P005 bookkeeping runs on synthetic data).
- G4 rule: GO if C3 (cloud-base height skill) and at least one of C1 (cross-dataset recognition) or C2 (expert-level agreement).

## Exit criteria
- [x] Document committed before any model training.

## Fit & data-risk notes
Threats-to-validity table maps each risk (leakage, shortcuts, label noise, time misalignment, validation over-tuning) to its controlling phase.

## Deviations from plan & why
None.

## Next phase
P010 — Hardware budget profiling.
