# P002 — Phase protocol & logs

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Put the plan and the working rules into the repository: phase template, honesty rules, state tracking, changelog and experiment log.

## Inputs / dependencies
P001; plan `STRATIA/plans/STRATIA_Implementation_Plan.md` (2026-10-04).

## Work log
1. Copied the plan to `docs/PLAN.md` with corrections agreed with the owner on 2026-10-04:
   - exact DINOv3 model IDs (ViT-S/16, ViT-B/16, ViT-L/16); the 7B model (26.9 GB weights) is excluded because it cannot load on the 6 GB GPU;
   - METAR airports VOBL (Kempegowda) and VOBG (HAL) for Bengaluru;
   - storage: raw archive on the 1 TB external HDD, training caches on the internal SSD (no 2 TB SSD purchase needed).
2. Added `docs/phases/TEMPLATE.md` (with the mandatory fit-report section), `CONTRIBUTING.md` (phase workflow; research honesty rules: no fabricated numbers or labels, locked test sets, kept negative results, verified citations, generated splits), `PROJECT_STATE.md` (with gate dates), `CHANGELOG.md`, `EXPERIMENT_LOG.md`.

## Verification
- Gated model access checked with the owner's read token: config files of all four DINOv3 checkpoints download; weight sizes ViT-S 86 MB, ViT-B 343 MB, ViT-L 1,213 MB, 7B 26,864 MB.
- P001's phase document already follows the template sections.

## Exit criteria
- [x] Template used by P001–P002.

## Fit & data-risk notes
The honesty rules (locked test sets, generated splits, run registry) are the project's first defence against leakage and over-optimistic results.

## Deviations from plan & why
Plan corrections listed above, all owner-agreed.

## Next phase
P003 — Project environment.
