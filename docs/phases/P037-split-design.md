# P037 — Split design

Status: DONE     Date: 2026-10-08     Commit: (this commit)

## Objective
Write down what a split may cut, which protocols the paper reports and the guarantees the generator must prove,
so that P038 implements a reviewed design rather than improvising one.

## Inputs / dependencies
P024/P025 groups, P026 conflict flags, P027 temporal blocks, P029 (why out-of-camera protocols are the ones that
count), P030 stratification columns, P034 labels, `docs/research_design.md` (H1, criteria C1–C3).

## Work log
1. `docs/splits.md`: the unit (connected component of "same group" and "same block"), five protocols (in-domain
   blocked, LODO, held-out station, few-shot target, published for comparison), seven guarantees, what is
   deliberately not done, expected unit counts.
2. `configs/splits.yaml`: fractions, sources (SWIM family = one source), block rules per dataset, stratification
   columns, test exclusions, LODO folds, held-out-station variants, few-shot protocol, published splits.

## Verification
- Reviewed against the Stage C findings: every rule cites the report it comes from (duplicates, near-duplicates,
  conflicts, temporal blocks, shortcuts, imbalance). The config is parsed by P038's generator and its tests.

## Exit criteria
- [x] `docs/splits.md` reviewed (this phase) and implemented by P038.

## Fit & data-risk notes
- Units are coarse for the time series (Montenegro 10 weekly blocks, Eye2Sky 9 days): in-domain numbers on them
  will have wide intervals; that is the honest resolution and the paper says so.
- LODO is the main protocol for H1; in-domain blocked splits exist for model selection and for the comparison
  with published numbers, never the other way round.

## Deviations from plan & why
- None; "grouped / temporal / site / camera splits" of the plan map to the unit definition, the held-out-station
  protocol and LODO.

## Next phase
P038: Split generator.
