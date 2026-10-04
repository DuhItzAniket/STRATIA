# P020 — External weak references (METAR)

Status: PARTIAL     Date: 2026-10-04     Commit: (this commit)

## Objective
Prepare coarse external references for the B0268 site: METAR cloud reports from Bengaluru airports VOBL (Kempegowda) and VOBG (HAL) via the Iowa Environmental Mesonet archive; ERA5 notes.

## Inputs / dependencies
P019 (B0268 captures with UTC times); no B0268 logger campaign exists yet.

## Work log
1. Marked **deferrable (D)** in the CVPR fast-track (`docs/PLAN.md` §5): METAR reports give layer amount and base height at an airport 10–35 km from the camera, useful only as a qualitative, ordinal check for the B0268, which is not part of any pre-registered criterion.
2. Decision: implement the METAR parser and download together with the first real B0268 campaign (after CloudScope P003 acceptance), when there are captures to compare with. No data downloaded now.

## Verification
None (deferred).

## Exit criteria
- [ ] METAR parser + one month downloaded — deferred.

## Fit & data-risk notes
METAR is never used as training labels or as a quantitative test; it cannot affect pre-registered results.

## Deviations from plan & why
Deferred per the plan's D marking; status PARTIAL until implemented.

## Next phase
P021 — Unified sample schema.
