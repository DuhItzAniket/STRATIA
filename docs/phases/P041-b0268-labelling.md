# P041 — B0268 labelling protocol

Status: BLOCKED     Date: 2026-10-10     Commit: (this commit)

## Objective
Label at least 300 frames from the owner's Arducam B0268 with two annotators: image-level genus set, étage and
oktas for all of them, and dense sky / cloud / glare / obstruction masks for 100; report the agreement.

## Inputs / dependencies
CloudScope sky-logger captures with JSON sidecars (P019, second exit criterion, still open); two labellers; the WMO
ontology (P033) and the dataset mapping rules (P034); the few-shot B0268 protocol of `docs/splits.md` (day blocks).

## Work log
1. Not started: no logger run exists yet (owner item, CloudScope P003/P032). The only B0268 data on disk are the
   25 legacy frames from one five-minute window (P019), which cannot form a labelled set.
2. Owner decision 2026-10-10: Stage E proceeds without this phase; Gate G2 (P042) is signed with it deferred. The
   guide, the local labelling page and the agreement script are written in this phase once frames exist.

## Verification
None yet.

## Exit criteria
- [ ] >= 300 frames labelled by 2 people.
- [ ] Agreement reported (Krippendorff's alpha per target with `stratia/labels/agreement.py`, as for Montenegro in P035).

## Fit & data-risk notes
- Single-annotator bias is the guard the plan names; two labellers are required, not optional.
- The few-shot protocol splits B0268 by calendar day (P037). Captures must cover at least six distinct days,
  preferably ten or more spread over weeks and times of day, or the day-blocked split has nothing to hold out.
- Until this phase closes, hypothesis H4 (geometry inputs help transfer to a consumer camera) and the B0268
  out-of-distribution numbers (P083) cannot be produced; Paper A's criteria C1 and C2 do not depend on them.

## Deviations from plan & why
Deferred by the owner on 2026-10-10 because the frames do not exist; the latest date to close it is before P083.

## Next phase
P042 — Gate G2 (conditional sign-off), then Stage E.
