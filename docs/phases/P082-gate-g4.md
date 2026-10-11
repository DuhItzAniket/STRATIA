# P082 — Gate G4: Model vs baselines (CVPR go/no-go)

Status: DONE (gate decided)     Date: 2026-10-11     Commit: (this commit)

## Objective
Compare the Stage H candidates with the pre-registered success criteria of the research design (§5) on validation
data and decide: GO for the CVPR 2027 submission path (P089–P093, then the paper) or NO-GO (loop P073–P080 at most
twice, retarget ICCV 2027).

## Inputs / dependencies
`docs/research_design.md` §5 (C1–C3, decision rule GO = C3 and (C1 or C2)); P059 baseline reference lines; P073 /
P075 / P081 results; `scripts/gate_g4.py` → `docs/data/gate_g4.md`.

## Work log
1. `scripts/gate_g4.py` reads the Stage H result files and writes one row per criterion component with the model's
   number, the reference, the margin and pass / fail, plus a secondary (not gating) row for the val-tuned-threshold
   LODO comparison applied to model and probes alike.

## Verification
| Criterion | Quantity | Model | Reference | Required | Pass |
|---|---|---|---|---|---|
| C1 | LODO étage macro-F1, mean of the CCSN / MGCD / Montenegro / Almería folds | 0.548 | 0.583 (best P054 probe) | ≥ reference + 0.05 | **no** (−0.035) |
| C1 (secondary) | same, per-column thresholds chosen on val, model and probes alike | 0.650 | 0.668 | not gating | no (−0.017) |
| C1 | in-domain genus macro-F1 (CCSN, MGCD; like for like) | 0.719 | 0.716 | ≥ reference − 0.01 | yes |
| C2 | Montenegro étage-set agreement / human ceiling | 1.047 | 0.90 | ≥ 0.90 | **yes** |
| C2 | Montenegro cover (±1 okta) agreement / human ceiling | 1.027 | 0.90 | ≥ 0.90 | **yes** |
| C3 | CBH skill over climatology, CDLRA (OLDLR) validation weeks | +0.471 | 0.20 | ≥ 0.20 | **yes** |
| C3 | CBH skill, held-out CDLRB (WESTE) site, OLDLR-only training | +0.448 | 0.10 | ≥ 0.10 | **yes** |
| C3 | CBH MAE vs the ceilometer-supervised CNN | 566 m | 724 m | ≤ 652 m (10 % below) | **yes** |
| C3 | 80 % interval coverage (after the P081 cross-fitted temperature; raw 0.917) | 0.802 | — | 0.75–0.85 | **yes** |

**Decision: GO.** C3 holds on every component and C2 holds; C1 fails on its LODO component under both the
pre-registered rule and the secondary one.

What the paper will say about C1: the cross-dataset étage transfer of the model does not beat a linear probe on the
same frozen features; the P073 diagnostic attributes it to label-definition shift of "mid" (genus-derived in CCSN,
per-pixel layers in Almería, rater votes in Montenegro, absent from MGCD's étage labels) and to calibration, not
ranking (AUROC 0.78–0.88 on the transferable columns). The claim H1 is therefore weakened to "matches" rather
than "exceeds" published-style cross-dataset recognition on étage, and the paper's cross-camera evidence rests on
C2 (agreement with human experts on an unseen camera's images at the ceiling) and C3 (CBH skill on a never-seen
site). Supporting criteria: no sky-parsing collapse (0.22–0.30 cloud on clear-sky frames, prototype 85–99 %);
`ood_score` AUROC 0.96–1.00 (P068).

## Exit criteria
- [x] Compared to the pre-registered criteria on validation data; decision recorded; the reasons for the C1 failure
  recorded with the diagnostic that supports them.

## Fit & data-risk notes
- The decision uses validation data that early stopping, the probes' C and the tuned thresholds have all seen;
  the locked test evaluation (P090) is the one that counts, and the CVPR schedule allows exactly one.
- The C3 coverage component passes only after temperature scaling; P089 must deliver that calibration on its own
  split before P090.
- What would turn C1: Stage I's B0268 labels and pseudo-labels (P083–P088) add a camera with exhaustive étage
  sets; the P093 error analysis splits the LODO error by "mid".

## Deviations from plan & why
- The gate was run on 11 Oct, ahead of the 3 Nov target, because the frozen-feature pipeline made Stage H a
  one-day affair; the loop clause is not invoked.

## Next phase
P083 — Stage I, domain adaptation and B0268 (blocked on the P041 frames); the P089 calibration and P090 locked test
follow for the CVPR path.
