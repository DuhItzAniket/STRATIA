# P050 — Metadata dropout

Status: DONE     Date: 2026-10-10     Commit: (this commit)

## Objective
Train-time masking of the ray map and the Sun metadata with null tokens, so the model works with and without
geometry and cannot use the presence of geometry as a shortcut; prove that a model path runs in both states.

## Inputs / dependencies
P045 ray maps and meta vector (the contract's zero state for unknown geometry); P029 (dataset identity is readable
from pixels; the ray map must not add a trivial second cue); `docs/contract.md`; PLAN P062 (SkyRay encoder with
null token, meta FiLM), of which this phase implements the null-token path.

## Work log
1. `stratia/augment/dropout.py`: `GeometryDropout(p_ray=0.3, p_meta=0.3, tie=False)` zeroes the ray map and/or the
   meta vector per sample (independent draws unless tied), with a batch version that returns the masks;
   `is_null_ray_map` / `is_null_meta` name the contract's null state.
2. `stratia/model/skyray.py`: `SkyRayEncoder` (MLP on the unit direction for valid patches, a learned null token
   where the valid flag is 0, so an all-zero map becomes a sequence of null tokens instead of the embedding of a
   meaningless zero vector); `MetaFiLM` (feature-wise affine from [cos z, sin z] when valid, learned null
   parameters otherwise, identity at initialisation); `GeometryStub` (encoder + FiLM + mean pool + linear head) as
   the stand-in model until Stage F.
3. Policy: dropout applies to calibrated and uncalibrated samples alike (the latter are zero already), so the
   share of null inputs is not a dataset signature; `p = 0.3` is the plan's value and is a config field for P091.

## Verification
- `tests/test_dropout_skyray.py` (4): measured dropout rates 0.30 ± 0.03 for both inputs with independent draws
  (joint rate 0.09 ± 0.02); tied mode drops both together; batch version masks exactly the drawn rows and leaves
  inputs untouched; an all-zero ray map yields all null tokens and a real map yields null tokens exactly on its
  invalid patches; null FiLM is the identity at initialisation; the stub model trains for three steps on a
  mixed batch (finite loss, the null token receives gradient) and runs with everything known and with nothing
  known.
- Whole suite passes; `ruff check .` clean.

## Exit criteria
- [x] Model runs with and without metadata (the stub path of P062; both extremes and mixed batches).

## Fit & data-risk notes
- The shortcut this phase closes: a model that only ever sees ray maps on Eye2Sky and MGCD-like frames would learn
  "geometry present ⇒ all-sky imager"; dropout on calibrated samples removes the correlation at the input.
- Over-reliance on metadata (time-of-day ⇒ class) is already blocked by the contract (no time, date or location
  inputs); the Sun vector remains, and its dropout rate is the ablation knob (P091: metadata dropout on/off).
- Risk the other way: too much dropout wastes the geometry; 0.3 is the plan's prior, P080/P091 measure it.

## Deviations from plan & why
- The full SkyRay encoder and its FiLM wiring into the backbone are P062; this phase ships the null-token path
  they will reuse, because the exit criterion needs a model path to run.

## Next phase
Stage F — P051 onward (the download for P047 is running; rerun the Sun validation for the 22 April OLDLR and
27 June WESTE calibrations before pairing).
