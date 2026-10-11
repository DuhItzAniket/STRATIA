# P062 — SkyRay encoder

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Wire the P050 null-token encoders to the backbone: SkyRay tokens added to the patch tokens with a learned
strength, Sun-metadata FiLM on the CLS token and on the tokens, and the per-patch zenith cosine the CBH head's
attention prior reads; one traced graph serves the known and the unknown geometry states.

## Inputs / dependencies
P050 `SkyRayEncoder` / `MetaFiLM`; P045 ray maps (`ray_map`, `meta_vector`); ADR-005 §1–3; P061 shape table.

## Work log
1. `stratia/model/skyray.py`: `GeometryEncoder(d_model, hidden, alpha=0.1, film_tokens=True)` →
   `(tokens', cls')` with `tokens' = FiLM(tokens + α · SkyRay(ray_map))` and `cls' = FiLM(cls)`. `α` is a learned
   scalar initialised at 0.1 so a fresh model is close to the geometry-free backbone and the fusion strength can be
   read off a checkpoint. `MetaFiLM.gamma_beta` exposes the modulation so CLS and tokens share one computation.
   `zenith_cos(ray_map)` returns the z channel (= cos of the zenith angle in the contract's frame) and the valid
   flag per patch for the CBH prior (P067).
2. Everything stays a function of the inputs (`torch.where` on the valid flags, no Python branch on data), which
   the trace test below holds it to.

## Verification
- `tests/test_geometry_encoder.py` (4): shapes; the unknown-geometry sample gets exactly the offset α · null token
  on every patch and an untouched CLS (FiLM null = identity at initialisation); the known sample's invalid patches
  get the same offset and its valid patches do not; no cross-talk between samples of a mixed batch (joint run
  equals per-sample runs); gradients reach `alpha`, the null token and the null FiLM parameters; `zenith_cos` on
  the OLDLR fisheye peaks above 0.99 at the zenith patch and is never below the horizon on valid patches; a graph
  traced with geometry known reproduces eager outputs for the unknown state too.
- P050's four tests still pass (the stub and the standalone encoders are unchanged).

## Exit criteria
- [x] Unit tests; with and without metadata both run (and both run through one traced graph).

## Fit & data-risk notes
- α starts small on purpose: if the geometry is useless the model can keep it near 0; if α grows large during
  Stage H, the ablation of P091 (ray map on/off) has an easy first read.
- FiLM on tokens as well as CLS gives the dense heads access to the Sun's elevation (glare class, P074); the flag
  `film_tokens` turns it off for the ablation.

## Deviations from plan & why
- None.

## Next phase
P063 — Global heads.
