# P071 — Model sanity suite

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Assemble the full model and the checks every training configuration must pass before a real run: contract shapes,
overfit one batch, shuffled labels at chance, constant input, determinism; all on a tiny fake backbone so they run
in CI in seconds, and on the real model before Stage H.

## Inputs / dependencies
P062–P068 modules; `stratia/contract.py`; PLAN §4 (pipeline-bug row: the legacy `ImageFolder` Cu→Ac mapping and the
NumPy 2 break are the kind of thing this suite catches).

## Work log
1. `stratia/model/stratia.py`: `STRATIA(backbone, d_model, dense, cbh, ood)` with `forward(image, ray_map, meta)`
   (backbone in the graph) and `forward_tokens(tokens, cls, ray_map, meta)` (cached features); `outputs()` filters
   to the contract names in the contract order; `head_parameters()` groups parameters for per-head optimiser
   settings. `FakeBackbone` (per-patch mean colour projected to d) is the CI stand-in.
2. `stratia/model/sanity.py`: `toy_batch` (images whose colour encodes the genus, half the batch with geometry,
   every target type with partial masks), `total_loss` (all P064 losses on the outputs), `overfit_one_batch`,
   `shuffled_labels_check` (train on half with permuted genus labels, score the other half with its own permuted
   labels), `constant_input_check` (max spread over identical inputs), `determinism_check` (parameters after a few
   steps under `use_deterministic_algorithms`).

## Verification
- `tests/test_model_sanity.py` (5): every contract output has the `output_shapes` shape and the probability heads
  sum to 1; the null-geometry state runs and changes the CBH view; overfit-one-batch reaches < 5 % of the first
  loss in 150 steps; shuffled labels stay within 0.25 of chance on the held-out half after 30 steps; constant
  inputs give a spread < 1e-5 (no batch coupling); two runs with the same seed give bit-identical parameters and a
  different seed does not; the six losses run on the model outputs with partial masks and give finite gradients.
- `tests/test_ood.py` (2) and the P062–P070 suites make 27 Stage G tests; the whole suite runs in CI on CPU.

## Exit criteria
- [x] All pass in CI (tiny).

## Fit & data-risk notes
- The shuffled-label check is a leakage detector for the *pipeline*, not for the data; the P024–P027 audits cover
  the data. A real-model run of the suite before each Stage H training is the protocol (P073 onwards).
- `determinism_check` uses warn-only deterministic mode: on CUDA some kernels are non-deterministic and the Stage H
  reports quote mean ± std over seeds (P004) rather than bit equality.

## Deviations from plan & why
- None.

## Next phase
P072 — Training engine.
