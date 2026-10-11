# P070 — Multi-task balancing

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Combine the heads' losses without hand-tuned weights and watch for negative transfer: Kendall uncertainty
weighting with one learned log-variance per task, and a task-gradient cosine monitor on the shared parameters
logged per step.

## Inputs / dependencies
P064 masked losses (exact 0 for an absent task); P069 per-task batches; ADR-006 §8; PLAN §4 "negative transfer".

## Work log
1. `stratia/model/balance.py`: `UncertaintyWeighting(tasks)` → L = Σ_t exp(−s_t)·L_t + s_t over the tasks present
   in the batch (presence = explicit flags or "loss requires grad and is not exactly 0"), with s_t floored at −6 so
   a task cannot be weighted up without bound; `weights()` reports exp(−s_t). `GradConflictMonitor(shared_params)`
   computes each task's gradient on the shared parameters (`autograd.grad`, graph retained) and the pairwise
   cosines plus per-task gradient norms; `conflicts(cosines, threshold=−0.1)` names the opposed pairs. The monitor
   is a diagnostic: v1 logs it and does not alter the update (the plan's "gradient-conflict monitor").
2. Policy: all seven loss terms are tasks (genus, étage, oktas, sky_parse, layer, cbh, consistency terms folded into
   their head's task); the weighting and the cosines are logged per step by the engine (P072).

## Verification
- `tests/test_balance.py` (2): an absent task contributes nothing (neither its loss nor its s_t) under the
  heuristic and under explicit flags; training the log-variances on a 10.0 vs 0.1 loss pair down-weights the large
  one (log σ² of the large task ends above the small one's); on a shared linear layer the monitor reports cosine
  > 0.99 for a scaled copy of a loss, < −0.5 for its negative, lists exactly the opposed pairs, and leaves the graph
  usable for the real backward.

## Exit criteria
- [x] Logged per step (the engine logs `weights/<task>` and `grad_cos/<a>|<b>`).

## Fit & data-risk notes
- Uncertainty weighting can silence a hard task by inflating its variance; the floor and the per-step log make that
  visible, and P076 compares joint against single-task training head by head.
- The cosine monitor costs one extra backward per task per logged step; the engine computes it every `log_every`
  steps, not every step.

## Deviations from plan & why
- None.

## Next phase
P071 — Model sanity suite.
