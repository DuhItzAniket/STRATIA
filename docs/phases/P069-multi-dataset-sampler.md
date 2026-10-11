# P069 — Multi-dataset sampler

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
A batch sampler for the union of STRATIA's datasets: per-task batches (one tensor shape per batch),
temperature-balanced dataset sampling so the big sets do not drown the small ones, and a per-sample label mask so
rows with different label types share a batch.

## Inputs / dependencies
configs/datasets.yaml task lists; P034 labels (which heads a row has targets for); P064 masked losses; P065–P067
batch shapes (cls, 32 × 32 grid, 8 × 8 grid).

## Work log
1. `stratia/data/sampler.py`: `SamplerConfig(tau=0.5, group_probs, batch_sizes, batches_per_epoch, min_share)`;
   `dataset_probs(counts, tau, min_share)` = p_d ∝ n_d^τ with an exact floor (floored sets fixed, the rest
   rescaled, iterated until stable); `expected_shares(table, cfg)` = the share of samples each (group, dataset)
   receives per epoch under the config; `TaskBatchSampler(table, cfg, seed)` yields `(group, indices)` with one
   group per batch, datasets drawn inside the group by their probabilities and rows uniformly with replacement,
   deterministic per (seed, epoch); `sampling_stats` compares observed and expected shares; `label_mask(row)` =
   one flag per head from the row's `has_<head>` columns; `group_mask_from_tasks` maps a dataset's task list to
   the sampler groups it serves (global / dense / cbh).
2. Policy: τ = 0.5 (square-root balance, the plan's "temperature-balanced"); group probabilities 0.5 / 0.3 / 0.2
   and batch sizes 64 / 16 / 64 are the Stage H defaults and config fields.

## Verification
- `tests/test_sampler.py` (3): p_d for τ ∈ {0, 0.5, 1} and the floor; on a 8,000 / 2,000 / 200 synthetic union
  every batch is single-group with the configured size, rows without dense labels never enter a dense batch,
  observed shares over 1,200 batches match the expected ones within 0.02 for all five (group, dataset) cells,
  shares sum to 1, and iteration is deterministic per (seed, epoch) and differs between epochs; label masks and
  task → group mapping.

## Exit criteria
- [x] Sampling stats match config.

## Fit & data-risk notes
- Sampling with replacement inside a dataset means an epoch is a budget, not a pass; small datasets are seen
  several times per epoch (Montenegro ≈ 9 × under τ = 0.5 against MGCD), which is the intended balance and the
  thing to watch for overfitting on the small sets (P079 learning curves).
- The temperature is the P091 ablation knob for "big datasets drowning small ones".

## Deviations from plan & why
- None.

## Next phase
P070 — Multi-task balancing.
