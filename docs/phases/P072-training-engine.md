# P072 — Training engine

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
The training mechanics every Stage H run shares, with none of the task logic: AMP, gradient accumulation,
clipping, EMA, checkpoint / resume, a NaN guard, early stopping on a validation metric, per-step logging.

## Inputs / dependencies
P005 `Run` logging; P010 (batch 16 at 512 px: accumulation and AMP are how larger effective batches fit); P070
weighting (a step function's concern, not the engine's); P071 sanity suite.

## Work log
1. `stratia/train/engine.py`: `EngineConfig` (epochs, accumulate, clip_norm, amp, ema_decay, max_nan_steps,
   monitor / mode / patience / min_delta, checkpoint_every, log_every, device); `Engine(model, optimizer, cfg,
   scheduler, log, checkpoint_dir)` with `train_step(batch, step_fn)` (autocast + GradScaler on CUDA, loss /
   accumulate, optimizer step every `accumulate` micro-batches, pre-clip gradient norm reported, scheduler and EMA
   stepped per optimizer step), `evaluate(eval_fn)` on the EMA weights (restored after), best / patience tracking
   and a `best.pt`, `fit(loader_fn, step_fn, eval_fn, resume)` with `last.pt` every `checkpoint_every` optimizer
   steps and at every epoch end. The NaN guard skips a non-finite loss, zeroes the gradients and raises after
   `max_nan_steps` in a row. Checkpoints hold model, optimizer, scheduler, scaler, EMA, step, epoch, position
   inside the epoch, early-stopping state, history and the Python / NumPy / torch (/ CUDA) RNG states, written
   atomically (`.tmp` + replace).
2. Resume contract: `loader_fn(epoch)` must order its batches as a function of the epoch and a fixed seed; the
   engine replays the epoch's batches up to the checkpointed position and restores the RNG states, so the resumed
   run is the uninterrupted run.

## Verification
- `tests/test_engine.py` (3): a 3-epoch run with accumulation 2, EMA and a step scheduler, interrupted by a
  simulated crash inside epoch 1 after a checkpoint and resumed in a differently initialised model, ends with
  bit-identical parameters, EMA weights and validation history to the uninterrupted run; accumulation over two
  half-batches equals one full batch step (eval mode, no dropout) and clipping bounds the applied update while
  reporting the pre-clip norm; the NaN guard counts, resets on a finite loss and raises after the limit; early
  stopping stops after `patience` evaluations without improvement and leaves `best.pt`.

## Exit criteria
- [x] Resume reproduces metrics.

## Fit & data-risk notes
- Early stopping reads validation only (locked-test policy); the monitored key is per run config.
- EMA evaluation means the reported numbers belong to the averaged weights; the live weights are what `last.pt`
  resumes from and what `best.pt` stores alongside the EMA shadow.

## Deviations from plan & why
- None.

## Next phase
P073 — Stage H, Stage-1 global heads on cached features.
