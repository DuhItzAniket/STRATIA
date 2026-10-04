# P005 — Experiment tracking

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Every run records its metrics and is listed in a run registry, so every number in the project can be traced to a run, a git commit, a config hash and a data hash.

## Inputs / dependencies
P004 (`Run` folders).

## Work log
1. `Run.log()` appends to `metrics.jsonl` and, optionally, TensorBoard (`tb/`); `Run.finish()` writes `summary.json`.
2. On exit (normal or exception), `Run` appends a row to `runs/registry.csv`: run ID, start/end time, phase, name, git SHA, dirty flag, config hash, data hash, status (`done` or `failed: <Exception>`), summary metrics, run folder.
3. Fixed a robustness bug found while testing with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`): the environment snapshot called `get_device_name(0)` when PyTorch reported CUDA available but zero devices visible; it now checks the device count.
4. `EXPERIMENT_LOG.md` gains its first entry (reproducibility check).
MLflow was not added; the CSV registry plus TensorBoard covers the needs of a single-machine project.

## Verification
- `runs/registry.csv` contains the four P004 reproducibility runs (two invocations × two runs), all `done`, config hash `b011a8395a`.
- Unit tests (committed with P006): run artefacts and registry row written; failure recorded as `failed: RuntimeError`; TensorBoard events file written.

## Exit criteria
- [x] Registry row written automatically by a dummy run.

## Fit & data-risk notes
The registry's `git_dirty` flag shows runs made from uncommitted code; such runs may inform debugging but not reported results.

## Deviations from plan & why
MLflow optional and not used (see above).

## Next phase
P006 — Tests & CI.
