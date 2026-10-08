# P039 — Lock test sets

Status: DONE     Date: 2026-10-08     Commit: (this commit)

## Objective
Make test peeking impossible to do silently: hash every test set of every protocol into a committed lock, require a
logged reason for any final evaluation on them, and let CI refuse training configs that reference a test manifest.

## Inputs / dependencies
P038 split files (`data/splits/*.parquet`, git-ignored) and their hashes; the manifest content hash (P032).

## Work log
1. `stratia/data/test_lock.py`: test entries (name → count and SHA-256 of sorted sample ids) for the in-domain,
   LODO and held-out-station files; `build_lock`, `verify_lock` (missing, changed and unlocked files are all
   reported), `well_formed`, `require_final_reason` (appends to `runs/final_eval_log.md`, refuses reasons under ten
   characters), `scan_training_configs` (a config with a `train` or `training` section may not name a test file or
   `split: test`).
2. `scripts/lock_test_sets.py` (refuses to overwrite without `--force`) wrote `test_lock.json`: 13 test sets, 53,252
   ids (seven in-domain, five LODO, two station variants minus overlaps), with the config version, seed and manifest hash.
3. `scripts/check_test_lock.py` runs in CI (new workflow step): lock present and well-formed, configs clean; locally
   it also compares the lock with the split files.
4. Tests (`tests/test_test_lock.py`, 3): lock built, verified and tampering detected (changed split, missing file,
   malformed entry); final reason required and logged; training-config scanner flags bad configs and leaves
   evaluation configs alone.

## Verification
- `python scripts/lock_test_sets.py` → `test_lock.json`; `python scripts/check_test_lock.py` → "test lock OK: 13 test
  sets; lock well-formed, configs clean, split files match the lock".
- `python -m pytest -q`: all pass; `ruff check .` clean.

## Exit criteria
- [x] Lock committed (`test_lock.json`); `eval --final` hook ready (`require_final_reason`, to be wired into the
  evaluation CLI in Stage F); CI check in `.github/workflows/ci.yml`.

## Fit & data-risk notes
- The lock binds the test sets to the manifest hash and the split config; regenerating splits with another seed or
  manifest changes the hashes and fails the local check until a deliberate `--force` re-lock, which the commit must
  explain.
- The B0268 test days are not in the lock yet (no labelled days); P041 adds them with a re-lock.

## Deviations from plan & why
- None.

## Next phase
P040: Label-noise estimation.
