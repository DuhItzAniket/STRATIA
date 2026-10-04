# P006 — Tests & CI

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Automated tests and continuous integration so that every push is linted, tested and secret-scanned.

## Inputs / dependencies
P004–P005 utilities.

## Work log
1. `tests/test_utils.py` (7 tests): config hash ignores key order and changes with values; `defaults` + overrides composition; seeding reproduces CPU random numbers; run artefacts and registry row; TensorBoard events; failure recorded in the registry.
2. `pyproject.toml`: ruff (line length 120; rules E, F, W, I, B, UP) and pytest settings (`gpu` marker).
3. `.github/workflows/ci.yml`: Ubuntu, Python 3.11, CPU-only PyTorch (`requirements-ci.txt`, `torch==2.5.1+cpu`), `ruff check`, `pytest -m "not gpu"`, gitleaks secret scan. No GPU or Hugging Face token is needed in CI.
4. `.pre-commit-config.yaml`: ruff, large-file guard (1 MB), YAML check, whitespace fixers, gitleaks. Installing the hook locally (`pre-commit install`) is optional; CI enforces the same checks.

## Verification
- Local: `ruff check .` → all checks passed; `pytest` → all tests pass, including with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`).
- Ruff found and I fixed one real bug before this commit: `scripts/profile_backbones.py` deleted variables in a `finally` block that an inner function still captured.
- CI result on GitHub for this commit: recorded in the next phase document (P007), since it is only known after pushing.

## Exit criteria
- [x] CI defined and green locally; GitHub result recorded in P007.

## Fit & data-risk notes
Tests run on synthetic data only; no dataset or weights are needed in CI.

## Deviations from plan & why
None.

## Next phase
P007 — Data registry spec.
