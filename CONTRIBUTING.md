# Contributing to STRATIA

STRATIA is developed phase by phase following [`docs/PLAN.md`](docs/PLAN.md). These rules apply to every change, human- or AI-written.

## Phase workflow

1. Copy [`docs/phases/TEMPLATE.md`](docs/phases/TEMPLATE.md) to `docs/phases/P###-<slug>.md` at the start of the phase.
2. Do the work; record commands, config hashes, run IDs and outputs.
3. Verify every exit criterion with evidence.
4. Update [`PROJECT_STATE.md`](PROJECT_STATE.md), [`CHANGELOG.md`](CHANGELOG.md) and, for experiments, [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
5. Commit and push:
   ```
   git commit -m "P###: <title>" -m "<what changed and how it was verified>" -m "Phase-Status: DONE"
   git push origin main
   ```
   `Phase-Status` is `DONE`, `PARTIAL` or `BLOCKED`.
6. End of a stage: `git tag -a stage-<X>-complete -m "..."` and `git push origin --tags`.

## Research honesty rules

1. **No fabricated numbers.** Every metric in a document, table or the paper comes from a run listed in `runs/registry.csv` with its git SHA, config hash and data hash.
2. **No fabricated labels.** Labels come from a dataset, from documented human annotation, or are marked as pseudo-labels with their source. Pseudo-labels are never mixed into evaluation sets.
3. **The test sets are locked** (`test_lock.json`, from P039). They are evaluated once, at P090, with a logged reason. Model selection, early stopping, thresholds and prompt choices use validation data only.
4. **Negative results are kept.** Failed experiments are written up in the phase document and the experiment log, not deleted.
5. **Every citation is verified** to exist (title, authors, venue) before it enters the paper.
6. **Splits come from the split generator** (P038), never ad hoc; group, site and time leakage are tested automatically.

## Never commit

Datasets, images, feature caches, run folders, model weights, Hugging Face tokens or other secrets, or `configs/paths.yaml` (machine-specific). Weights are released via GitHub Releases or Hugging Face; datasets are referenced by manifest + SHA-256.

## Code style

Python 3.11, formatted and linted with `ruff`; tests with `pytest` (CPU-only tests in CI). Loaders assert label mappings explicitly (a mismatched `class_to_idx` once trained Cu images as Ac in the CloudScope prototype).
