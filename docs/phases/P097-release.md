# P097 — Release preparation

Status: DONE (prepared; publication is the owner's action)     Date: 2026-11-05     Commit: (this commit)

## Objective
Plan (Stage K, P097): reproduction scripts (`scripts/reproduce_table*.py`), data cards, licences, weights on a
GitHub Release or Hugging Face; exit criterion "fresh-clone reproduction of one table works".

## Inputs / dependencies
- `results/P090_locked_test.json` (+ `.sha256`), the Stage J outputs `data/ablations.parquet`,
  `data/human_ceiling_model.parquet`, `data/efficiency.json`, `data/error_analysis.parquet`; the data card
  (`docs/data/data_card.md`) and `configs/datasets.yaml` for the licences; the model package of P084/P096
  (`models/stratia-1.0.1-demo/`, git-ignored).

## Work log
1. `scripts/reproduce_tables.py`: every paper table (`paper/tables/tab1_criteria … tab7_efficiency.{tex,md}`) and
   every number quoted in the prose (`paper/numbers.tex`, 43 macros) are generated from the committed result files;
   the Stage J outputs are copied into `results/` (27 KB in all) so that a fresh clone reproduces everything with
   `python scripts/reproduce_tables.py` and no data. Checked by deleting nothing but pointing the script at
   `results/` first (`result_file()`), then regenerating: identical tables.
2. `docs/release/licences.md`: code Apache-2.0; weights research-use / non-commercial because the SWIM family
   (CC BY-NC 4.0) and MGCD (research terms) are in training; per-source licence table with the record each was
   verified against; images never redistributed.
3. `docs/release/release_notes_1.0.1-demo.md`: what the graph is, how 1.0.1 differs from 1.0.0 (heads adapted to
   B0268, calibration refitted, dense / CBH unchanged), the locked-test numbers, known limitations, the package
   (`stratia-1.0.1-demo_2026-10-23.zip`, 88.1 MB, SHA-256 `b41202af…`, in the owner's OneDrive backup) and the
   publishing steps.
4. CHANGELOG entries for P090–P097.
5. **Not done, by design:** no GitHub Release, no Hugging Face upload. The owner decides and presses the button
   (P100); the rule since Stage I is that nothing of the owner's data or weights goes public without their word.

## Deviations from the plan
- INT8 / accuracy-drop report (P096 second half) and the Pi 5 student (P095) remain deferred; the release ships the
  FP32 graph only.

## Outputs
- `scripts/reproduce_tables.py`, `results/{ablations.parquet, human_ceiling_model.parquet, efficiency.json, error_analysis.parquet}`,
  `paper/tables/`, `paper/numbers.tex`, `docs/release/licences.md`, `docs/release/release_notes_1.0.1-demo.md`,
  `CHANGELOG.md`.
