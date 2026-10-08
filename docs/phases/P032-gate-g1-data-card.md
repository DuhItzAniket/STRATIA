# P032 — Gate G1: Data audit sign-off

Status: DONE     Date: 2026-10-08     Commit: (this commit)

## Objective
Close Stage C: write the data card that collects every Stage C finding in the Datasheets-for-Datasets structure,
list the open items with owners, and sign the gate with zero blockers for Paper A.

## Inputs / dependencies
P023–P031 reports under `docs/data/`; P028 policy; `configs/datasets.yaml` licences; the manifest.

## Work log
1. `docs/data/data_card.md`: motivation, composition (sources, cameras, labels, spans, licences), collection,
   preprocessing and cleaning (one paragraph per audit phase with the key numbers and the report it comes from),
   intended and unintended uses, distribution and licensing, maintenance, known limitations and open items with
   owners, and the sign-off table.
2. Housekeeping found by the audit and fixed here: the manifest builder now parses Almería's timestamps from its
   file names (`scripts/build_manifest.py`), so every time-series dataset has `utc` for day blocking, and it pins
   the Eye2Sky days it ingests (`EYE2SKY_DAYS`, 1–9 April 2022): the first rebuild attempt pulled in all 542,503
   frames of the April–July download that has arrived on disk since P021, which is not what the audit, the cache
   and the feature files were built on. With the pin, the manifest was rebuilt and checked to contain the same
   52,032 rows in the same order (only the Almería `utc` and `time_source` columns changed), so every cached table
   and feature file aligned to it stays valid.
3. `PROJECT_STATE.md`: gate G1 marked signed; blockers reworded (the missing ceilometer-site images block Paper B's
   pairing, not Paper A).

## Verification
- `python scripts/build_manifest.py` rebuilt the manifest: validation passed; Almería rows with `utc`: 818 of 818;
  `sample_id` order identical to the previous manifest (content hash `464078b6d033dd39` unchanged); all other
  columns identical. `docs/data/manifest_summary.md` regenerated.
- The data card's numbers were copied from the Stage C reports and spot-checked against them.
- `python -m pytest -q`: all pass; `ruff check .` clean.

## Exit criteria
- [x] Data card written (`docs/data/data_card.md`).
- [x] Signed with blockers = 0 for Paper A; open items listed with owners and phases.

## Fit & data-risk notes
- The gate is signed for Paper A. Paper B has one hard dependency outside the repository: images at the ceilometer
  stations (OLDLR, WESTE) for April–July 2022. Until they arrive P036 builds the target table on the ceilometer side
  only.
- The decision to keep nine Eye2Sky days in the manifest is deliberate (audit and cache tractability); P038 decides
  whether more days are ingested at a lower cadence for the splits.
- Licences constrain distribution of weights: anything trained with the SWIM family is non-commercial.

## Deviations from plan & why
- The plan's exit criterion is "signed; blockers = 0". The Almería timestamp fix was added here because the data
  card could not honestly say "every time-series dataset has timestamps" without it.

## Next phase
P034: Dataset → ontology mapping (P033 done ahead of this gate).
