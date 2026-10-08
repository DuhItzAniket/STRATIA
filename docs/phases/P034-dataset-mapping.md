# P034 — Dataset → ontology mapping

Status: DONE     Date: 2026-10-08     Commit: (this commit)

## Objective
Turn every native label in the manifest into a set-valued STRATIA label through `configs/ontology.yaml`, or
exclude it for a stated reason, and merge the label conflicts P026 found, so that Stage E trains on one label table.

## Inputs / dependencies
P033 ontology; P021 manifest (`source_label`, Montenegro code distributions); P030 mask cloud fractions
(`data/mask_cloud_fraction.parquet`); P026 conflict pairs (`data/label_conflicts.parquet`).

## Work log
1. `stratia/labels/mapping.py`: `map_class` (CCSN, MGCD, SWIMCAT), `map_montenegro` (étages from the majority
   altitude classes, genera from the majority C_L / C_M / C_H codes, "not visible" kept as unknown), `map_mask_row`
   (segmentation sets: cloud presence from the mask, Almería étages from its layer masks), the conflict merge
   (union of the genus sets of exact and copy conflicts as alternatives, flagged), summaries and the report.
2. `scripts/map_labels.py`: writes `data/labels.parquet` (one row per manifest sample: `genus_set`,
   `genus_alternatives`, `extra`, `etage_set`, `cloud_present`, `label_source`, `excluded_reason`, `label_conflict`)
   and `docs/data/label_mapping.md` (every native class and what it maps to; coverage per dataset; exclusion reasons).
3. Ontology fix: MGCD classes are keyed by the manifest's `source_label` (`cumulus` … `mixed`, the folder names
   without their number), found when the first run raised `KeyError: 'cumulus'`.
4. Tests (`tests/test_mapping.py`, 3): single, merged, clear and unknown classes; Montenegro from altitude classes
   and codes incl. "not visible"; a full mapping with mask rows and a conflict merge, with summary and report.

## Verification
- Run: `python scripts/map_labels.py` (1 s): 52,032 rows.

| Dataset | Rows | Genus set | of which alternatives | Étage set | Clear | Contrail | Cloud, genus unknown | No label | Conflicts merged |
|---|---|---|---|---|---|---|---|---|---|
| CCSN | 2,543 | 2,543 | 200 | 2,345 | 0 | 199 | 0 | 0 | 200 |
| MGCD | 8,000 | 6,980 | 4,204 | 4,099 | 1,338 | 0 | 1,020 (mixed) | 0 | 0 |
| Montenegro | 2,522 | 2,363 | 286 | 2,522 | 499 | 0 | 159 | 0 | 0 |
| SWIMCAT | 784 | 224 (clear) | 0 | 224 | 224 | 0 | 560 | 0 | 0 |
| Almería | 818 | 0 | 0 | 818 | 83 | 0 | 735 | 0 | 0 |
| SWIM masks (4 sets) | 8,052 | 0 | 0 | 17 (clear frames) | 17 | 0 | 8,035 | 0 | 0 |
| Eye2Sky, B0268 | 29,313 | 0 | 0 | 0 | 0 | 0 | 0 | 29,313 | 0 |

- Every native class of CCSN, MGCD, SWIMCAT and Montenegro appears in the mapping table of the report with its
  target; nothing is unmapped and nothing is forced: 4,204 MGCD images carry a set of alternatives because their
  class merges genera, 1,020 are "mixed" (cloud, genus unknown), and the 200 CCSN images in exact or copy conflicts
  carry the union of their labels and the `label_conflict` flag.
- Tests: 3 new; `python -m pytest -q`: all pass; `ruff check .` clean.

## Exit criteria
- [x] Every label in every dataset mapped or explicitly excluded (`docs/data/label_mapping.md`, exclusion reasons
  with counts).

## Fit & data-risk notes
- **Set-valued targets are the honest reading of merged classes**; the genus loss (P065) must accept "any member"
  targets (alternatives) and "all members" targets (co-present genera) differently, which the `genus_alternatives`
  flag enables.
- **Étage sets are unknown where alternatives span étages** (MGCD "stratocumulus" = Sc/St/As: low or mid), so
  MGCD gives 4,099 étage labels, not 8,000; the étage head trains on CCSN, Montenegro, Almería (from masks) and the
  single-étage MGCD classes.
- **The segmentation sets contribute cloud presence and masks, not genera**; SWIMCAT contributes clear/cloud only.
- Labels live in `data/labels.parquet`, not in the manifest, so the manifest and everything aligned to it (features,
  hashes, groups) never change when a mapping rule does.

## Deviations from plan & why
- WEBCAM's "fog → obscured" mapping cannot be done (dataset not obtained); Montenegro's obscured code (N = 9) is
  carried as an `obscured_share` in P035's targets instead.

## Next phase
P035: Montenegro soft labels & human ceiling.
