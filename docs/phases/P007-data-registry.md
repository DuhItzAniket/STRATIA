# P007 — Data registry spec

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Register every dataset STRATIA may use, with source, licence (and whether it was verified), local location, expected size and status, so that later phases never read data that is not registered.

## Inputs / dependencies
Local `DATA/` folder; research notes (sources and licences); P003 environment.

## Work log
1. `configs/datasets.yaml`: 12 datasets (CCSN, MGCD, SWIMCAT, SWIMSEG family, Almería, Montenegro, Eye2Sky, DeepSky, WEBCAM, own B0268, HBMCD, LenghuSky-8) with tasks, source URL/DOI, licence + `licence_status`, local path(s), image globs where masks sit beside images, expected counts, status and notes.
2. `configs/paths.example.yaml` (committed) → `configs/paths.yaml` (machine-specific, git-ignored): `data_root`, `cache_root`, `runs_root`.
3. `stratia/data/registry.py` (load and validate the registry, resolve folders, count images) and `scripts/check_registry.py`.
4. Licences read from local files: SWIMCAT and SWIMSEG are **CC BY-NC 4.0** (non-commercial: fine for research, recorded); Montenegro CC BY 4.0; DeepSky code MIT. Others marked `unverified` until P015.

## Verification
`python scripts/check_registry.py --count`:

| Dataset | Status | Counted | Expected |
|---|---|---|---|
| ccsn | present | 2,543 | 2,543 |
| mgcd | present | 8,000 | 8,000 |
| swimcat | present | 784 | 784 |
| swimseg family | present | 8,052 | 8,052 (1,013 + 115 + 6,768 + 156) |
| almeria | present | 818 | 818 |
| montenegro | present | 2,522 | 2,522 |
| b0268 | partial | 25 | 25 |
| eye2sky | partial | (inventory in P012) | — |
| deepsky, webcam, hbmcd, lenghusky8 | not downloaded | — | — |

The first run reported double counts for SWIMSEG and Almería because masks sit beside images; fixed by per-dataset `images_glob`.

## Exit criteria
- [x] All local datasets registered.

## Fit & data-risk notes
Licence status is explicit per dataset; CC BY-NC data cannot be redistributed commercially (affects released derivatives, not research use).

## Deviations from plan & why
- Bug found while committing: the P001 ignore pattern `data/` also matched the package folder `stratia/data/`, so the registry code could not be committed. Patterns for data, runs, caches, weights and outputs are now anchored to the repository root (`/data/` etc.).

## Next phase
P008 — Interface contract v1.
