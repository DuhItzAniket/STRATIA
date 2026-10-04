# P001 — Repo bootstrap

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Initialise the empty STRATIA repository with a README, licence, ignore rules and the package skeleton, and prove that pushing works.

## Inputs / dependencies
Empty GitHub repository `DuhItzAniket/STRATIA` (default branch `main`); plan `STRATIA/plans/STRATIA_Implementation_Plan.md`.

## Work log
1. Cloned the empty repository into `C:/Users/Luikz/Downloads/STRATIA/repos/STRATIA`.
2. Added `README.md` (scope: model only; CloudScope owns hardware), `LICENSE` (Apache-2.0, same text as CloudScope, SHA-256 `cfc7749b96f63bd3…`), `.gitattributes` (LF normalisation, as in CloudScope), `.gitignore` (data, runs, caches, weights, model files, machine paths, secrets, paper build outputs).
3. Created the package `stratia/` (`__version__ = "0.0.1"`). Other folders (`configs/`, `scripts/`, `tests/`, `docs/adr`, `docs/data`, `paper/`) receive content in their phases (git does not track empty folders).

## Verification
- `git push -u origin main` succeeded (see commit).
- `.gitignore` excludes `*.safetensors`, `*.pth`, `*.onnx`, `data/`, `runs/`, `caches/`, `configs/paths.yaml`, `hf_token*`.

## Exit criteria
- [x] First push to GitHub succeeds.

## Fit & data-risk notes
None yet. Ignore rules make it hard to commit datasets or weights by accident.

## Deviations from plan & why
- Licence: Apache-2.0 chosen for consistency with CloudScope (owner confirmed Apache-2.0 for CloudScope at its Gate R); can be changed before the first release.

## Next phase
P002 — Phase protocol & logs.
