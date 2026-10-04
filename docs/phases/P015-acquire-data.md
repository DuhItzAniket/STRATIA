# P015 — Acquire missing public data

Status: PARTIAL     Date: 2026-10-04     Commit: (this commit)

## Objective
Obtain the remaining public datasets (DeepSky, WEBCAM, HBMCD, GCD, optional LenghuSky-8) with verified licences and checksums.

## Inputs / dependencies
P007 registry; P012 (Eye2Sky gap).

## Work log
1. Queried official sources without downloading: Zenodo API (Almería, Montenegro, Eye2Sky, DeepSky), Harvard Dataverse API (CCSN), GitHub API and READMEs (DeepSky, WEBCAM, HBMCD, TJNU, LenghuSky-8), Hugging Face API (LenghuSky-8).
2. **Findings that change the registry:**
   - DeepSky's Zenodo record (8208505), assumed by the research notes to hold the images, contains **only the paper**; the images are available on request from the authors.
   - WEBCAM has no download link in its repository or code; the paper's data-availability statement must be checked.
   - HBMCD is only on Baidu Pan; GCD needs a signed agreement.
3. Licences verified and recorded: CCSN CC0 1.0, Almería CC BY 4.0, Montenegro CC BY 4.0, Eye2Sky CDLA-Sharing 1.0, LenghuSky-8 Apache-2.0.
4. `configs/datasets.yaml` updated; owner action list in `docs/data/acquisition.md`.

## Verification
`scripts/check_registry.py` passes; registry tests pass. Licence evidence is listed per dataset in `docs/data/acquisition.md`.

## Exit criteria
- [ ] DeepSky, WEBCAM, HBMCD, GCD registered **with data on disk** — not met: each needs an owner action (email request, paper lookup, Baidu account, signed agreement).
- [x] Licences verified where an official record exists.

## Fit & data-risk notes
Fewer recognition datasets means fewer leave-one-dataset-out folds (four sources remain: CCSN, MGCD, Montenegro, B0268). The cloud-base-height objective depends entirely on the pending Eye2Sky download at OLDLR/WESTE.

## Deviations from plan & why
Status PARTIAL: acquisition requires owner actions; the phase is re-opened when data arrives (checksums and counts then recorded here).

## Next phase
P016 — Montenegro parser.
