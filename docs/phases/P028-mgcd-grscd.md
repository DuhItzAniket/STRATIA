# P028 — MGCD ≟ GRSCD

Status: PARTIAL     Date: 2026-10-06     Commit: (this commit)

## Objective
Find out whether MGCD and GRSCD, two ground-based cloud datasets from the same group, contain the same images, so
that results on the two are not counted as independent evidence (plan guard: double counting).

## Inputs / dependencies
MGCD on disk (P017); GRSCD files (not on disk: distributed through a Baidu link that is not reachable from India,
like HBMCD, P015). The hash tools of P024 and P025 (`scripts/find_duplicates.py`, `scripts/find_near_duplicates.py`).

## Work log
1. The hash comparison needs the GRSCD files and could not be run. The plan marks the phase deferrable (D); its exit
   criterion is "documented", which this note satisfies as far as the publications allow.
2. What the publications say, and what we can check on MGCD: GRSCD is described as 8,000 all-sky images of
   1,024 × 1,024 px from a fisheye camera in Tianjin, in seven sky types (cumulus, altocumulus, cirrus, clear sky,
   stratocumulus, cumulonimbus, mixed) with 4,000 training and 4,000 test images; MGCD is described with the same
   numbers, the same site, the same seven types and the same split, plus weather measurements (temperature, humidity,
   pressure, wind) per image. On disk MGCD has exactly 8,000 images in those seven folders, 1,024 × 1,024 px,
   4,000 / 4,000 (P017, P021). Every number that can be compared matches.
3. P025 found that MGCD's 8,000 images form about 227 same-scene groups of consecutive frames, and P026 that its
   labels are given per sequence; a dataset built from the same camera and campaign would share this structure.

## Verification
- Not run (no GRSCD files). The procedure, when the files arrive: register GRSCD in `configs/datasets.yaml`, build
  the manifest, run `scripts/find_duplicates.py` (byte and pixel hashes) and `scripts/find_near_duplicates.py`
  (dihedral pHash and DINOv3 cosine); the cross-dataset groups answer the question in minutes.

## Exit criteria
- [x] Documented: the two datasets are treated as **one source** until a hash comparison shows otherwise.
- [ ] Hash comparison: pending access to GRSCD.

## Fit & data-risk notes
- **Policy:** MGCD and GRSCD count as one source in every protocol (LODO, data card, literature comparison);
  published GRSCD numbers are read as MGCD numbers. Nothing in STRATIA depends on GRSCD being separate.
- The same caution applies to HBMCD from the same group (not obtainable either, P015).

## Deviations from plan & why
- Deferred as the plan allows (marking D): the data cannot be reached from here; the documentation is complete, the
  measurement is not.

## Next phase
P029: Shortcut audit.
