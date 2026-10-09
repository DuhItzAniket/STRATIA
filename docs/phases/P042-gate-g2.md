# P042 — Gate G2: Splits & labels sign-off

Status: DONE     Date: 2026-10-10     Commit: (this commit)

## Objective
Close Stage D: review P033–P041 against their exit criteria with evidence, rerun the mechanical checks, and sign
the gate so that Stage E (geometry) and the baselines can start on fixed labels and splits.

## Inputs / dependencies
Phase docs P033–P041; `configs/ontology.yaml`, `data/labels.parquet`, `data/montenegro_targets.parquet`,
`data/ceilometer_targets.parquet`, `configs/splits.yaml`, `data/splits/*`, `test_lock.json`,
`data/label_noise_flags.parquet`; the reports under `docs/data/`.

## Work log
1. Review of every Stage D phase against the plan's exit criterion (table below).
2. Mechanical checks rerun on 2026-10-10 (Verification).
3. P041 is BLOCKED on the owner's camera frames (no logger run exists). Decision: sign the gate **conditionally**,
   with P041 deferred and a hard latest date before P083 (B0268 zero-shot evaluation). Nothing in Stage E through
   Stage H consumes B0268 labels; the consequences are listed in P041.
4. `PROJECT_STATE.md`: G2 marked signed (conditional), stage advanced to E.

## Review

| Phase | Exit criterion (PLAN) | Evidence | Verdict |
|---|---|---|---|
| P033 WMO ontology | Reviewed against the WMO Cloud Atlas and WMO-No. 306 tables | `configs/ontology.yaml`; `tests/test_ontology.py`; review notes in P033 | passed |
| P034 Dataset → ontology | Every label in every dataset mapped or explicitly excluded | `docs/data/label_mapping.md` (exclusions with reasons); `data/labels.parquet`; `tests/test_mapping.py` | passed |
| P035 Montenegro ceiling | Ceiling table (the bar STRATIA is compared to) | `docs/data/montenegro_ceiling.md`: étage set 85.2 %, cover ±1 okta 87.3 %; C2 bar = 90 % of these | passed |
| P036 Ceilometer → targets | Target table for all paired frames | `data/ceilometer_targets.parquet` (30 s grid, both sites, weak and WMO thresholds); frames pair in P047 once the OLDLR/WESTE images are on disk (download running) | passed (table side) |
| P037 Split design | `docs/splits.md` reviewed | `docs/splits.md`, `configs/splits.yaml`; seven guarantees listed | passed |
| P038 Split generator | Tests prove no overlap (hash + group + time) | `tests/test_splits.py`; `docs/data/splits_report.md`; `data/splits/hashes.json` | passed |
| P039 Lock test sets | Lock committed | `test_lock.json` (13 sets); CI step `scripts/check_test_lock.py` | passed |
| P040 Label noise | Flag list + decisions; no automatic deletion | `data/label_noise_flags.parquet`; decisions in P040 and `docs/data/label_noise_report.md` | passed |
| P041 B0268 labelling | >= 300 frames by 2 people; agreement reported | no frames exist (owner item) | **deferred** |

## Verification
- `python -m pytest -q` (2026-10-10, with the Stage E tests already present): 149 passed.
- `python scripts/check_test_lock.py`: "test lock OK: 13 test sets; lock well-formed, configs clean, split files
  match the lock".
- `ruff check .`: clean.

## Exit criteria
- [x] Signed: **conditional**, P041 deferred with the latest date before P083; all other Stage D criteria met.

## Fit & data-risk notes
What the gate now guarantees for every later training phase: no image-level random split anywhere (units = exact
groups ∪ temporal blocks), test sets hashed and locked with a logged-reason rule, labels set-valued where the source
is ambiguous, label noise flagged (never deleted), and the human ceiling fixed before any model number exists.
What it does not: an out-of-distribution consumer-camera set. The paper's C1/C2 claims stand without it; H4 waits.

## Deviations from plan & why
The plan foresaw P041 complete (image-level half) by this gate on 18 Oct; the camera frames do not exist, so the
gate is signed conditionally eight days early to let Stage E start.

## Next phase
P043 — Camera models (Stage E).
