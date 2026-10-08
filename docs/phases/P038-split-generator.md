# P038 — Split generator

Status: DONE     Date: 2026-10-08     Commit: (this commit)

## Objective
Implement the split design (P037) with guarantees that are proven by code before anything is written: no group and
no temporal block crosses a split, conflict-merged images never reach a test set, held-out sources are absent from
LODO training, cameras and dates are separated in the station protocol, and every split file carries a hash.

## Inputs / dependencies
P037 (`configs/splits.yaml`, `docs/splits.md`); P024 exact groups; P025 near-duplicate pairs and groups; P027
blocks; P030 cloud-fraction bins; P034 labels (`label_conflict`).

## Work log
1. `stratia/data/splits.py`: block keys, union-find units, greedy stratified assignment of units (fills the split
   whose stratum deficit the unit reduces most; conflict-merged units choose between train and val only), in-domain
   split per source, LODO folds (tainted units removed), held-out-station variants, `verify` (the guarantees of
   `docs/splits.md` §3), SHA-256 hashes of sorted sample ids, summaries and the report.
2. `scripts/make_splits.py`: builds the group keys and blocks, runs the protocols, refuses to write when a guarantee
   fails, writes `data/splits/{in_domain,lodo,held_out_station}.parquet`, `data/splits/hashes.json`,
   `docs/data/splits_report.md`.
3. Tests (`tests/test_splits.py`, 3): units join groups and blocks (both stations of a day in one unit, 7-day
   Montenegro blocks); the assignment balances strata, respects exclusions, keeps units whole, is deterministic and
   passes `verify`; LODO and station protocols, a planted group violation is caught, hashes and report.
4. **Unit rule refined on the real data** (recorded in `docs/splits.md` and the config): the first run fused eight of
   nine Eye2Sky days and Montenegro's ten weeks into three units, because P025's near-duplicate links inside a time
   series (pHash matches between smooth-sky frames days apart or at the other station, same-scene links across days)
   are coincidences and weather look-alikes, not copies (P026, P027). Inside a time series the block is now the unit
   and only exact duplicates add links; near-duplicate links apply where no block exists (CCSN, MGCD, SWIM family).

## Verification
- Run: `python scripts/make_splits.py` (3 s). All guarantees hold; 8,661 units over 52,032 images, the largest unit
  is one Eye2Sky day (3,320 frames of both stations).

| Source | Images | Units | Train | Val | Test | Test share |
|---|---|---|---|---|---|---|
| CCSN | 2,543 | 2,382 | 1,776 | 255 | 512 | 20.1 % |
| MGCD | 8,000 | 452 | 5,472 | 959 | 1,569 | 19.6 % |
| SWIM family | 8,836 | 5,588 | 6,182 | 886 | 1,768 | 20.0 % |
| Montenegro | 2,522 | 11 (weeks) | 1,565 | 511 | 446 | 17.7 % |
| Almería | 818 | 218 (days) | 570 | 84 | 164 | 20.0 % |
| Eye2Sky | 29,288 | 9 (days) | 19,476 | 3,288 | 6,524 | 22.3 % |
| B0268 | 25 | 1 | 25 | 0 | 0 | — (few-shot protocol, P041) |

- Stratum shares stay within a point of each other across splits for CCSN (every genus), MGCD, the SWIM members and
  Almería's cloud-fraction bins; Montenegro's eleven weekly units allow only a coarse balance (low cloud 59 / 52 /
  61 %, clear 22 / 27 / 14 %), which is the resolution of a weekly-blocked time series.
- LODO folds and the station protocol (disjoint days: AURIC 9,738 train frames on six days, BARSE 4,906 test frames
  on the other three; same days: 14,644 / 14,644) are listed with their hashes in `docs/data/splits_report.md`.
- `python -m pytest -q`: all pass; `ruff check .` clean.

## Exit criteria
- [x] Tests prove no overlap (hash + group + time): `verify` runs on every generation and the unit tests plant a
  violation and see it caught; hashes in `data/splits/hashes.json`.

## Fit & data-risk notes
- **A unit rule can over-merge as easily as under-merge.** Taken literally, the P025 groups would have turned
  Eye2Sky into two units; the audits' own caveats (P026: pHash is unreliable on smooth skies; P027: look-alikes
  across days are weather) had to be applied to the split rule. The generator prints unit counts per source so that
  such a collapse is visible immediately.
- Eye2Sky's in-domain split has six, one and two days; any in-domain number on it is a nine-day statistic. More
  days at a lower cadence are the fix if Eye2Sky in-domain results are ever headline numbers (open item in the data card).
- The greedy assignment is deterministic for the config seed; changing the seed changes the hashes, and P039 locks
  the test files so that cannot happen silently.

## Deviations from plan & why
- Near-duplicate links are not applied inside time-series datasets (reason above); documented in P037's design file
  as the revised unit definition.

## Next phase
P039: Lock test sets.
