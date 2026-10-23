# Split design (P037)

What a split is allowed to cut, which protocols the paper reports, and the guarantees the generator (P038) must
prove. Parameters live in `configs/splits.yaml`; every number here comes from a Stage C report.

## 1. The unit a split may not cut

A **unit** is a connected component of two relations over images:

- *same group*: the exact-duplicate groups of P024 (661 images, pixel-identical copies) everywhere; the
  near-duplicate links of P025 (dihedral-pHash copies and same-scene pairs at DINOv3 cosine ≥ 0.97) only where no
  temporal block exists (CCSN, MGCD, SWIM family). Inside a time series the block is the unit: a pHash match between
  smooth-sky frames days apart or at the other station is a coincidence, not a copy (P026), and a same-scene link
  across days or stations is a weather look-alike (the 448-image Eye2Sky group of P025 spans several days and both
  stations). Applied, those links fuse eight of nine Eye2Sky days and Montenegro's ten weeks into three units.
  Copies cross datasets inside the SWIM family (34.6 % of SWINySEG are copies of SWIMSEG/SWINSEG), which is why the
  family is one source.
- *same block*: the temporal blocks of P027: one calendar day for Eye2Sky (both stations on the same date share a
  block, since same-moment frames 15 km apart are correlated through the weather), contiguous 7-day blocks for
  Montenegro (its frames stay correlated across days), one calendar day for Almería and for B0268.

Splitting by units, not by images, is what makes the published random splits of MGCD (701 same-scene pairs across
train/test), SWINySEG and SWIMCAT (duplicates on both sides) and Almería (193 same-scene pairs across its splits)
unrepeatable here. The official splits are reported only as "published protocol" numbers for comparison.

## 2. Protocols

| Protocol | What it measures | Train | Test | Notes |
|---|---|---|---|---|
| **In-domain, blocked** (per source) | performance on a source with no leakage | 70 % of units | 20 % of units (10 % val) | greedy stratified assignment of units; conflict-merged images (P026) never in test |
| **Leave-one-dataset-out (LODO)** | transfer to an unseen source (the paper's main claim, H1, criterion C1) | train + val units of all other sources | the held-out source minus its in-domain val units (locked); its in-domain val units are the LODO *val* set (model selection, gate G4) — amendment of 2026-10-11, config version 2 | folds: CCSN, MGCD, Montenegro, SWIM family, Almería |
| **Held-out station** (Eye2Sky) | transfer to an unseen camera of the same type | AURIC | BARSE | two variants: same days (shared weather) and disjoint days (P027) |
| **Few-shot target** (B0268) | adaptation to the user's camera | k whole adaptation days of B0268 (k = 0, 1, 2) plus the sources | *val*: the other adaptation days (the gain is measured here); *test*: the locked day 2026-10-20 | P041 (2026-10-22): 393 labelled frames over 4 days; the locked test day is the last day with >= 100 labelled frames; every combination of k adaptation days is a draw (7 draws: `k0_d0`, `k1_d0`..`k1_d2`, `k2_d0`..`k2_d2`); k = 2 replaces k = 3 because 4 days leave 3 adaptation days; B0268 is excluded from LODO training |
| **Published protocol** (MGCD, Almería) | comparison with the literature | their train split | their test split | leaks by construction; reported, never used for model selection |

Stratification balances each source's native labels across its splits (CCSN genera, MGCD sky types, SWIMCAT
categories, Montenegro's primary class, Almería's cloud-fraction bin, the SWIM members' shares), so the class ratios
of P030 hold in every split. Sampling inside training follows P030 (square-root source sampling, class-balanced
weights); the split only decides membership.

## 3. Guarantees the generator must prove (P038)

1. No group id has members in two splits (exact and near-duplicate groups).
2. No temporal block has members in two splits; for Eye2Sky, no date appears in two splits for either station.
3. No `label_conflict` image is in a test split.
4. In LODO, the held-out source's images (and every image in a group linked to them) are absent from training.
5. In the held-out-station variant `disjoint_days`, no training frame shares a date with a test frame.
6. Every split file carries a SHA-256 of its sorted sample ids; the test files are locked (P039) and any final
   evaluation on them logs a reason.
7. The assignment is deterministic (seed in the config) and re-running the generator reproduces the hashes.

## 4. What is deliberately not done

- No buffer days between blocks of different splits (P027: Eye2Sky and Almería reach the different-day level within
  a day; Montenegro's residual week-to-week correlation of 0.09–0.17 is accepted and reported rather than paid for
  with an 11-day gap that would cost a sixth of the data).
- No class-uniform resampling and no removal of duplicates: groups stay whole and are counted once in the data card.
- No image-level random split anywhere, not even for ablations.
- **B0268 (P041, 2026-10-22).** Four labelled afternoons exist, not the six the plan asked for. The rule for the
  locked test day was fixed before any number was looked at: the last `test_days` (1) calendar days with at least
  `min_frames_per_test_day` (100) labelled frames. It picks 2026-10-20 (120 frames); the 33-frame 2026-10-21
  session is too small. The adaptation days are 2026-10-18, 2026-10-19 and 2026-10-21. The draws are every
  combination of k adaptation days for k in {0, 1, 2}, at most `repeats` (5) per k, so none is dropped:

  | Draw | Train days | Val days (gain measured here) | Test |
  |---|---|---|---|
  | `k0_d0` | none | 10-18, 10-19, 10-21 (273 frames) | 10-20 (120) |
  | `k1_d0` / `k1_d1` / `k1_d2` | 10-18 / 10-19 / 10-21 | the other two (153 / 153 / 240) | 10-20 (120) |
  | `k2_d0` / `k2_d1` / `k2_d2` | 10-18+10-19 / 10-18+10-21 / 10-19+10-21 | 10-21 / 10-19 / 10-18 (33 / 120 / 120) | 10-20 (120) |

  The locked day is never evaluated before the final run. The in-domain assignment of b0268 (`in_domain/b0268/*`,
  5 units: the 4 labelled days and the legacy day) is vestigial: it mirrors the protocol (test = the locked day,
  everything else train) so the lock carries one notion of the target's test set, but b0268 is not a training
  source (`stratia.train.data.GLOBAL_SOURCES`) and `lodo_folds` excludes the few-shot target from every LODO
  training set. The 25 unlabelled legacy frames are not part of the protocol. No buffer day between the test day
  and the adaptation days: 2026-10-19 and 2026-10-21 are adjacent to it, which is the same choice as for Eye2Sky
  and Almería above.

## 5. Sizes to expect

Units per source (P024/P025/P027): CCSN about 2,200 (149 small groups), MGCD about 450 (227 same-scene groups plus
singletons), Montenegro 10 weekly blocks, SWIM family about 5,000 (1,700 groups plus singletons), Almería about 200
days, Eye2Sky 9 days (both stations), B0268 1 day. Montenegro and Eye2Sky therefore have few units: their in-domain
splits are coarse (whole weeks, whole days), which is the honest resolution of a time series; the paper reports
them with that caveat, and P038 decides whether to ingest more Eye2Sky days at a lower cadence.
