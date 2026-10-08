# P040 — Label-noise estimation

Status: DONE     Date: 2026-10-08     Commit: (this commit)

## Objective
Estimate how many native class labels a frozen-feature probe disagrees with confidently (confident-learning
style), review the most confident flags by eye, and record decisions; no automatic deletion or relabelling.

## Inputs / dependencies
P025 features (DINOv3 ViT-S/16 CLS); P038 units (folds by unit, so near-duplicates never inform each other's
out-of-fold probability); P029 (probe accuracies: CCSN 46 %, MGCD 85 %, SWIMCAT 100 %); P026/P035 (conflicts,
rater agreement).

## Work log
1. `stratia/labels/noise.py`: out-of-fold probabilities from a regularised logistic-regression probe (C = 0.1; the
   first run with C = 1.0 saturated at 1.0 and made every flag look certain); confident-joint flags (an image is
   flagged when another class reaches its own mean-probability threshold and is the most probable of those) with a
   **strong** tier (suggested probability ≥ 0.5 and margin ≥ 0.25); summaries, contact sheets, report.
2. `scripts/label_noise.py`: CCSN (11 genera), MGCD (7 types), SWIMCAT (5), Montenegro primary class (5); writes
   `data/label_noise_flags.parquet`, `docs/data/label_noise_report.md`, `docs/data/figures/label_noise_*.jpg`.
3. Tests (`tests/test_label_noise.py`, 2): planted label flips are mostly recovered and clean labels rarely flagged
   on synthetic data; report and contact sheet.
4. Reviewed the sheets (Verification).

## Verification
- Run: `python scripts/label_noise.py` (17 s).

| Dataset | Images | Flagged | Strong flags | Most frequent given → suggested |
|---|---|---|---|---|
| CCSN | 2,543 | 1,173 (46.1 %) | 742 (29.2 %) | Sc → St 58, Cc → Ac 53, Ns → St 42, St → Sc 40 |
| MGCD | 8,000 | 554 (6.9 %) | 554 (6.9 %) | stratocumulus ↔ cumulonimbus, altocumulus → mixed, mixed → cirrus |
| SWIMCAT | 784 | 0 | 0 | — |
| Montenegro | 2,522 | 1,396 (55.4 %) | 270 (10.7 %) | low → vertical development 597, clear → vertical development 227 |

- **Reading the rates.** The plain flag rate is only meaningful when the probe is good: CCSN's 46 % and Montenegro's
  55 % come from a weak probe (eleven fine genera; a 3 % "vertical development" class whose threshold is near zero)
  and overstate noise; the strong tier is the reviewable estimate (CCSN 29 %, MGCD 7 %, Montenegro 11 %, SWIMCAT 0).
- **Review of the sheets.** CCSN's strongest flags are mostly real errors or neighbour confusions: a lone cumulus
  labelled Ac, towering cumulonimbus labelled Ns or Sc, low grey decks labelled Ci, cirrus labelled Cs or Ac, and
  several landscape photographs with little sky; the pairs match P026's conflict pairs (Cc/Cs, Ac/As, Sc/St, Cb/Cu).
  MGCD's flags sit on its merged-class boundaries (thin cirrus labelled "mixed", uniform grey overcast labelled
  "cumulonimbus"), the same boundaries where consecutive frames flip label (P026). Montenegro's flags are uniform
  grey overcast frames whose rater majority says "clouds of vertical development": the probe's "low clouds" is the
  more defensible reading, and the pattern says the raters used that class for dark skies.
- Tests: 2 new; `python -m pytest -q`: all pass; `ruff check .` clean.

## Exit criteria
- [x] Flag list: `data/label_noise_flags.parquet` (given, suggested, probabilities, margin, flagged, strong).
- [x] Decisions: section "Review and decisions" of the report and the notes below; nothing deleted or relabelled.

## Fit & data-risk notes
- **CCSN carries real label noise at the genus level** (P026 and this phase agree): the paper reports CCSN genus
  results with and without strongly flagged test images, and the étage-level numbers as the reliable ones.
- **Montenegro's "clouds of vertical development" is not a reliable class**: it is rare, disagreed on (P035) and
  mostly flagged here. Evaluation on Montenegro folds it into the low étage (as the ontology does) and does not
  score it as a class of its own.
- **MGCD's noise is boundary noise** between merged classes, consistent with set-valued targets (P034): a model
  should be scored on the set, not the single type.
- Flags may down-weight training samples (P065) but never remove them; the flag shares go into the data card as a
  third label-noise estimate next to the conflict rates and the rater agreement.

## Deviations from plan & why
- The plan marks the phase deferrable; it was done now because the flags feed the evaluation design (with/without
  flagged test images) and the data card. Manual review covered the top sixteen flags per dataset (contact sheets),
  not every flag.

## Next phase
P041: B0268 labelling protocol (needs the logger frames and two labellers).
