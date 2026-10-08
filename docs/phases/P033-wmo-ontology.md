# P033 — WMO ontology

Status: DONE     Date: 2026-10-08     Commit: (this commit)

## Objective
One file that says what a label means: the ten WMO genera and their étages, the extra output classes (clear,
contrail), the WMO code tables the Montenegro raters used (C_L, C_M, C_H, h, N) and every dataset's native classes,
each mapped to genera or explicitly excluded, so that P034 can harmonise labels without inventing anything.

## Inputs / dependencies
Interface contract v1 (`docs/contract.md`: `genus_logits` order, `etage_logits`, `oktas_probs`, `cbh_probs`);
dataset registry (`configs/datasets.yaml`); P026 (which genus pairs conflict); P036's weak height thresholds.

## Work log
1. `configs/ontology.yaml`: genera with primary étage and the étages they extend into; `genus_classes` in the
   contract's order; étage height bands (weak thresholds 2 km and 6 km, with the WMO mid-latitude ranges noted);
   code tables 0513 (C_L), 0515 (C_M), 0509 (C_H), 1600 (h), 2700 (N) with the genera each code names, including
   "either/or" codes (`alternatives: true`) and "not visible" (`/` → null); dataset classes for CCSN (one genus each,
   Ct → contrail), MGCD (merged types → genus sets; mixed → cloud of unknown genus), SWIMCAT (clear / cloud only),
   Montenegro (altitude classes → étages; codes), Almería and SWIM masks, Eye2Sky (ceilometer targets), B0268
   (P041 fields).
2. `stratia/labels/ontology.py`: loader with validation (every genus in one étage, contiguous height bands,
   complete code tables, every dataset class on known genera or explicitly excluded, `genus_classes` complete),
   lookups (étage of a genus or of a height, genera of a code, h ranges, oktas).
3. Tests (`tests/test_ontology.py`, 4): the shipped file is consistent; heights and codes; dataset classes; a
   broken file is rejected with named problems.

## Verification
- `python -m pytest -q tests/test_ontology.py`: 4 passed; the validator reports no problem on the shipped file.
- Review against the WMO sources: genera, étages and the "extends into" notes follow the International Cloud
  Atlas (genus definitions and the étage table); the code meanings follow WMO-No. 306 Vol. I.1 code tables 0513,
  0515, 0509, 1600 and 2700. The wording was written from those tables as known to the author of this phase and
  must be checked line by line against the current edition before the paper's appendix is final (open item in
  the data card).
- MGCD's merged categories (altocumulus with cirrocumulus, cirrus with cirrostratus, stratocumulus with stratus and
  altostratus, cumulonimbus with nimbostratus) follow the dataset's own category definitions and are stored as
  genus sets with `alternatives: true`, never as one genus.

## Exit criteria
- [x] `configs/ontology.yaml` reviewed against the WMO Cloud Atlas and the WMO-No. 306 code tables as described
  above; validated by tests.

## Fit & data-risk notes
- **Nimbostratus is a middle-étage genus** by WMO definition although its base is usually low; the ontology keeps
  the WMO étage and records the extension. The ceilometer's weak height thresholds (P036) will often call an Ns base
  "low"; evaluation at étage level must say which definition it uses (genus-derived or height-derived).
- **Merged classes become sets, not noise:** a set-valued target (MGCD "altocumulus" = {Ac, Cc}) trains the model
  to put mass on either member, which is the honest reading of the label; forcing one genus would import a 50 %
  error rate into the training signal.
- **SWIMCAT contributes no genus information** (appearance categories of 125 px patches): clear / cloud only, as
  the plan requires; it stays useful for cloud presence and as a source for the shortcut and leakage protocol.
- The `h` code of Montenegro maps to height bands (code 9 = "2,500 m or more, or no cloud" is ambiguous and is kept
  as its own bin; P035 handles it).

## Deviations from plan & why
- None in scope; the file carries more than the plan's three mappings (code tables, extra classes, mask classes,
  B0268 fields) so that every later phase reads labels from one place.

## Next phase
P034: Dataset → ontology mapping.
