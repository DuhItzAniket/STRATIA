# Label mapping report (P034)

Every native label mapped through `configs/ontology.yaml` (P033) to a set-valued STRATIA label or excluded for a stated reason; output `data/labels.parquet` (one row per manifest sample).

## Native classes and what they map to

| Dataset | Native class | Maps to | Étages | Note |
|---|---|---|---|---|
| ccsn | Ac | Ac | mid |  |
| ccsn | As | As | mid |  |
| ccsn | Cb | Cb | low |  |
| ccsn | Cc | Cc | high |  |
| ccsn | Ci | Ci | high |  |
| ccsn | Cs | Cs | high |  |
| ccsn | Ct | contrail | none |  |
| ccsn | Cu | Cu | low |  |
| ccsn | Ns | Ns | mid |  |
| ccsn | Sc | Sc | low |  |
| ccsn | St | St | low |  |
| mgcd | cumulus | Cu | low |  |
| mgcd | altocumulus | Ac or Cc | high, mid | altocumulus and cirrocumulus |
| mgcd | cirrus | Ci or Cs | high | cirrus and cirrostratus |
| mgcd | clearsky | clear | none |  |
| mgcd | stratocumulus | Sc or St or As | low, mid | stratocumulus, stratus and altostratus |
| mgcd | cumulonimbus | Cb or Ns | low, mid | cumulonimbus and nimbostratus |
| mgcd | mixed | cloud, genus unknown |  | several cloud types in one frame; genus excluded |
| swimcat | A-sky | clear | none |  |
| swimcat | B-pattern | cloud, genus unknown |  |  |
| swimcat | C-thick-dark | cloud, genus unknown |  |  |
| swimcat | D-thick-white | cloud, genus unknown |  |  |
| swimcat | E-veil | cloud, genus unknown |  |  |
| montenegro | Clear | clear | none | plus genera from the majority CL/CM/CH codes |
| montenegro | Low clouds | étage low | low | plus genera from the majority CL/CM/CH codes |
| montenegro | Middle clouds | étage mid | mid | plus genera from the majority CL/CM/CH codes |
| montenegro | High clouds | étage high | high | plus genera from the majority CL/CM/CH codes |
| montenegro | Clouds of vertical development | étage low (Cu or Cb) | low | plus genera from the majority CL/CM/CH codes |

## Coverage per dataset

| Dataset | Rows | Genus set | of which alternatives | Étage set | Clear | Contrail | Cloud, genus unknown | No label | Conflicts merged |
|---|---|---|---|---|---|---|---|---|---|
| almeria | 818 | 0 | 0 | 818 | 83 | 0 | 735 | 0 | 0 |
| b0268 | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 25 | 0 |
| ccsn | 2,543 | 2,543 | 200 | 2,345 | 0 | 199 | 0 | 0 | 200 |
| eye2sky | 29,288 | 0 | 0 | 0 | 0 | 0 | 0 | 29,288 | 0 |
| mgcd | 8,000 | 6,980 | 4,204 | 4,099 | 1,338 | 0 | 1,020 | 0 | 0 |
| montenegro | 2,522 | 2,363 | 286 | 2,522 | 499 | 0 | 159 | 0 | 0 |
| shwimseg | 156 | 0 | 0 | 0 | 0 | 0 | 156 | 0 | 0 |
| swimcat | 784 | 224 | 0 | 224 | 224 | 0 | 560 | 0 | 0 |
| swimseg | 1,013 | 0 | 0 | 1 | 1 | 0 | 1,012 | 0 | 0 |
| swinseg | 115 | 0 | 0 | 0 | 0 | 0 | 115 | 0 | 0 |
| swinyseg | 6,768 | 0 | 0 | 16 | 16 | 0 | 6,752 | 0 | 0 |

## Exclusion reasons

| Reason | Rows |
|---|---|
| eye2sky: no image-level label | 29,288 |
| swinyseg: mask labels only (no genus) | 6,768 |
| mgcd class 'mixed': genus not named by the dataset | 1,020 |
| swimseg: mask labels only (no genus) | 1,013 |
| almeria: mask labels only (no genus) | 818 |
| swimcat class 'C-thick-dark': genus not named by the dataset | 251 |
| shwimseg: mask labels only (no genus) | 156 |
| swimcat class 'D-thick-white': genus not named by the dataset | 135 |
| swinseg: mask labels only (no genus) | 115 |
| montenegro: no genus named by the majority codes | 92 |
| swimcat class 'B-pattern': genus not named by the dataset | 89 |
| swimcat class 'E-veil': genus not named by the dataset | 85 |
| montenegro: no genus named by the majority codes (not visible) | 67 |
| b0268: no image-level label | 25 |

## Decisions

- **Merged classes become sets of alternatives, never one genus**: MGCD 'altocumulus' is {Ac, Cc}, 'cirrus' {Ci, Cs}, 'stratocumulus' {Sc, St, As}, 'cumulonimbus' {Cb, Ns}; the loss will reward mass on any member (P065). Rejected: picking the first-named genus, which would plant a known error rate in the training signal.
- **'Mixed' and the SWIMCAT patch categories carry cloud presence only** (genus unknown), so they train the cloud / clear and oktas-type heads and not the genus head. Rejected: dropping them (they are 1,020 and 560 images of real sky).
- **Montenegro genera come from the majority C_L / C_M / C_H codes, étages from the altitude classes**; a code '/' (part of the sky not visible) leaves that part unknown rather than empty. The rater distributions themselves stay in the manifest for soft targets (P035).
- **Exact and copy conflicts (P026) are merged**: the members take the union of their genus sets as alternatives and are flagged `label_conflict`; the split generator keeps flagged pictures out of every test split (P038). Same-scene disagreements are not merged: the pictures differ.
- **The segmentation datasets contribute masks, cloud presence and (Almería) an étage set from the layer masks**; no genus. Eye2Sky and B0268 carry no image-level class until P036 and P041.
- The manifest's own `genus_set` / `etage_set` columns stay empty; labels live in `data/labels.parquet` and are joined on `sample_id`, so the manifest (and every table aligned to it) never changes when a mapping rule does.
