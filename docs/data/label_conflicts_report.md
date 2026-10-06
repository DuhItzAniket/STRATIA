# Label-conflict audit (P026)

Pairs of images that are the same picture (exact duplicates, P024; copies, P025) or the same scene (cosine >= 0.97, P025), compared on the labels each dataset ships with. Nothing is corrected; the policy at the end says what the later phases do with each kind of conflict.

## Pairs compared

| Kind | Pairs | Of which within one dataset | With a class label on both sides | With masks on both sides |
|---|---|---|---|---|
| exact | 390 | 390 | 34 | 356 |
| copy | 4,217 | 1,756 | 131 | 4,086 |
| same_scene | 41,671 | 41,624 | 38,710 | 2,079 |

## Class-label conflicts

A conflict is a pair of same-dataset images with two different native labels. For exact duplicates and copies this is a labelling error or a deliberately ambiguous photograph; for same-scene pairs (frames seconds or minutes apart, re-posted crops) it is the rate at which the label changes while the picture barely does: a floor on label noise.

| Dataset | Kind | Pairs | Conflicts | Rate | Images involved |
|---|---|---|---|---|---|
| ccsn | copy | 129 | 105 | 81.4% | 196 |
| ccsn | exact | 17 | 3 | 17.6% | 6 |
| ccsn | same_scene | 25 | 22 | 88.0% | 44 |
| mgcd | same_scene | 36,532 | 222 | 0.6% | 163 |
| montenegro | same_scene | 2,077 | 595 | 28.6% | 419 |
| swimcat | copy | 2 | 0 | 0.0% | 0 |
| swimcat | exact | 17 | 0 | 0.0% | 0 |
| swimcat | same_scene | 76 | 0 | 0.0% | 0 |

### Which labels conflict

| Dataset | Kind | Labels | Pairs |
|---|---|---|---|
| ccsn | copy | Ns / St | 17 |
| ccsn | copy | Cb / Cu | 10 |
| ccsn | copy | Cc / Cs | 9 |
| ccsn | copy | Ac / Cc | 7 |
| ccsn | copy | Sc / St | 6 |
| ccsn | copy | Cs / Sc | 5 |
| ccsn | copy | As / Cb | 4 |
| ccsn | copy | As / St | 4 |
| ccsn | exact | Ac / As | 2 |
| ccsn | exact | Cc / Cs | 1 |
| ccsn | same_scene | Ns / St | 4 |
| ccsn | same_scene | Sc / St | 4 |
| ccsn | same_scene | Cc / Cs | 3 |
| ccsn | same_scene | Cc / St | 2 |
| ccsn | same_scene | Cs / St | 2 |
| ccsn | same_scene | Ns / Sc | 2 |
| ccsn | same_scene | Ac / Cc | 1 |
| ccsn | same_scene | Ac / Ns | 1 |
| mgcd | same_scene | altocumulus / mixed | 74 |
| mgcd | same_scene | cumulonimbus / cumulus | 42 |
| mgcd | same_scene | altocumulus / stratocumulus | 31 |
| mgcd | same_scene | cumulonimbus / stratocumulus | 30 |
| mgcd | same_scene | cirrus / mixed | 28 |
| mgcd | same_scene | cumulus / mixed | 9 |
| mgcd | same_scene | altocumulus / cirrus | 5 |
| mgcd | same_scene | cirrus / clearsky | 2 |
| montenegro | same_scene | Clear / High clouds | 216 |
| montenegro | same_scene | Clear / Low clouds | 63 |
| montenegro | same_scene | Clear / Clear,Low clouds | 57 |
| montenegro | same_scene | Clear / Clear,High clouds | 53 |
| montenegro | same_scene | High clouds / High clouds,Low clouds | 29 |
| montenegro | same_scene | Clear,Low clouds / Low clouds | 27 |
| montenegro | same_scene | Clear,High clouds / High clouds | 22 |
| montenegro | same_scene | Clear / Middle clouds | 19 |

### Exact duplicates and copies with different labels

| Kind | Dataset | Image i | Label i | Image j | Label j |
|---|---|---|---|---|---|
| exact | ccsn | `CCSN/CCSN_v2/Ac/Ac-N186.jpg` | Ac | `CCSN/CCSN_v2/As/As-N139.jpg` | As |
| exact | ccsn | `CCSN/CCSN_v2/Ac/Ac-N202.jpg` | Ac | `CCSN/CCSN_v2/As/As-N175.jpg` | As |
| exact | ccsn | `CCSN/CCSN_v2/Cc/Cc-N179.jpg` | Cc | `CCSN/CCSN_v2/Cs/Cs-N244.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N006.jpg` | Ac | `CCSN/CCSN_v2/As/As-N014.jpg` | As |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N013.jpg` | Ac | `CCSN/CCSN_v2/Cc/Cc-N029.jpg` | Cc |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N059.jpg` | Ac | `CCSN/CCSN_v2/Cc/Cc-N033.jpg` | Cc |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N154.jpg` | Ac | `CCSN/CCSN_v2/Cc/Cc-N046.jpg` | Cc |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N156.jpg` | Ac | `CCSN/CCSN_v2/Cc/Cc-N052.jpg` | Cc |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N171.jpg` | Ac | `CCSN/CCSN_v2/Cc/Cc-N109.jpg` | Cc |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N181.jpg` | Ac | `CCSN/CCSN_v2/Sc/Sc-N071.jpg` | Sc |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N182.jpg` | Ac | `CCSN/CCSN_v2/As/As-N150.jpg` | As |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N186.jpg` | Ac | `CCSN/CCSN_v2/St/St-N171.jpg` | St |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N186.jpg` | Ac | `CCSN/CCSN_v2/As/As-N141.jpg` | As |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N192.jpg` | Ac | `CCSN/CCSN_v2/Cc/Cc-N133.jpg` | Cc |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N206.jpg` | Ac | `CCSN/CCSN_v2/Cc/Cc-N144.jpg` | Cc |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N208.jpg` | Ac | `CCSN/CCSN_v2/Ns/Ns-N137.jpg` | Ns |
| copy | ccsn | `CCSN/CCSN_v2/Ac/Ac-N216.jpg` | Ac | `CCSN/CCSN_v2/St/St-N192.jpg` | St |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N003.jpg` | As | `CCSN/CCSN_v2/Sc/Sc-N145.jpg` | Sc |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N010.jpg` | As | `CCSN/CCSN_v2/Sc/Sc-N099.jpg` | Sc |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N019.jpg` | As | `CCSN/CCSN_v2/Cb/Cb-N212.jpg` | Cb |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N028.jpg` | As | `CCSN/CCSN_v2/Ci/Ci-N003.jpg` | Ci |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N032.jpg` | As | `CCSN/CCSN_v2/Cb/Cb-N082.jpg` | Cb |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N043.jpg` | As | `CCSN/CCSN_v2/Cs/Cs-N087.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N043.jpg` | As | `CCSN/CCSN_v2/Cs/Cs-N114.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N075.jpg` | As | `CCSN/CCSN_v2/St/St-N113.jpg` | St |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N092.jpg` | As | `CCSN/CCSN_v2/Cs/Cs-N124.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N108.jpg` | As | `CCSN/CCSN_v2/Ns/Ns-N087.jpg` | Ns |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N110.jpg` | As | `CCSN/CCSN_v2/Ci/Ci-N074.jpg` | Ci |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N136.jpg` | As | `CCSN/CCSN_v2/Ns/Ns-N100.jpg` | Ns |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N139.jpg` | As | `CCSN/CCSN_v2/St/St-N171.jpg` | St |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N141.jpg` | As | `CCSN/CCSN_v2/St/St-N171.jpg` | St |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N145.jpg` | As | `CCSN/CCSN_v2/St/St-N172.jpg` | St |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N148.jpg` | As | `CCSN/CCSN_v2/Cb/Cb-N019.jpg` | Cb |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N149.jpg` | As | `CCSN/CCSN_v2/Cu/Cu-N182.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N152.jpg` | As | `CCSN/CCSN_v2/Cu/Cu-N008.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/As/As-N159.jpg` | As | `CCSN/CCSN_v2/Cb/Cb-N007.jpg` | Cb |
| copy | ccsn | `CCSN/CCSN_v2/Cb/Cb-N064.jpg` | Cb | `CCSN/CCSN_v2/Cu/Cu-N049.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/Cb/Cb-N068.jpg` | Cb | `CCSN/CCSN_v2/Cu/Cu-N173.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/Cb/Cb-N113.jpg` | Cb | `CCSN/CCSN_v2/Cu/Cu-N063.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/Cb/Cb-N143.jpg` | Cb | `CCSN/CCSN_v2/Cu/Cu-N023.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/Cb/Cb-N145.jpg` | Cb | `CCSN/CCSN_v2/Cu/Cu-N017.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/Cb/Cb-N202.jpg` | Cb | `CCSN/CCSN_v2/Sc/Sc-N030.jpg` | Sc |
| copy | ccsn | `CCSN/CCSN_v2/Cb/Cb-N202.jpg` | Cb | `CCSN/CCSN_v2/Ci/Ci-N042.jpg` | Ci |
| copy | ccsn | `CCSN/CCSN_v2/Cb/Cb-N213.jpg` | Cb | `CCSN/CCSN_v2/Cu/Cu-N026.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/Cb/Cb-N214.jpg` | Cb | `CCSN/CCSN_v2/Cu/Cu-N007.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/Cb/Cb-N219.jpg` | Cb | `CCSN/CCSN_v2/Cu/Cu-N036.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/Cb/Cb-N222.jpg` | Cb | `CCSN/CCSN_v2/Cu/Cu-N048.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/Cb/Cb-N230.jpg` | Cb | `CCSN/CCSN_v2/Cu/Cu-N058.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/Cc/Cc-N012.jpg` | Cc | `CCSN/CCSN_v2/Cs/Cs-N049.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/Cc/Cc-N023.jpg` | Cc | `CCSN/CCSN_v2/Cs/Cs-N081.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/Cc/Cc-N027.jpg` | Cc | `CCSN/CCSN_v2/St/St-N125.jpg` | St |
| copy | ccsn | `CCSN/CCSN_v2/Cc/Cc-N092.jpg` | Cc | `CCSN/CCSN_v2/Ci/Ci-N059.jpg` | Ci |
| copy | ccsn | `CCSN/CCSN_v2/Cc/Cc-N092.jpg` | Cc | `CCSN/CCSN_v2/Cs/Cs-N136.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/Cc/Cc-N122.jpg` | Cc | `CCSN/CCSN_v2/Cs/Cs-N167.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/Cc/Cc-N132.jpg` | Cc | `CCSN/CCSN_v2/Ns/Ns-N111.jpg` | Ns |
| copy | ccsn | `CCSN/CCSN_v2/Cc/Cc-N165.jpg` | Cc | `CCSN/CCSN_v2/Cs/Cs-N215.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/Cc/Cc-N174.jpg` | Cc | `CCSN/CCSN_v2/Cs/Cs-N228.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/Cc/Cc-N176.jpg` | Cc | `CCSN/CCSN_v2/Cs/Cs-N234.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/Cc/Cc-N180.jpg` | Cc | `CCSN/CCSN_v2/Cs/Cs-N241.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/Cc/Cc-N189.jpg` | Cc | `CCSN/CCSN_v2/Cs/Cs-N242.jpg` | Cs |
| copy | ccsn | `CCSN/CCSN_v2/Ci/Ci-N005.jpg` | Ci | `CCSN/CCSN_v2/Cu/Cu-N060.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/Ci/Ci-N013.jpg` | Ci | `CCSN/CCSN_v2/Cu/Cu-N164.jpg` | Cu |
| copy | ccsn | `CCSN/CCSN_v2/Ci/Ci-N024.jpg` | Ci | `CCSN/CCSN_v2/Ct/Ct-N128.jpg` | Ct |

45 further copy conflicts are in `data/label_conflicts.parquet`.

## Pairs that cross a published split

Same-dataset pairs whose two images sit in different official splits (train/val/test as released). Any such pair is leakage in the published protocol.

| Dataset | Kind | Pairs crossing the official split |
|---|---|---|
| almeria | same_scene | 193 |
| mgcd | same_scene | 701 |

## Mask agreement between copies

For exact duplicates and copies of segmentation images, the mask of image i is flipped or rotated the same way as its picture and compared with the mask of image j, pixel by pixel. A pair whose agreement is below 0.95 is a mask conflict.

| Datasets | Kind | Pairs | Median agreement | 10th percentile | Minimum | Median cloud IoU | Below threshold |
|---|---|---|---|---|---|---|---|
| almeria | exact | 1 | 0.699 | 0.699 | 0.699 | 0.611 | 1 |
| shwimseg | copy | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| shwimseg | exact | 3 | 0.960 | 0.960 | 0.960 | 0.881 | 0 |
| swimseg | exact | 26 | 0.949 | 0.889 | 0.764 | 0.893 | 13 |
| swimseg / swinyseg | copy | 2,213 | 0.997 | 0.991 | 0.761 | 0.993 | 66 |
| swinseg / swinyseg | copy | 248 | 0.995 | 0.988 | 0.873 | 0.988 | 5 |
| swinyseg | copy | 1,620 | 0.998 | 0.980 | 0.762 | 0.995 | 51 |
| swinyseg | exact | 326 | 1.000 | 0.997 | 0.764 | 1.000 | 20 |

## Rater agreement (where several raters labelled one image)

Majority share = the largest fraction of raters giving the same answer for an image.

| Dataset | Quantity | Images | Median majority share | Unanimous | No majority (<= 50 %) |
|---|---|---|---|---|---|
| montenegro | oktas | 2,522 | 0.75 | 29.5% | 25.3% |
| montenegro | h | 2,522 | 0.60 | 13.4% | 39.3% |
| montenegro | cl | 2,522 | 0.75 | 29.0% | 27.0% |
| montenegro | cm | 2,522 | 0.67 | 31.4% | 36.9% |
| montenegro | ch | 2,522 | 0.75 | 30.5% | 21.2% |

## Policy

1. **Exact duplicates and copies with different class labels** are counted once in the data card and are excluded from every test split. Their training label is decided by the harmonisation phases (P033, P034): where the ontology holds a set (genus_set), the set is the union of the conflicting labels; where it holds one value and the conflicting labels disagree at that level, the sample carries no label for that task. No label is edited by hand.
2. **Same-scene disagreements** are kept as they are: the images differ, and a sky can change between frames. The disagreement rate is reported in the data card as the label-consistency floor of each dataset, and the group-level splits (P038) keep such pairs on one side.
3. **Mask conflicts between copies** are listed in `data/mask_agreement.parquet`; the SWIM family is one source for the leave-one-dataset-out protocol (P025), so which mask is "right" does not change a split; conflicting pairs are excluded from any test split and go to the segmentation harmonisation (P034) for review.
4. **Rater disagreement** stays in the manifest as a distribution; training may use it as a soft label and evaluation reports agreement with the majority (P034 decides). Nothing is collapsed here.

## Contact sheets

- `docs/data/figures/label_conflict_exact_ccsn.jpg`
- `docs/data/figures/label_conflict_copy_ccsn.jpg`
- `docs/data/figures/label_conflict_same_scene_ccsn.jpg`
- `docs/data/figures/label_conflict_same_scene_mgcd.jpg`
- `docs/data/figures/label_conflict_same_scene_montenegro.jpg`
- `docs/data/figures/mask_conflicts_worst.jpg`
- `docs/data/figures/mask_agreement_typical.jpg`
