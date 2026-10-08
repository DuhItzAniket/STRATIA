# Montenegro soft labels and human ceiling (P035)

11,191 annotations by 9 raters on 2,522 images (P016), translated through `configs/ontology.yaml` into the labels STRATIA predicts. alpha = Krippendorff's alpha (1 = perfect, 0 = chance); pairwise = share of rater pairs that agree; leave-one-out = a rater against the majority (or median) of the other raters, the bar a model is compared with (criterion C2).

| Quantity | Level | Units | alpha | Pairwise | Leave-one-out | Note |
|---|---|---|---|---|---|---|
| raw altitude_class | nominal | 2,270 | 0.340 | 0.562 | 0.712 |  |
| raw CL | nominal | 2,270 | 0.372 | 0.495 | 0.668 |  |
| raw CM | nominal | 2,270 | 0.249 | 0.470 | 0.655 |  |
| raw CH | nominal | 2,270 | 0.264 | 0.533 | 0.668 |  |
| raw N (oktas 0-8) | interval | 2,260 | 0.914 | 0.837 | 0.873 | pairwise and leave-one-out within ±1 okta |
| raw N (oktas 0-8), ordinal | ordinal | 2,260 | 0.904 | 0.837 | 0.873 |  |
| raw h (height code 0-9) | ordinal | 2,207 | 0.361 | 0.398 | 0.373 | exact code; '/' excluded |
| raw h (height code 0-9), ±1 band | ordinal | 2,207 | 0.361 | 0.740 | 0.761 |  |
| étage set | nominal | 2,270 | 0.539 | 0.755 | 0.852 | exact set from the altitude class |
| étage low present | nominal | 2,270 | 0.620 | 0.828 | 0.884 |  |
| étage mid present | nominal | 2,270 | 0.199 | 0.886 | 0.927 |  |
| étage high present | nominal | 2,270 | 0.393 | 0.870 | 0.912 |  |
| genus set | nominal | 2,269 | 0.279 | 0.330 | 0.583 | exact set from C_L / C_M / C_H |
| genus Ci present | nominal | 2,269 | 0.549 | 0.850 | 0.907 |  |
| genus Cc present | nominal | 2,269 | 0.065 | 0.975 | 0.986 |  |
| genus Cs present | nominal | 2,269 | 0.135 | 0.929 | 0.959 |  |
| genus Ac present | nominal | 2,269 | 0.372 | 0.763 | 0.818 |  |
| genus As present | nominal | 2,269 | 0.286 | 0.864 | 0.895 |  |
| genus Ns present | nominal | 2,269 | 0.310 | 0.881 | 0.908 |  |
| genus Sc present | nominal | 2,269 | 0.486 | 0.753 | 0.818 |  |
| genus St present | nominal | 2,269 | 0.347 | 0.916 | 0.939 |  |
| genus Cu present | nominal | 2,269 | 0.423 | 0.718 | 0.799 |  |
| genus Cb present | nominal | 2,269 | 0.094 | 0.935 | 0.958 |  |
| cloud present (N > 0) | nominal | 2,270 | 0.674 | 0.923 | 0.953 |  |

## Raters against the others (étage set and total cover)

| Rater | Units (étage) | Étage agreement | Units (N) | N within ±1 okta |
|---|---|---|---|---|
| 68 | 1,210 | 0.893 | 1,446 | 0.856 |
| 69 | 1,991 | 0.882 | 2,176 | 0.894 |
| 70 | 1,963 | 0.895 | 2,178 | 0.938 |
| 71 | 1,963 | 0.813 | 2,155 | 0.770 |
| 72 | 70 | 0.714 | 70 | 0.871 |
| 73 | 2 | 1.000 | 2 | 1.000 |
| 74 | 173 | 0.977 | 178 | 0.944 |
| 75 | 183 | 0.934 | 189 | 0.915 |
| 76 | 2,010 | 0.779 | 2,151 | 0.891 |
| all | 9,565 | 0.852 | 10,545 | 0.873 |

## Decisions

- **Human ceiling for criterion C2:** a rater agrees with the majority of the other raters on the étage set in 85.2% of images (alpha 0.54) and on total cloud cover within ±1 okta in 87.3% (interval alpha 0.91). A model's agreement with the rater majority is compared with these numbers; 90 % of them is the C2 bar.
- **Genus from the codes is the weakest signal** (exact genus set alpha 0.28); per-genus presence is used as a soft target (share of raters), never as a hard label.
- **Height code h** (ordinal alpha 0.36; ±1 band leave-one-out 76.1%) stays a local soft evaluation target; the ceilometer is the CBH reference (P031, P036).
- Targets per image: `data/montenegro_targets.parquet` (étage and genus shares, oktas and height-band distributions, obscured share, cloud share); training uses them as distributions (P030: no resampling of soft labels).
