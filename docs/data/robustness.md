# Corruption robustness (P088)

Stage H candidates on corrupted validation images, backbone run live: 1200 recognition rows (CCSN, MGCD, Montenegro, Almería), 400 segmentation images, 100 clear-sky Eye2Sky frames, 800 CBH pairs. Each cell: metric under the corruption (change vs clean in brackets).

| Corruption | Mean étage F1 | Genus F1 (CCSN, MGCD) | Sky mIoU | Clear-sky cloud fraction | CBH MAE (m) | CBH skill |
|---|---|---|---|---|---|---|
| clean | 0.838 | 0.703 | 0.901 | 0.292 | 606 | +0.446 |
| exposure x0.6 | 0.830 (-0.008) | 0.708 (+0.004) | 0.899 (-0.002) | 0.306 (+0.014) | 611 (+6) | +0.441 (-0.005) |
| exposure x1.5 | 0.822 (-0.016) | 0.694 (-0.010) | 0.900 (-0.001) | 0.312 (+0.019) | 602 (-4) | +0.450 (+0.004) |
| haze 40% | 0.817 (-0.022) | 0.713 (+0.010) | 0.893 (-0.008) | 0.308 (+0.016) | 664 (+58) | +0.393 (-0.053) |
| dirty lens | 0.804 (-0.034) | 0.635 (-0.068) | 0.893 (-0.008) | 0.308 (+0.016) | 608 (+2) | +0.445 (-0.002) |
| sun glare | 0.760 (-0.078) | 0.656 (-0.048) | 0.816 (-0.085) | 0.467 (+0.174) | 746 (+140) | +0.318 (-0.128) |
| jpeg q30 | 0.812 (-0.026) | 0.729 (+0.025) | 0.894 (-0.007) | 0.207 (-0.086) | 779 (+174) | +0.288 (-0.159) |
| sensor noise | 0.741 (-0.098) | 0.678 (-0.026) | 0.848 (-0.053) | 0.258 (-0.034) | 1014 (+408) | +0.073 (-0.373) |
| blur 2px | 0.806 (-0.032) | 0.680 (-0.024) | 0.895 (-0.006) | 0.246 (-0.046) | 742 (+137) | +0.321 (-0.125) |

Rows in `data/robustness.parquet`. Corruptions follow the P048 photometric functions where one exists.
