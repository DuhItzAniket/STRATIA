**Cloud-base height on the locked ceilometer-paired test pairs (2,965 cloudy frames, 25 days per site): MAE, skill over the training climatology, share within 500 m, and 80% interval coverage with the validation-fitted calibration.**

| Model | MAE (m) [CI] | Skill [CI] | ±500 m | 80% cov. |
|---|---|---|---|---|
| STRATIA (both sites in training) | 872 [679, 1055] | 0.566 [0.48, 0.66] | 0.546 | 0.720 |
|   OLDLR (CDLRA) | 933 [744, 1127] | 0.536 [0.44, 0.63] | 0.531 | 0.698 |
|   WESTE (CDLRB) | 813 [601, 1013] | 0.596 [0.50, 0.70] | 0.561 | 0.742 |
| STRATIA, geometry zeroed | 875 [680, 1050] | 0.565 [0.48, 0.66] | 0.546 | 0.764 |
| STRATIA trained on OLDLR only, tested on WESTE | 1134 [835, 1456] | 0.436 [0.28, 0.58] | 0.496 | 0.605 |
| Ridge on frozen features (P058) | 1096 [852, 1365] | 0.455 | 0.470 | – |
| Training climatology | 2012 | – | – | – |
