# Ceilometer targets (P036)

Targets on a 30 s grid for every day of each ceilometer, from the clean records within ±30 s (P031): no cloud overhead, or the étage of the median lowest base (weak thresholds 2 / 6 km; WMO 2 / 7 km for the ablation), with the cloudy share as confidence, the base spread inside the window, the mean layer count and the second layer where there is one. Table `data/ceilometer_targets.parquet`.

## CDLRA

| Subset | Grid points | With a label | Mixed windows | none | low | mid | high | mixed | WMO: low / mid / high | >= 2 layers | Base spread (median) | Confidence (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 339,840 | 94.1% | 5.7% | 31.2% | 35.0% | 20.1% | 11.9% | 1.7% | 52.2% / 34.1% / 13.7% | 16.5% | 30 m | 1.00 |
| daytime | 220,153 | 94.0% | 6.9% | 30.0% | 38.2% | 20.1% | 9.6% | 2.1% | 56.3% / 33.0% / 10.8% | 15.7% | 33 m | 1.00 |

## CDLRB

| Subset | Grid points | With a label | Mixed windows | none | low | mid | high | mixed | WMO: low / mid / high | >= 2 layers | Base spread (median) | Confidence (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 339,840 | 96.9% | 5.7% | 30.2% | 36.5% | 20.1% | 11.5% | 1.8% | 53.6% / 32.9% / 13.5% | 16.5% | 30 m | 1.00 |
| daytime | 220,360 | 97.1% | 6.7% | 29.0% | 39.5% | 19.9% | 9.5% | 2.1% | 57.3% / 31.4% / 11.3% | 15.9% | 35 m | 1.00 |

## Decisions

- **Label = the ceilometer's view in the frame's ±30 s window:** 'none' when no clean record sees cloud, the étage of the median lowest base when at least half of them do, 'mixed' otherwise; the share behind the label is the confidence and goes into the loss as a weight (P030: soft labels are distributions). Windows without a clean record (rain, window particles, optics, error bits, or no data) give no label and become the camera's 'obscured / unlabelled' cases.
- **Multi-layer skies are kept:** the second layer's median height and the mean layer count travel with the label, so a model's lowest-base error can be analysed by layer count (P087) and nothing is discarded for being ambiguous.
- **Two étage thresholds are stored** (weak 2 / 6 km from the ontology; WMO 2 / 7 km): the ablation planned in P036 is a column switch, not a rebuild.
- Pairing with frames (P047) joins a frame's time to the nearest grid point (15 s away at most) or recomputes the window at the exact frame time with `targets_at`; both use the same rule.
