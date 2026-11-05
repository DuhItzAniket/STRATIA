**Ablations of the global heads on validation data (one seed; the reference's three-seed spread is below 0.01): mean étage macro-F1 over the sources and mean genus macro-F1; the LODO rows give the held-out fold's étage F1.**

| Group | Variant | Étage F1 | Genus F1 |
|---|---|---|---|
| reference | Gate G4 heads (geometry dropout 0.3, soft and set-valued targets, hier | 0.817 | 0.719 |
| geometry (SkyRay) | no geometry: ray map and Sun always dropped (p_drop 1.0) | 0.809 | 0.720 |
| geometry (SkyRay) | geometry always given (p_drop 0.0) | 0.829 | 0.709 |
| targets | hard targets: first listed genus, majority étages, majority okta | 0.794 | 0.729 |
| loss | no hierarchy loss (lambda_h 0) | 0.813 | 0.712 |
| loss | uncertainty (Kendall) task weighting | 0.814 | 0.716 |
| Eye2Sky étage rows | without the ceilometer étage rows (indomain) | 0.823 | 0.738 |
| Eye2Sky étage rows (LODO) | held-out ccsn, with Eye2Sky étage rows | 0.489 | – |
| Eye2Sky étage rows (LODO) | held-out ccsn, without | 0.486 | – |
| Eye2Sky étage rows (LODO) | held-out mgcd, with Eye2Sky étage rows | 0.749 | – |
| Eye2Sky étage rows (LODO) | held-out mgcd, without | 0.871 | – |
| Eye2Sky étage rows (LODO) | held-out montenegro, with Eye2Sky étage rows | 0.360 | – |
| Eye2Sky étage rows (LODO) | held-out montenegro, without | 0.258 | – |
| Eye2Sky étage rows (LODO) | held-out almeria, with Eye2Sky étage rows | 0.593 | – |
| Eye2Sky étage rows (LODO) | held-out almeria, without | 0.327 | – |
| regularisation (P080) | head dropout 0 | 0.804 | 0.703 |
| regularisation (P080) | head dropout 0.3 | 0.817 | 0.743 |
| regularisation (P080) | weight decay 0 | 0.816 | 0.722 |
| regularisation (P080) | weight decay 0.2 | 0.815 | 0.715 |
| regularisation (P080) | no EMA | 0.820 | 0.711 |
| regularisation (P080) | EMA 0.999 | 0.812 | 0.708 |
