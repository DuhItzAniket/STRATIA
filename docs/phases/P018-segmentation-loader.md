# P018 — Segmentation datasets loader

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Index the segmentation datasets and map every mask to STRATIA's unified labels (sky parsing: invalid / sky / cloud / sun-glare; cloud layer: low / mid / high; 255 = ignore), verifying each source encoding from the data.

## Inputs / dependencies
P007 registry; SWIMSEG, SWINSEG, SWINySEG, SHWIMSEG, Almería. LenghuSky-8 labels are not downloaded (optional, P015).

## Work log
1. Inspected each dataset's masks against its images before mapping (class shares, blue/red ratio, brightness):
   - SWIMSEG and SWINySEG: 0 = sky, 255 = cloud;
   - **SWINSEG masks are JPEG**, so compression noise produces intermediate values → thresholded at 128;
   - SHWIMSEG: one mask per HDR set shared by three LDR exposures (the "high" exposure is near-white);
   - Almería: 0 camera mask, 1 sky, 2/3/4 low/mid/high cloud; `validation.csv` defines the official validation subset (its lines end with a stray comma, handled).
2. `stratia/data/segmentation.py`: `build_index` (one row per image/mask pair with dataset, official split, camera) and `load_masks` (unified sky-parse and layer maps).
3. `scripts/segmentation_qa.py`: existence and size checks for all pairs, class shares, a physics check of the encoding (clear sky has a higher blue/red ratio than cloud in daytime data), overlay sheets → `docs/data/segmentation_qa.md`, `docs/data/figures/qa_seg_*.jpg`; index saved to `data/segmentation_index.parquet` (ignored).

## Verification
| Dataset | Pairs | Missing / size mismatch | Sky / cloud / invalid share | B/R sky vs cloud | Encoding check |
|---|---|---|---|---|---|
| SWIMSEG | 1,013 | 0 / 0 | 40.2 / 59.8 / 0% | 2.38 > 1.25 | pass |
| SWINSEG (night) | 115 | 0 / 0 | 58.1 / 41.9 / 0% | (0.89, 0.70) | colour prior not applicable at night |
| SWINySEG | 6,768 | 0 / 0 | 45.6 / 54.4 / 0% | 3.52 > 1.29 | pass |
| SHWIMSEG | 156 | 0 / 0 | 48.1 / 51.9 / 0% | 1.24 > 1.00 | pass |
| Almería | 818 (train 616, val 154, test 48) | 0 / 0 | 41.2 / 36.4 / 22.4% | 1.18 > 0.90 | pass |

Visual check of the Almería sheet: clear sky blue, camera mask grey outside the fisheye circle, low cumulus near the horizon and overcast red (low layer), an altocumulus field yellow (mid), a cirrus veil around the Sun cyan (high). Unit tests cover the Almería mapping, the JPEG threshold and unknown datasets.

## Exit criteria
- [x] Visual QA sheet per dataset; label IDs verified.

## Fit & data-risk notes
- No dataset labels **sun/glare** or **obstructions** besides Almería's camera mask: those classes need the B0268 dense labels (P041) and synthetic glare (P048).
- The SWIM family comes from one Singapore camera family; it trains sky/cloud separation but says little about other cameras (cross-camera evaluation uses Almería's 4-camera test set).
- SWINSEG's night masks cannot be checked with a colour prior; they are used as given.

## Deviations from plan & why
LenghuSky-8 is not included (not downloaded; optional).

## Next phase
P019 — B0268 ingest.
