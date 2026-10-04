# Data acquisition status and owner actions (P015)

Checked 2026-10-04 through official APIs and repository pages (no data downloaded).

## Licences verified

| Dataset | Licence | Evidence |
|---|---|---|
| CCSN | CC0 1.0 | Harvard Dataverse API (doi:10.7910/DVN/CADDPD) |
| Almería ASI segmentation | CC BY 4.0 | Zenodo API, record 14639170 |
| Montenegro multi-annotator | CC BY 4.0 | Zenodo API, record 21787669 (and local `LICENSE.txt`) |
| Eye2Sky | CDLA-Sharing 1.0 | Zenodo API, record 12804613 |
| SWIMCAT, SWIMSEG | CC BY-NC 4.0 | Local `license.html` |
| LenghuSky-8 | Apache-2.0 | Hugging Face dataset card (ungated) |
| MGCD / TJNU GCD | not stated | GitHub repository has no licence; terms by agreement — research use only, no redistribution |

## What could not be downloaded, and why

| Dataset | Finding | Action (owner) | Priority |
|---|---|---|---|
| **Eye2Sky at the ceilometer sites** | 0 pairable images on disk (P012) | Download **OLDLR** and **WESTE** (optionally OLUOL, OLWIN) images for April–July 2022 from the Eye2Sky data server (https://eye2sky.de/data/); ≈ 0.3 GB per station-day | **Critical** (cloud-base height, criterion C3) |
| **DeepSky** | Zenodo record 8208505 contains only the paper; the GitHub repository (dimkastan/DeepSky-classification-dataset) says "Work in progress — for any request please contact" the authors | Email the contact listed in that README to request the images, citing the IMVIP 2023 paper | High (temporal-split benchmark for cross-dataset recognition) |
| **WEBCAM** | The repository says the data is "available for public download in association with the manuscript"; no link in the repository or the authors' notebook | Open the IEEE Access paper (doi:10.1109/ACCESS.2025.3634057) in a browser, find its data-availability statement, or contact the corresponding author | High (a third camera type: webcams) |
| **TJNU GCD** | Distribution by signed agreement | Fill in the agreement on github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset | Medium |
| **HBMCD (GBCID)** | Only on Baidu Pan (link and extraction code in the repository's `Download_Link` file) | Download with a Baidu account if possible | Optional |
| **LenghuSky-8 labels** | Public (GitHub ≈ 25 MB; Hugging Face Apache-2.0) | Approve download (≈ 25 MB) | Optional |

## Effect on the plan if nothing more arrives

Cross-dataset recognition (criterion C1) still has four sources with different cameras: CCSN (consumer photos), MGCD (fisheye), Montenegro (low-cost fixed camera) and the owner's B0268. Cloud-base height (criterion C3) **cannot be trained or evaluated** without Eye2Sky images at OLDLR/WESTE; that download is the project's single most important pending item.
