# Licences (P097)

What each part of the release may be used for. The code is Apache-2.0 (`LICENSE`). Model weights trained on this
collection inherit the most restrictive licence of their training sources (data card §7): the SWIM family's
CC BY-NC 4.0 and MGCD's research-use terms make every trained head **non-commercial, research use** until those two
sources are either cleared or dropped from training.

| Part | Licence | Source of the statement |
|---|---|---|
| STRATIA code (`stratia/`, `scripts/`, `tests/`), configs, docs | Apache-2.0 | `LICENSE` |
| CloudScope desktop application (separate repository) | see its `LICENSE` | CloudScope repository |
| Model weights `stratia-1.0.0-demo`, `stratia-1.0.1-demo` (ONNX + card) | research use, non-commercial (inherited; see below) | data card §7 |
| Split manifests, lock file (`test_lock.json`), result files (`results/`) | CC BY 4.0 | this release |
| B0268 captures and their labels (`DATA/B0268`, not in the repository) | owner's data; CC BY 4.0 on release | owner |

## Training and evaluation sources

| Source | Licence | Verified against | Used for |
|---|---|---|---|
| CCSN | CC0 1.0 | Harvard Dataverse API (doi:10.7910/DVN/CADDPD), 2026-10-04 | genus / étage training and test |
| MGCD | distributed by Tianjin Normal University; terms to be confirmed (research use) | repository README | sky-type / étage training and test |
| SWIMCAT, SWIMSEG, SWINSEG, SWINySEG, SHWIMSEG | CC BY-NC 4.0 | local `license.html` (SWIMSEG); same terms assumed for the siblings | sky types, sky/cloud masks |
| Almería (DLR all-sky imager segmentation) | CC BY 4.0 | Zenodo record 14639170, 2026-10-04 | sky/cloud/layer masks |
| Montenegro multi-annotator set | CC BY 4.0 | Zenodo record 21787669 and local `LICENSE.txt` | human ceiling, étage / oktas |
| Eye2Sky images and ceilometers | CDLA-Sharing 1.0 (record); images also stated CC BY-SA 3.0 | Zenodo record 12804613, 2026-10-04 | cloud-base height, étage rows |
| B0268 (ours) | owner's data | — | adaptation and the locked day |
| DINOv3 ViT-S/16 weights (frozen backbone) | DINOv3 licence (Meta) | Hugging Face model card | backbone |

Not used for training: DeepSky, WEBCAM, HBMCD (not obtained; `configs/datasets.yaml`), LenghuSky-8 (labels only).

## What this means for a public release

- Code, manifests, lock, results and the paper tables: release freely (Apache-2.0 / CC BY 4.0).
- Weights: release under a research-only, non-commercial notice that names the SWIM family and MGCD as the reason,
  and the DINOv3 licence for the backbone. A commercial variant would retrain the dense head without the SWIM
  masks and the global heads without MGCD.
- Images: never redistributed; the release points at each source's record and ships only ids and labels.
