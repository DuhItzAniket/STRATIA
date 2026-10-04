"""Sky/cloud and cloud-layer segmentation datasets mapped to STRATIA's label scheme (P018).

Unified labels (stratia-contract v1): sky parsing {0 invalid/obstruction, 1 sky, 2 cloud, 3 sun/glare},
cloud layer {0 low, 1 mid, 2 high}; 255 = ignore (unlabelled, or non-cloud pixels in the layer map).

Source encodings, verified on the data in P018 (clear sky is bluer than cloud: higher blue/red ratio):
* SWIMSEG, SWINySEG: PNG masks, 0 = sky, 255 = cloud.
* SWINSEG (night): JPEG masks (compression noise) -> thresholded at 128; 0 = sky, 255 = cloud.
* SHWIMSEG: one PNG mask per HDR set, shared by the low/med/high LDR exposures; 255 = cloud.
* Almería (DLR): PNG 0 camera mask, 1 sky, 2 low, 3 mid, 4 high cloud.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IGNORE = 255
SKY_INVALID, SKY, CLOUD, SUN_GLARE = 0, 1, 2, 3
LAYER_LOW, LAYER_MID, LAYER_HIGH = 0, 1, 2
SWIM_FAMILY = {"swimseg": ("images", "GTmaps", "{stem}_GT.png"), "swinseg": ("images", "GTmaps", "{stem}_GT.jpg"),
               "swinyseg": ("images", "GTmaps", "{stem}.png")}


def _rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def build_index(data_root: str | Path) -> pd.DataFrame:
    """One row per (image, mask) pair with dataset, official split and camera where known."""
    root = Path(data_root)
    rows = []
    for name, (img_dir, gt_dir, pattern) in SWIM_FAMILY.items():
        base = root / "SWIMCAT" / name
        for img in sorted((base / img_dir).glob("*")):
            mask = base / gt_dir / pattern.format(stem=img.stem)
            rows.append({"dataset": name, "image_file": _rel(img, root), "mask_file": _rel(mask, root),
                         "split": "none", "camera": "WSI-Singapore", "has_layers": False})
    sh = root / "SWIMCAT" / "shwimseg"
    for img in sorted((sh / "LDR-cropped").glob("*.jpg")):
        exposure, set_id = img.stem.split("-", 1)                      # e.g. med-set01
        rows.append({"dataset": "shwimseg", "image_file": _rel(img, root),
                     "mask_file": _rel(sh / "GT-masks" / f"GT-{set_id}.png", root), "split": "none",
                     "camera": f"WAHRSIS-{exposure}", "has_layers": False})
    al = root / "Almeria All Sky"
    kontas = al / "kontas_2017" / "kontas_2017"
    val_list = kontas / "validation.csv"
    val = set()
    if val_list.exists():   # one name per line with a trailing comma; header "fileNames"
        val = {ln.strip().strip(",") for ln in val_list.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()}
    for img in sorted((kontas / "images").glob("*.jpg")):
        rows.append({"dataset": "almeria", "image_file": _rel(img, root),
                     "mask_file": _rel(kontas / "seg_masks" / f"{img.stem}.png", root),
                     "split": "val" if img.stem in val else "train", "camera": "Kontas", "has_layers": True})
    test = al / "test_set" / "test_set"
    for img in sorted((test / "images").glob("*/*.jpg")):
        cam = img.parent.name.removeprefix("Cloud_Cam_")
        rows.append({"dataset": "almeria", "image_file": _rel(img, root),
                     "mask_file": _rel(test / "seg_masks" / img.parent.name / f"{img.stem}.png", root),
                     "split": "test", "camera": cam, "has_layers": True})
    return pd.DataFrame(rows)


def load_masks(dataset: str, mask_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(sky_parse, layer) uint8 maps in the unified scheme."""
    m = np.array(Image.open(mask_path))
    if m.ndim == 3:
        m = m[..., 0]
    if dataset == "almeria":
        sky = np.full(m.shape, IGNORE, np.uint8)
        sky[m == 0], sky[m == 1], sky[np.isin(m, (2, 3, 4))] = SKY_INVALID, SKY, CLOUD
        layer = np.full(m.shape, IGNORE, np.uint8)
        layer[m == 2], layer[m == 3], layer[m == 4] = LAYER_LOW, LAYER_MID, LAYER_HIGH
        return sky, layer
    if dataset in {"swimseg", "swinseg", "swinyseg", "shwimseg"}:
        sky = np.where(m >= 128, CLOUD, SKY).astype(np.uint8)
        return sky, np.full(m.shape, IGNORE, np.uint8)
    raise ValueError(f"unknown segmentation dataset {dataset!r}")
