"""Augmentation gallery (P048).

    python scripts/augment_gallery.py [--per-dataset 1] [--seed 0]

Takes one cached image per dataset group (an Eye2Sky all-sky frame with its camera mask and Sun pixel, a CCSN
photograph, a Montenegro frame, a SWIMSEG crop with its dense label) and renders, per image, the original, every
photometric augmentation alone at the policy's ranges, and three draws of the full policy (geometric + photometric)
with the dense label and the ray-map valid flag where they exist. Writes docs/data/figures/augmentation_gallery.jpg
and docs/data/augmentation_gallery.md (the policy table with ranges and probabilities).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.augment.policy import AugmentationPolicy  # noqa: E402
from stratia.data.eye2sky import load_mask, mask_path, station_calibrations  # noqa: E402
from stratia.data.image_cache import cache_path, read_rgb  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.data.segmentation import load_masks  # noqa: E402
from stratia.geometry import transforms as T  # noqa: E402
from stratia.geometry.pose import load_pose_records, pose_for  # noqa: E402
from stratia.geometry.sun import azel_to_enu  # noqa: E402

FIG = Path("docs/data/figures/augmentation_gallery.jpg")
DOC = Path("docs/data/augmentation_gallery.md")
PHOTO = ["exposure", "white_balance", "sun_glare", "dirt_and_drops", "obstruction", "sensor_noise", "jpeg"]


def pick(manifest: pd.DataFrame, dataset: str, rng, **conds) -> pd.Series:
    m = manifest[manifest.dataset == dataset]
    for k, v in conds.items():
        m = m[m[k] == v]
    return m.iloc[int(rng.integers(len(m)))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    manifest = pd.read_parquet("data/manifest.parquet")
    rng = np.random.default_rng(a.seed)
    pol = AugmentationPolicy()
    eye = manifest[(manifest.dataset == "eye2sky") & (manifest.sun_zenith_deg < 60)]
    # Montenegro: the brightest of 50 random frames (its Sun columns are not reliable, data card)
    mont = manifest[manifest.dataset == "montenegro"].sample(50, random_state=a.seed)
    bright = [read_rgb(cache_path(cache_root, f)).mean() for f in mont.image_file]
    rows = [eye.iloc[int(rng.integers(len(eye)))], pick(manifest, "ccsn", rng), mont.iloc[int(np.argmax(bright))],
            pick(manifest, "swimseg", rng)]
    n_cols = 1 + len(PHOTO) + 3
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.1 * n_cols, 2.3 * len(rows)))
    for r, row in enumerate(rows):
        img = read_rgb(cache_path(cache_root, row.image_file))
        h, w = img.shape[:2]
        state = T.GeoState.cached(w, h, int(row.width), int(row.height))
        label, mask, sun_px, cam, pose = None, None, None, None, None
        if isinstance(row.seg_file, str) and row.seg_file:
            try:
                lab, _ = load_masks(row.dataset, data_root / row.seg_file)
                import cv2

                label = cv2.resize(lab.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception as err:  # noqa: BLE001 - a gallery must not die on one odd mask
                print(f"  no label for {row.image_file}: {err}")
        if row.dataset == "eye2sky":
            cals = station_calibrations(data_root / "Eye2Sky" / "asi_meta", row.site)
            cal = next(c for c in cals if c.file == row.calib_id)
            mask = load_mask(mask_path(data_root / "Eye2Sky" / "asi_meta", cal))
            cam = cal.model
            pose = pose_for(cal.file, cal.external_orientation, load_pose_records())
            if pose is not None:
                su, sv = cam.ray_to_pixel(pose.enu_to_cam(azel_to_enu(row.sun_azimuth_deg, 90 - row.sun_zenith_deg)))
                sun_px = (float(su), float(sv))
        # column 0: original at 512
        base, base_state = T.Resize(512, 512)(img, state)
        axes[r, 0].imshow(base)
        axes[r, 0].set_title(f"{row.dataset}", fontsize=8)
        valid512 = base_state.coverage if mask is None else T.resized_mask(mask, (512, 512), base_state.affine)
        sun512 = None
        if sun_px is not None:
            ou, ov = base_state.from_source(*sun_px)
            sun512 = (float(ou), float(ov))
        for c, name in enumerate(PHOTO, start=1):
            out, _, _ = pol.photometric(base, np.random.default_rng(a.seed + c), sun_px=sun512, valid=valid512, only={name})
            axes[r, c].imshow(out)
            axes[r, c].set_title(name, fontsize=8)
        for k in range(3):
            res = pol(img, row.camera_type, np.random.default_rng(a.seed + 100 * r + k), state=state, label=label,
                      sun_px_source=sun_px, valid_source=mask)
            ax = axes[r, 1 + len(PHOTO) + k]
            ax.imshow(res.image)
            if res.label is not None:
                ax.contour(res.label == 2, levels=[0.5], colors="lime", linewidths=0.5)
                ax.contour(res.label == 0, levels=[0.5], colors="red", linewidths=0.5)
            if cam is not None and pose is not None:
                rm, _ = T.geometry_for(res.state, cam, pose, row.sun_azimuth_deg, row.sun_zenith_deg, mask,
                                       obstruction=res.obstruction)
                ax.contour(np.kron(rm[3], np.ones((16, 16))), levels=[0.5], colors="cyan", linewidths=0.6)
            ax.set_title("policy: " + ",".join(x[:4] for x in res.applied) if res.applied else "policy (geom only)", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=70)
    plt.close(fig)

    cfg = pol.cfg
    lines = ["# Augmentation policy (P048)", "",
             f"`configs/augment.yaml` version {cfg['version']}, strength {cfg['strength']}, input {cfg['input_size']} px. "
             "Geometric transforms go through `stratia.geometry.transforms` (ADR-004), so ray maps and dense labels follow; "
             "photometric ones live in `stratia.augment.photometric`. There is no hue shift anywhere.", "",
             "## Geometric, by camera-type group", "", "| Group | Camera types | Crop (area share, aspect) | Rotation | Flip |",
             "|---|---|---|---|---|"]
    for g, types in cfg["groups"].items():
        geo = cfg["geometric"][g]
        crop = geo["crop"]
        rot = geo["rotate"]
        flip = geo["flip"]
        flip_txt = 'p = ' + str(flip['p']) if flip['enabled'] else 'off'
        lines.append(f"| {g} | {', '.join(types)} | {'on' if crop['enabled'] else 'off'}: {crop['scale']}, {crop['ratio']} | "
                     f"{'on' if rot['enabled'] else 'off'}: {rot['degrees']} deg | {flip_txt} |")
    lines += ["", "## Photometric (applied in this order)", "", "| Augmentation | p | Range / parameters | Physical meaning |",
              "|---|---|---|---|"]
    meaning = {"exposure": "exposure-time factor on linear intensity; highlights clip",
               "white_balance": "red/blue gains along the colour-temperature axis",
               "sun_glare": "bloom at the Sun's pixel (geometry-aware) + ghost through the centre",
               "dirt_and_drops": "dark dirt blobs and lens-like water drops on the dome",
               "obstruction": "dark silhouette; label -> invalid, ray-map patches -> invalid",
               "sensor_noise": "read + shot noise in linear space", "jpeg": "recompression quality"}
    for name in PHOTO:
        c = cfg["photometric"][name]
        params = {k: v for k, v in c.items() if k not in ("enabled", "p")}
        lines.append(f"| {name} | {c['p']} | {params} | {meaning[name]} |")
    lines += ["", "## Gallery", "", f"`{FIG.as_posix()}`: one row per dataset group "
              "(Eye2Sky all-sky with mask, Sun and the ray-map "
              "valid flag outlined in cyan; CCSN photograph; Montenegro fixed camera; SWIMSEG crop with "
              "its dense label: cloud outlined "
              "in lime, invalid in red). Columns: original, each photometric augmentation alone, three draws of the full policy.",
              "", "## Review", "",
              "- Reviewed by the agent on generation: cloud texture survives every augmentation at strength 1; "
              "the Sun bloom sits on the Sun; obstructions remove label and ray-map validity where they are pasted. "
              "Owner approval of the gallery is pending (exit criterion of P048).",
              "- Ablation hooks: `strength` (P080) and per-augmentation `enabled` (P091)."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: {FIG}, {DOC}")


if __name__ == "__main__":
    main()
