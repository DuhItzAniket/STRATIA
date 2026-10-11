"""Corruption-robustness suite (P088): the candidate heads on corrupted validation images.

    python scripts/robustness_suite.py [--max-global 1200] [--max-cbh 800]

Corruptions (applied to the cached 768 px image before the full-frame 512 px resize): exposure ×0.6 and ×1.5, haze
(blend toward a flat grey, 40 %), dirty lens (P048 `dirt_and_drops`), sun glare (P048 `sun_glare` at the Sun's pixel
when the geometry is known, else at a random position), JPEG quality 30, sensor noise (read 0.03, shot 0.06), blur
(Gaussian, σ = 2 px). The backbone runs live on every corrupted image; the heads are the Stage H candidates: the
global heads (P073 `indomain_e2s`, on the 8 x 8 pooled grid + cls), the dense heads (P074) and the CBH head (P075
base, with the frame's ray map). Metrics as in Stage H: mean étage macro-F1 and genus macro-F1 (CCSN, MGCD) on a
subset of the recognition val rows, sky-parsing mIoU on the segmentation val images and the clear-sky cloud fraction,
CBH MAE / skill on a subset of the validation pairs. Output: docs/data/robustness.md, data/robustness.parquet.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.augment import photometric as P  # noqa: E402
from stratia.baselines import cbh as C  # noqa: E402
from stratia.baselines import segmentation as S  # noqa: E402
from stratia.data.image_cache import cache_path, read_rgb  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.data.segmentation import CLOUD  # noqa: E402
from stratia.features.backbones import FrozenBackbone  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.model.dense import DenseHeads  # noqa: E402
from stratia.train import cbh_data as CD  # noqa: E402
from stratia.train import data as D  # noqa: E402
from stratia.train.global_eval import GlobalModel, score_dataset  # noqa: E402

DOC = Path("docs/data/robustness.md")
OUT = Path("data/robustness.parquet")
SIZE = 512


def corruptions(rng: np.random.Generator) -> dict:
    def haze(img):
        grey = np.full_like(img, 200)
        return cv2.addWeighted(img, 0.6, grey, 0.4, 0)

    def blur(img):
        return cv2.GaussianBlur(img, (0, 0), 2.0)

    def exposure(f):
        return lambda img: P.exposure(img, rng, f, f)

    return {
        "clean": lambda img, sun=None: img,
        "exposure x0.6": lambda img, sun=None: exposure(0.6)(img),
        "exposure x1.5": lambda img, sun=None: exposure(1.5)(img),
        "haze 40%": lambda img, sun=None: haze(img),
        "dirty lens": lambda img, sun=None: P.dirt_and_drops(img, rng),
        "sun glare": lambda img, sun=None: P.sun_glare(img, rng, sun_px=sun, strength=1.0),
        "jpeg q30": lambda img, sun=None: P.jpeg(img, rng, 30, 30),
        "sensor noise": lambda img, sun=None: P.sensor_noise(img, rng, 0.03, 0.06),
        "blur 2px": lambda img, sun=None: blur(img),
    }


@torch.no_grad()
def tokens_for(bb: FrozenBackbone, img: np.ndarray, pool: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA).transpose(2, 0, 1))[None]
    out = bb.forward(x, patches=True)
    tok = out["patches"].float()
    if pool:
        side = int(np.sqrt(tok.shape[1]))
        tok = torch.nn.functional.avg_pool2d(tok.transpose(1, 2).reshape(1, -1, side, side), pool).flatten(2).transpose(1, 2)
    return tok, out["cls"].float()


def load_global(run_dir: Path, device: str) -> tuple[GlobalModel, pd.DataFrame]:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    rows = D.global_rows(eye2sky_etage=False)
    model = GlobalModel(384, int(cfg["hidden"]), float(cfg["dropout"]), bool(cfg["kendall"])).to(device)
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if state.get("ema"):
        model.load_state_dict({**model.state_dict(), **state["ema"]["shadow"]})
    return model.eval(), rows[rows.split == "val"].reset_index(drop=True)


def load_dense(run_dir: Path, device: str) -> DenseHeads:
    model = DenseHeads(384).to(device)
    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{run_dir} has no checkpoint")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--global-run", default=None)
    ap.add_argument("--dense-run", default=None)
    ap.add_argument("--cbh-run", default=None)
    ap.add_argument("--max-global", type=int, default=1200)
    ap.add_argument("--max-dense", type=int, default=400)
    ap.add_argument("--max-cbh", type=int, default=800)
    ap.add_argument("--clear-frames", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = Path("runs")
    g_run = Path(a.global_run) if a.global_run else sorted(runs.glob("*global_indomain_e2s_s0*"))[-1]
    d_run = Path(a.dense_run) if a.dense_run else sorted(runs.glob("*dense_sky_layer_s0*"))[-1]
    c_run = Path(a.cbh_run) if a.cbh_run else sorted(runs.glob("*cbh_head_s0_[[]P075_base[]]*"))[-1]
    bb = FrozenBackbone("dinov3_vits16")
    corr = corruptions(rng)

    # ---- global rows (subset) with their targets
    g_model, g_rows = load_global(g_run, device)
    g_rows = g_rows[g_rows.dataset.isin(["ccsn", "mgcd", "montenegro", "almeria"])]
    g_rows = g_rows.sample(min(a.max_global, len(g_rows)), random_state=a.seed).reset_index(drop=True)
    g_imgs = [read_rgb(cache_path(cache_root, f)) for f in g_rows.image_file]
    # ---- dense val subset
    tab = S.seg_tables(data_root)
    sky_masks, _ = S.load_both_masks(tab, data_root)
    va = np.flatnonzero((tab.split == "val").to_numpy())
    va = rng.choice(va, min(a.max_dense, len(va)), replace=False)
    d_model = load_dense(d_run, device)
    d_imgs = [read_rgb(cache_path(cache_root, tab.image_file.iloc[i])) for i in va]
    d_masks = [cv2.resize(sky_masks[i], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) for i in va]
    clear = S.clear_sky_frames(a.clear_frames)
    cam_masks = S.camera_masks(clear, data_root, SIZE) if len(clear) else []
    clear_imgs = [read_rgb(cache_path(cache_root, f)) for f in clear.image_file]
    # ---- CBH val subset with ray maps
    grid8 = FeatureCache(cache_root, "dinov3_vits16", 512, transform="ffr1-patches-all-pool4")
    pairs = CD.load_pairs(grid8)
    pv = pairs[pairs.split == "val"].sample(min(a.max_cbh, int((pairs.split == "val").sum())), random_state=a.seed)
    pv = pv.reset_index(drop=True)
    rms, metas = CD.ray_maps_for(pv, data_root)
    c_model = CD.Model(384).to(device)
    c_model.load_state_dict(torch.load(c_run / "checkpoints" / "best.pt", map_location=device, weights_only=False)["model"])
    c_model.eval()
    c_imgs = [read_rgb(cache_path(cache_root, f)) for f in pv.image_file]
    true_m = pv.height_m.to_numpy().astype(np.float32)
    cloud = pv.cloud.to_numpy()
    mae_clim = float(C.score_heights(np.full(int((~np.isnan(true_m)).sum()), C.climatology(pairs[pairs.split == "train"])),
                                     true_m[~np.isnan(true_m)]).mae)
    # the Sun's pixel for the glare corruption on CBH frames: the ray-map cell closest to the Sun direction
    sun_px = []
    for rm in rms:
        d = rm[:3].reshape(3, -1)
        valid = rm[3].reshape(-1) > 0.5
        if valid.any():
            k = int(np.argmax(np.where(valid, d[0], -np.inf)))              # x = toward the Sun's azimuth
            r, c = divmod(k, rm.shape[2])
            sun_px.append(((c + 0.5) * 768 / rm.shape[2], (r + 0.5) * 768 / rm.shape[1]))
        else:
            sun_px.append(None)
    print(f"global {len(g_rows)} rows, dense {len(d_imgs)} images, clear {len(clear_imgs)}, cbh {len(pv)} pairs; "
          f"{len(corr)} corruptions", flush=True)

    rows = []
    for name, fn in corr.items():
        t1 = time.perf_counter()
        # global
        gen, eta, okt = [], [], []
        for img in g_imgs:
            tok, cls = tokens_for(bb, np.ascontiguousarray(fn(img)), pool=4)
            n = 1
            with torch.no_grad():
                out = g_model(tok.to(device), cls.to(device), torch.zeros(n, 4, 8, 8, device=device),
                              torch.zeros(n, 3, device=device))
            gen.append(torch.sigmoid(out["genus_logits"]).cpu().numpy()[0])
            eta.append(torch.sigmoid(out["etage_logits"]).cpu().numpy()[0])
            okt.append(out["oktas_probs"].cpu().numpy()[0])
        pred = {"genus": np.stack(gen), "etage": np.stack(eta), "oktas": np.stack(okt)}
        e_vals, g_vals = [], []
        for ds_name in g_rows.dataset.unique():
            sel = (g_rows.dataset == ds_name).to_numpy()
            sc = score_dataset(g_rows[sel].reset_index(drop=True), {k: v[sel] for k, v in pred.items()})
            if "etage_f1" in sc:
                e_vals.append(sc["etage_f1"])
            if "genus_f1" in sc:
                g_vals.append(sc["genus_f1"])
        row = {"corruption": name, "etage_f1": float(np.mean(e_vals)), "genus_f1": float(np.mean(g_vals))}
        # dense
        cm = np.zeros((S.N_CLASSES, S.N_CLASSES), dtype=np.int64)
        for img, m in zip(d_imgs, d_masks, strict=True):
            tok, _ = tokens_for(bb, np.ascontiguousarray(fn(img)), pool=None)
            with torch.no_grad():
                logits = torch.nn.functional.interpolate(d_model(tok.to(device))["sky_parse_logits"].float(), size=(SIZE, SIZE),
                                                         mode="bilinear", align_corners=False)
            cm += S.confusion(logits.argmax(1)[0].cpu().numpy().astype(np.uint8), m)
        row["sky_miou"] = S.scores_from_confusion(cm).miou
        fr = []
        for img, m in zip(clear_imgs, cam_masks, strict=True):
            tok, _ = tokens_for(bb, np.ascontiguousarray(fn(img)), pool=None)
            with torch.no_grad():
                logits = torch.nn.functional.interpolate(d_model(tok.to(device))["sky_parse_logits"].float(), size=(SIZE, SIZE),
                                                         mode="bilinear", align_corners=False)
            fr.append(float((logits.argmax(1)[0].cpu().numpy()[m] == CLOUD).mean()))
        row["clear_sky_cloud_fraction"] = float(np.mean(fr)) if fr else float("nan")
        # cbh
        probs = []
        for img, rm, mt, sp in zip(c_imgs, rms, metas, sun_px, strict=True):
            tok, cls = tokens_for(bb, np.ascontiguousarray(fn(img, sp)), pool=4)
            with torch.no_grad():
                out = c_model(tok.to(device), cls.to(device), torch.from_numpy(rm)[None].to(device),
                              torch.from_numpy(mt)[None].to(device))
            probs.append(out["cbh_probs"].float().cpu().numpy()[0])
        sc = CD.score(np.stack(probs), true_m, cloud, mae_clim)
        row.update({"cbh_mae_m": sc["mae_m"], "cbh_skill": sc["skill"], "cbh_cloud_bal_acc": sc["cloud_balanced_acc"]})
        row["seconds"] = time.perf_counter() - t1
        rows.append(row)
        print(f"{name:14s}: étage {row['etage_f1']:.3f} genus {row['genus_f1']:.3f} | mIoU {row['sky_miou']:.3f} clear-cloud "
              f"{row['clear_sky_cloud_fraction']:.3f} | CBH {row['cbh_mae_m']:.0f} m skill {row['cbh_skill']:+.3f} | "
              f"{row['seconds']:.0f} s",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    write_doc(df, len(g_rows), len(d_imgs), len(clear_imgs), len(pv))
    print(f"done in {time.perf_counter() - t0:.0f} s -> {DOC}", flush=True)
    return 0


def write_doc(df: pd.DataFrame, n_g: int, n_d: int, n_c: int, n_p: int) -> None:
    base = df[df.corruption == "clean"].iloc[0]
    lines = ["# Corruption robustness (P088)", "",
             f"Stage H candidates on corrupted validation images, backbone run live: {n_g} recognition rows (CCSN, MGCD, "
             "Montenegro, "
             f"Almería), {n_d} segmentation images, {n_c} clear-sky Eye2Sky frames, {n_p} CBH pairs. Each cell: metric under the "
             "corruption (change vs clean in brackets).", "",
             "| Corruption | Mean étage F1 | Genus F1 (CCSN, MGCD) | Sky mIoU | Clear-sky cloud fraction | CBH MAE (m) | "
             "CBH skill |",
             "|---|---|---|---|---|---|---|"]
    def d(r, k, fmt="{:+.3f}"):
        return "" if r.corruption == "clean" else " (" + fmt.format(getattr(r, k) - getattr(base, k)) + ")"

    for r in df.itertuples():
        lines.append(f"| {r.corruption} | {r.etage_f1:.3f}{d(r, 'etage_f1')} | {r.genus_f1:.3f}{d(r, 'genus_f1')} | "
                     f"{r.sky_miou:.3f}{d(r, 'sky_miou')} | {r.clear_sky_cloud_fraction:.3f}{d(r, 'clear_sky_cloud_fraction')} | "
                     f"{r.cbh_mae_m:.0f}{d(r, 'cbh_mae_m', '{:+.0f}')} | {r.cbh_skill:+.3f}{d(r, 'cbh_skill')} |")
    lines += ["", "Rows in `data/robustness.parquet`. Corruptions follow the P048 photometric functions where one exists."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
