"""Cloud-base-height training data on cached features (P067, P075, P076): the P047 pairs at a 10-minute cadence,
ray maps at the pooled grid, the P049 synthetic consumer views, and the scoring of a predicted bin distribution."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..baselines import cbh as C
from ..data.eye2sky import load_mask, mask_path, station_calibrations
from ..data.image_cache import read_rgb
from ..features.cache import FeatureCache
from ..geometry import synthesis as SYN
from ..geometry.cameras import load_cloudscope_model
from ..geometry.pose import load_pose_records, pose_for
from ..geometry.raymap import meta_vector, patch_rays, ray_map
from ..geometry.sun import sun_position
from ..model import cbh_bins as B
from ..model.cbh import CbhHead
from ..model.skyray import GeometryEncoder

SIZE, PATCH = 512, 64            # the pooled grid: 8 x 8 cells of 64 px


def load_pairs(grid: FeatureCache) -> pd.DataFrame:
    d = pd.read_parquet("data/cbh_pairs.parquet")
    d = d[d.paired & d.daytime & d.zenith_inside_mask & d.split.isin(["train", "val"]) & (d.utc.dt.minute % 10 == 0)].copy()
    d["sample_id"] = "eye2sky:" + d.image_file
    cached = set(grid.index().sample_id)
    d = d[d.sample_id.isin(cached)].reset_index(drop=True)
    d["cloud"] = d.label != "none"
    d["height_m"] = np.where(d.cloud, d.cbh_m, np.nan)
    return d


def ray_maps_for(pairs: pd.DataFrame, data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """(ray_maps [n, 4, 8, 8], metas [n, 3]) per pair from the station calibration, the fitted pose and the Sun."""
    meta_root = data_root / "Eye2Sky" / "asi_meta"
    records = load_pose_records()
    rms = np.zeros((len(pairs), 4, SIZE // PATCH, SIZE // PATCH), dtype=np.float32)
    metas = np.zeros((len(pairs), 3), dtype=np.float32)
    for (station, cid), part in pairs.groupby(["station", "calib_id"]):
        cal = next(c for c in station_calibrations(meta_root, station) if c.file == cid)
        pose = pose_for(cid, None, records)
        if pose is None:
            continue
        mask = load_mask(mask_path(meta_root, cal)) > 0
        rays = patch_rays(cal.model, None, SIZE, PATCH, mask)
        sun = sun_position(part.utc, cal.latitude, cal.longitude, cal.altitude)
        for i, az, zen in zip(part.index, sun.azimuth.to_numpy(), sun.apparent_zenith.to_numpy(), strict=True):
            rms[i] = ray_map(cal.model, pose, float(az), float(zen), out_size=SIZE, patch=PATCH, rays=rays)
            metas[i] = meta_vector(float(zen))
    return rms, metas


class PairDataset(Dataset):
    def __init__(self, rows_grid, rows_cls, grid, cls, rms, metas, heights, weights=None):
        self.rows_grid, self.rows_cls, self.grid, self.cls = rows_grid, rows_cls, grid, cls
        self.rms, self.metas, self.heights = rms, metas, heights
        self.weights = np.ones(len(heights), np.float32) if weights is None else np.asarray(weights, np.float32)

    def __len__(self):
        return len(self.rows_grid)

    def __getitem__(self, i):
        return (torch.from_numpy(np.asarray(self.grid[self.rows_grid[i]], dtype=np.float32)),
                torch.from_numpy(np.asarray(self.cls[self.rows_cls[i]], dtype=np.float32)),
                torch.from_numpy(self.rms[i]), torch.from_numpy(self.metas[i]),
                torch.tensor(self.heights[i], dtype=torch.float32), torch.tensor(self.weights[i]))


class ArrayDataset(Dataset):
    """In-memory tokens (synthetic views, P049) with the same item layout as PairDataset."""

    def __init__(self, grid, cls, rms, metas, heights, weights=None):
        self.grid, self.cls, self.rms, self.metas, self.heights = grid, cls, rms, metas, heights
        self.weights = np.ones(len(heights), np.float32) if weights is None else np.asarray(weights, np.float32)

    def __len__(self):
        return len(self.heights)

    def __getitem__(self, i):
        return (torch.from_numpy(self.grid[i]), torch.from_numpy(self.cls[i]), torch.from_numpy(self.rms[i]),
                torch.from_numpy(self.metas[i]), torch.tensor(self.heights[i], dtype=torch.float32),
                torch.tensor(self.weights[i]))


def synthetic_views(cache_root: Path, device: str, pool: int = 4) -> ArrayDataset | None:
    """P049 consumer views with a label: tokens (pooled grid + cls) from the backbone, ray maps from the view's own
    camera and pose, heights from the inherited ceilometer label."""
    from stratia.features.backbones import FrozenBackbone

    idx_path = Path("data/synthetic_views.parquet")
    if not idx_path.exists():
        return None
    sv = pd.read_parquet(idx_path)
    sv = sv[sv.label.notna()].reset_index(drop=True)
    if not len(sv):
        return None
    bb = FrozenBackbone("dinov3_vits16")
    nominal = load_cloudscope_model(Path("configs/cameras/b0268_nominal.camera.json"))
    grids, clss, rms, metas, heights = [], [], [], [], []
    side = SIZE // 16
    for r in sv.itertuples(index=False):
        img = read_rgb(cache_root / "synthetic_views" / r.view_file)
        valid = img.max(axis=-1) > 0
        x = torch.from_numpy(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA).transpose(2, 0, 1))[None]
        out = bb.forward(x, patches=True)
        g = out["patches"].float().transpose(1, 2).reshape(1, -1, side, side)
        grids.append(torch.nn.functional.avg_pool2d(g, pool).flatten(2).transpose(1, 2)[0].cpu().numpy().astype(np.float32))
        clss.append(out["cls"].float()[0].cpu().numpy().astype(np.float32))
        cam = SYN.scaled_camera(nominal, int(r.width))
        pose = SYN.pointing_pose(SYN.Pointing(float(r.azimuth_deg), float(r.elevation_deg), float(r.roll_deg)))
        rm = ray_map(cam, pose, float(r.sun_azimuth_deg), float(r.sun_zenith_deg), mask=valid, out_size=SIZE, patch=PATCH)
        rms.append(rm)
        metas.append(meta_vector(float(r.sun_zenith_deg)))
        heights.append(float(r.cbh_m) if r.label != "none" and pd.notna(r.cbh_m) else float("nan"))
    del bb
    torch.cuda.empty_cache()
    return ArrayDataset(np.stack(grids), np.stack(clss), np.stack(rms).astype(np.float32), np.stack(metas).astype(np.float32),
                        np.asarray(heights, np.float32))


class Model(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.geo = GeometryEncoder(d_model)
        self.head = CbhHead(d_model)

    def forward(self, tokens, cls, rm, meta):
        t, c = self.geo(tokens, cls, rm, meta)
        return self.head(t, c, rm)


@torch.no_grad()
def predict(model: Model, dl: DataLoader, device: str, null_geometry: bool = False) -> np.ndarray:
    model.eval()
    out = []
    for tok, cls, rm, meta, _, _ in dl:
        if null_geometry:
            rm, meta = torch.zeros_like(rm), torch.zeros_like(meta)
        out.append(model(tok.to(device), cls.to(device), rm.to(device), meta.to(device))["cbh_probs"].float().cpu().numpy())
    return np.concatenate(out)


def score(probs: np.ndarray, heights: np.ndarray, cloud: np.ndarray, mae_clim: float) -> dict:
    cloudy = ~np.isnan(heights)
    pred = B.median_height_m(probs)
    pred_c = np.where(np.isnan(pred), B.centres_m()[probs[:, :B.K_BINS].argmax(1)], pred)   # a height for every cloudy row
    sc = C.score_heights(pred_c[cloudy], heights[cloudy], mae_clim)
    lo, hi = B.interval_m(probs[cloudy])
    cover = float(np.mean((heights[cloudy] >= lo) & (heights[cloudy] <= hi)))
    target = B.soft_target(heights)
    crps_c = ((np.cumsum(probs[cloudy, :B.K_BINS] / probs[cloudy, :B.K_BINS].sum(1, keepdims=True), 1)
               - np.cumsum(target[cloudy, :B.K_BINS], 1)) ** 2).mean()
    pred_cloud = probs[:, B.NO_CLOUD] < 0.5
    bal = C.balanced_accuracy(cloud, pred_cloud)
    # mode-collapse guard: share of predictions in the most common predicted bin vs the labels' most common bin
    pred_bins = probs[cloudy, :B.K_BINS].argmax(1)
    true_bins = B.bin_of(heights[cloudy])
    top_pred = float(np.bincount(pred_bins, minlength=B.K_BINS).max() / len(pred_bins))
    top_true = float(np.bincount(true_bins, minlength=B.K_BINS).max() / len(true_bins))
    def ent(b):
        p = np.bincount(b, minlength=B.K_BINS) / len(b) + 1e-12
        return float(-(p @ np.log(p)))
    return {**sc.row(), "crps": float(crps_c), "coverage_80": cover, "cloud_balanced_acc": bal,
            "top_bin_share_pred": top_pred, "top_bin_share_true": top_true,
            "entropy_ratio": ent(pred_bins) / max(ent(true_bins), 1e-9), "n_cloudy": int(cloudy.sum())}
