"""Cached-feature data for Stage H (P073-P076): label tables with per-head targets and masks, ray maps at the
pooled grid, and torch datasets over the P051 caches.

Global rows (genus / étage / oktas) come from `data/labels.parquet` (P034 sets), `data/montenegro_targets.parquet`
(P035 shares) and, optionally, the ceilometer étage of the P047 frames; the split is the grouped in-domain split
(P038) or a LODO fold (P039 v2). Every row carries, per head, a target and a mask, plus per-class weights that say
which classes are *known* for that row (ADR-006 §1-2): a single-label CCSN picture knows one positive and weak
negatives (0.3) elsewhere; a SWIMCAT "cloud" patch knows only that it is not clear; a Montenegro vote knows every
share; an Almería mask gives étages and nothing about genera.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..contract import ETAGE_CLASSES, GENUS_CLASSES, N_OKTAS
from ..data.eye2sky import load_mask, mask_path, station_calibrations
from ..geometry.pose import load_pose_records, pose_for
from ..geometry.raymap import meta_vector, patch_rays, ray_map
from ..geometry.sun import sun_position
from ..labels.ontology import load_ontology

REPO_ROOT = Path(__file__).resolve().parents[2]
G, E = len(GENUS_CLASSES), len(ETAGE_CLASSES)
GIDX = {g: i for i, g in enumerate(GENUS_CLASSES)}
EIDX = {e: i for i, e in enumerate(ETAGE_CLASSES)}
SINGLE_LABEL_NEGATIVE = 0.3          # ADR-006 §1: single-label datasets are not exhaustive
GLOBAL_SOURCES = ("ccsn", "mgcd", "swim", "montenegro", "almeria")
GLOBAL_DATASETS = ("ccsn", "mgcd", "swimcat", "montenegro", "almeria")   # the SWIM masks carry no global label
SIZE, POOL_PATCH = 512, 64           # the pooled 8 x 8 grid


# ----------------------------------------------------------------------------------------------- label rows
def _set(raw) -> list:
    return [] if raw is None or (isinstance(raw, float) and np.isnan(raw)) else json.loads(raw)


def global_rows(root: str | Path = REPO_ROOT, lodo_fold: str | None = None, eye2sky_etage: bool = False,
                include_test: bool = False) -> pd.DataFrame:
    """One row per sample with global targets. Columns: sample_id, dataset, source, split, genus_set [12], genus_alt,
    genus_neg [12] (per-class negative weight; 0 = unknown), mask_genus, etage [3], etage_w [3], mask_etage,
    oktas [9], mask_oktas, has_geometry, calib_id, station, utc."""
    root = Path(root)
    lab = pd.read_parquet(root / "data/labels.parquet")
    man = pd.read_parquet(root / "data/manifest.parquet", columns=["sample_id", "dataset", "image_file", "utc", "calib_id",
                                                                   "site", "source_label"])
    ind = pd.read_parquet(root / "data/splits/in_domain.parquet", columns=["sample_id", "source", "split", "unit", "stratum"])
    d = lab.merge(man, on=["sample_id", "dataset"]).merge(ind, on="sample_id")
    d = d[d.source.isin(GLOBAL_SOURCES) & d.dataset.isin(GLOBAL_DATASETS)].copy()
    if lodo_fold is not None:
        lodo = pd.read_parquet(root / "data/splits/lodo.parquet")
        lodo = lodo[lodo.fold == lodo_fold][["sample_id", "role"]]
        d = d.merge(lodo, on="sample_id", how="inner")
        d["split"] = d.role.map({"train": "train", "val": "val", "test": "test"})
        d = d.drop(columns=["role"])
    if not include_test:
        d = d[d.split != "test"]
    mt = pd.read_parquet(root / "data/montenegro_targets.parquet")
    mt["sample_id"] = "montenegro:" + mt.image_file
    mt = mt.set_index("sample_id")

    rows = []
    for r in d.itertuples(index=False):
        genus = np.zeros(G, np.float32)
        neg = np.zeros(G, np.float32)
        alt = bool(r.genus_alternatives)
        mask_g = False
        etage = np.zeros(E, np.float32)
        etage_w = np.zeros(E, np.float32)
        mask_e = False
        oktas = np.zeros(N_OKTAS, np.float32)
        mask_o = False
        if r.dataset == "montenegro" and r.sample_id in mt.index:
            t = mt.loc[r.sample_id]
            gs = json.loads(t.genus_share)
            for g, v in gs.items():
                genus[GIDX[g]] = float(v)
            genus[GIDX["clear"]] = float(1.0 - t.cloud_share)
            neg[:] = 1.0                                         # exhaustive votes: every class known (soft targets)
            neg[GIDX["contrail"]] = 0.0
            alt = False
            mask_g = True
            es = json.loads(t.etage_share)
            for e, v in es.items():
                etage[EIDX[e]] = float(v)
            etage_w[:] = 1.0
            mask_e = True
            od = json.loads(t.oktas_dist)
            if od:
                for k, v in od.items():
                    oktas[int(k)] = float(v)
                oktas /= max(oktas.sum(), 1e-9)
                mask_o = True
        else:
            if r.genus_set is not None and not (isinstance(r.genus_set, float) and np.isnan(r.genus_set)):
                gset = _set(r.genus_set)
                if r.extra == "clear":
                    genus[GIDX["clear"]] = 1.0
                    neg[:] = 1.0
                    neg[GIDX["clear"]] = 0.0
                elif r.extra == "contrail":
                    genus[GIDX["contrail"]] = 1.0
                    neg[:] = SINGLE_LABEL_NEGATIVE
                    neg[GIDX["clear"]] = 1.0
                else:
                    for g in gset:
                        genus[GIDX[g]] = 1.0
                    neg[:] = SINGLE_LABEL_NEGATIVE
                    neg[GIDX["clear"]] = 1.0                     # a cloud genus is known: not clear
                mask_g = True
            elif r.cloud_present is True:                        # cloud of unnamed genus: only 'clear' is known false
                neg[GIDX["clear"]] = 1.0
                mask_g = True
            has_etage = r.etage_set is not None and not (isinstance(r.etage_set, float) and np.isnan(r.etage_set))
            es = _set(r.etage_set) if has_etage else None
            if es is not None:
                for e in es:
                    etage[EIDX[e]] = 1.0
                etage_w[:] = 1.0
                mask_e = True
        rows.append({"sample_id": r.sample_id, "dataset": r.dataset, "source": r.source, "split": r.split, "unit": r.unit,
                     "genus_set": genus, "genus_alt": alt, "genus_neg": neg, "mask_genus": mask_g, "etage": etage,
                     "etage_w": etage_w, "mask_etage": mask_e, "oktas": oktas, "mask_oktas": mask_o,
                     "calib_id": r.calib_id if isinstance(r.calib_id, str) else None,
                     "station": r.site if r.dataset == "eye2sky" else None,
                     "utc": r.utc, "native_label": r.stratum, "image_file": r.image_file})
    out = pd.DataFrame(rows)
    if eye2sky_etage:
        out = pd.concat([out, eye2sky_etage_rows(root, include_test)], ignore_index=True)
    out["has_geometry"] = out.calib_id.notna()
    return out


def eye2sky_etage_rows(root: str | Path = REPO_ROOT, include_test: bool = False, min_confidence: float = 0.8,
                       cadence_min: int = 10) -> pd.DataFrame:
    """The P047 frames as global rows with a partial étage label: the ceilometer's lowest-base étage is a known
    positive (weight 1), the étages below it are weak negatives (weight 0.3: no lower base at the zenith, lower cloud
    elsewhere in the hemisphere possible), the étages above it are unknown (weight 0); 'none' frames give no étage
    target (horizon clouds unseen)."""
    d = pd.read_parquet(Path(root) / "data/cbh_pairs.parquet")
    d = d[d.paired & d.daytime & d.zenith_inside_mask & (d.utc.dt.minute % cadence_min == 0)]
    d = d[d.confidence >= min_confidence]
    if not include_test:
        d = d[d.split.isin(["train", "val"])]
    d = d[d.label.isin(ETAGE_CLASSES)]
    rows = []
    for r in d.itertuples(index=False):
        etage = np.zeros(E, np.float32)
        w = np.zeros(E, np.float32)
        etage[EIDX[r.label]] = 1.0
        w[EIDX[r.label]] = 1.0
        w[: EIDX[r.label]] = SINGLE_LABEL_NEGATIVE                 # below the lowest base: weak negatives
        rows.append({"sample_id": "eye2sky:" + r.image_file, "dataset": "eye2sky_cbh", "source": "eye2sky", "split": r.split,
                     "unit": r.day, "genus_set": np.zeros(G, np.float32), "genus_alt": False,
                     "genus_neg": np.zeros(G, np.float32),
                     "mask_genus": False, "etage": etage, "etage_w": w, "mask_etage": True,
                     "oktas": np.zeros(N_OKTAS, np.float32),
                     "mask_oktas": False, "calib_id": r.calib_id, "station": r.station, "utc": r.utc, "native_label": r.label,
                     "image_file": r.image_file})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- geometry
@dataclass
class GeometryCache:
    """Ray maps [4, 8, 8] and meta [3] per sample for calibrated Eye2Sky rows (zeros elsewhere), cached on disk."""

    ray_maps: np.ndarray
    metas: np.ndarray

    @classmethod
    def build(cls, rows: pd.DataFrame, data_root: str | Path, cache_file: str | Path | None = None) -> GeometryCache:
        n = len(rows)
        rms = np.zeros((n, 4, SIZE // POOL_PATCH, SIZE // POOL_PATCH), np.float32)
        metas = np.zeros((n, 3), np.float32)
        if cache_file and Path(cache_file).exists():
            z = np.load(cache_file, allow_pickle=True)
            have = dict(zip(z["sample_id"].tolist(), range(len(z["sample_id"])), strict=True))
            idx = [have.get(s, -1) for s in rows.sample_id]
            hit = np.array([i >= 0 for i in idx])
            if hit.all():
                return cls(z["ray_maps"][idx], z["metas"][idx])
        geo = rows[rows.calib_id.notna() & rows.station.notna()]
        meta_root = Path(data_root) / "Eye2Sky" / "asi_meta"
        records = load_pose_records()
        for (station, cid), part in geo.groupby(["station", "calib_id"]):
            cal = next((c for c in station_calibrations(meta_root, station) if c.file == cid), None)
            pose = pose_for(cid, None, records) if cal else None
            if cal is None or pose is None:
                continue
            mask = load_mask(mask_path(meta_root, cal)) > 0
            rays = patch_rays(cal.model, None, SIZE, POOL_PATCH, mask)
            sun = sun_position(part.utc, cal.latitude, cal.longitude, cal.altitude)
            for i, az, zen in zip(part.index, sun.azimuth.to_numpy(), sun.apparent_zenith.to_numpy(), strict=True):
                rms[i] = ray_map(cal.model, pose, float(az), float(zen), out_size=SIZE, patch=POOL_PATCH, rays=rays)
                metas[i] = meta_vector(float(zen))
        if cache_file:
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, sample_id=rows.sample_id.to_numpy(), ray_maps=rms, metas=metas)
        return cls(rms, metas)


# ----------------------------------------------------------------------------------------------- datasets
class GlobalDataset(Dataset):
    """Cached pooled tokens + cls + geometry + global targets for the rows of a table (positions = table order)."""

    def __init__(self, rows: pd.DataFrame, grid_rows: np.ndarray, cls_rows: np.ndarray, grid, cls, geo: GeometryCache):
        self.rows = rows.reset_index(drop=True)
        self.grid_rows, self.cls_rows, self.grid, self.cls, self.geo = grid_rows, cls_rows, grid, cls, geo
        self.genus = np.stack(self.rows.genus_set.to_numpy())
        self.neg = np.stack(self.rows.genus_neg.to_numpy())
        self.etage = np.stack(self.rows.etage.to_numpy())
        self.etage_w = np.stack(self.rows.etage_w.to_numpy())
        self.oktas = np.stack(self.rows.oktas.to_numpy())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows.iloc[i]
        return {"tokens": torch.from_numpy(np.asarray(self.grid[self.grid_rows[i]], dtype=np.float32)),
                "cls": torch.from_numpy(np.asarray(self.cls[self.cls_rows[i]], dtype=np.float32)),
                "ray_map": torch.from_numpy(self.geo.ray_maps[i]), "meta": torch.from_numpy(self.geo.metas[i]),
                "genus_set": torch.from_numpy(self.genus[i]), "genus_alt": bool(r.genus_alt),
                "genus_neg": torch.from_numpy(self.neg[i]), "mask_genus": bool(r.mask_genus),
                "etage": torch.from_numpy(self.etage[i]), "etage_w": torch.from_numpy(self.etage_w[i]),
                "mask_etage": bool(r.mask_etage),
                "oktas": torch.from_numpy(self.oktas[i]), "mask_oktas": bool(r.mask_oktas), "index": i}


def collate(items: list[dict]) -> dict:
    out = {}
    for k in items[0]:
        v = items[0][k]
        if torch.is_tensor(v):
            out[k] = torch.stack([it[k] for it in items])
        elif isinstance(v, bool):
            out[k] = torch.tensor([it[k] for it in items], dtype=torch.bool)
        else:
            out[k] = torch.tensor([it[k] for it in items])
    return out


# ----------------------------------------------------------------------------------------------- native-class scoring
def native_class_sets(source: str, labels) -> dict[str, tuple[list[int], bool]]:
    """For a source's native labels: {label: (genus indices, alternatives)}; labels whose genera are unnamed map to
    an empty list (unscorable), clear classes to [clear]."""
    ont = load_ontology()
    ds = {"swim": "swimcat"}.get(source, source)
    out = {}
    for lab in labels:
        try:
            c = ont.dataset_class(ds, lab)
        except Exception:
            continue
        genera = c.get("genera")
        if genera is None:
            out[lab] = ([], False)
        elif not genera:
            out[lab] = ([GIDX["clear"]] if c.get("extra") == "clear" else [GIDX["contrail"]], False)
        else:
            out[lab] = ([GIDX[g] for g in genera], bool(c.get("alternatives")))
    return out


def native_scores(probs: np.ndarray, class_sets: dict[str, tuple[list[int], bool]]) -> tuple[np.ndarray, list[str]]:
    """Per-native-class score from genus probabilities: noisy-OR over the class's genera (max for a single genus);
    classes with unnamed genera are left out."""
    names = [k for k, (idx, _) in class_sets.items() if idx]
    scores = np.stack([1.0 - np.prod(1.0 - probs[:, class_sets[k][0]], axis=1) for k in names], axis=1)
    return scores, names
