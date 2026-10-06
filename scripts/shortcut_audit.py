"""Shortcut audit (P029): camera statistics, image variants, linear probes -> report and shortcut list.

    python scripts/shortcut_audit.py [--per-class 1500] [--low 96] [--rel 0.5] [--workers 8]

Stages, each cached so a re-run is cheap:
  1. camera statistics   cache/features/camera_stats_224.npz          mean, std per camera with >= 500 frames
  2. variants            cache/features/dinov3_vits16_224_cls_<variant>.npy
                         lowres (all images), skyonly and fixedonly (Eye2Sky, Montenegro)
  3. probes              data/shortcut_probes.parquet
  4. report              docs/data/shortcuts_report.md, docs/data/figures/shortcut_camera_statistics.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data import shortcuts as sc  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402

MASKED_CAMERAS = ("eye2sky-AURIC", "eye2sky-BARSE", "montenegro-lowcost")
MIN_FRAMES_FOR_STATS = 500

SHORTCUTS = [
    {"shortcut": "Dataset and camera identity (optics, site, processing, colour response)",
     "evidence": "dataset probe 99.7 % balanced (chance 9 %); SWIMSEG vs SWINySEG, one camera, 99.7 %; Eye2Sky station "
                 "100 % from the sky alone (`skyonly`) on unseen days, 99.2 % after down-sampling: masking and resolution "
                 "normalisation do not remove it",
     "mitigation": "leave-one-dataset-out and held-out-station evaluation (P037); camera-balanced sampling (P030, P039); "
                   "per-source numbers next to every pooled one"},
    {"shortcut": "Resolution and sharpness (125 px patches to 2,112 px all-sky frames, JPEG against PNG)",
     "evidence": "`lowres` (96 px) lowers the dataset probe by 0.7 points and the station probe by 0.7: a small part of "
                 "the identity signal",
     "mitigation": "resolution normalisation in training: random down-and-up scaling and JPEG re-encoding (P040); one input "
                   "size; results per source resolution"},
    {"shortcut": "Fixed structure: fisheye corners, horizon objects, mounting parts",
     "evidence": "25-37 % of an all-sky frame (Eye2Sky, MGCD, Almería); station 100 % and hour 43 % from `fixedonly`",
     "mitigation": "per-camera validity mask from the std image (`cache/features/camera_stats_224.npz`) at load time: ignore "
                   "for segmentation, grey or erased for classification (P040)"},
    {"shortcut": "Burned-in text and logos (Eye2Sky top-left block; Montenegro timestamp and logo)",
     "evidence": "inside the masks; Montenegro `fixedonly` predicts the hour at 41 % (chance 7 %) and the class at 35 % "
                 "balanced (chance 20 %)",
     "mitigation": "keep the text regions masked in training and evaluation (part of the per-camera mask, P040)"},
    {"shortcut": "Time of day and sun position",
     "evidence": "hour probes 27-33 % from the sky, 42-44 % from the fixed structure (chance 7 %); P027 same-hour similarity",
     "mitigation": "the sun as an explicit input (ray map, P044); results by sun-zenith bin (P030); day blocks (P027)"},
    {"shortcut": "Colour cast and exposure per camera",
     "evidence": "the mean images differ in tint and brightness between cameras; exposure state carries the hour",
     "mitigation": "colour jitter and per-image normalisation (P040); grey-world check per camera in the data card (P032)"},
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=1500, help="cap per class for the dataset-ID probe")
    ap.add_argument("--low", type=int, default=96, help="pixels on a side for the lowres variant")
    ap.add_argument("--rel", type=float, default=0.5, help="fixed pixel: std below this fraction of the camera median")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    paths = load_paths()
    cache_root = Path(paths["cache_root"])
    feats = cache_root / "features"

    m = pd.read_parquet("data/manifest.parquet")
    files = m.image_file.tolist()
    cams = m.camera_id.values
    base = np.load(feats / "dinov3_vits16_224_cls.npy")
    assert len(base) == len(m)

    # ---- 1. camera statistics --------------------------------------------------------------------------------------
    stats_path = feats / "camera_stats_224.npz"
    counts = m.camera_id.value_counts()
    big = [c for c in counts.index if counts[c] >= MIN_FRAMES_FOR_STATS]
    if stats_path.exists():
        z = np.load(stats_path)
        stats = {c: (z[f"{c}_mean"], z[f"{c}_std"]) for c in big if f"{c}_mean" in z}
    else:
        stats = {}
    for c in big:
        if c not in stats:
            t1 = time.perf_counter()
            stats[c] = sc.camera_statistics([f for f, cam in zip(files, cams, strict=True) if cam == c], cache_root,
                                            workers=a.workers)
            print(f"statistics {c}: {counts[c]:,} frames in {time.perf_counter() - t1:.0f} s", flush=True)
    np.savez(stats_path, **{f"{c}_{k}": v for c, (mean, std) in stats.items() for k, v in (("mean", mean), ("std", std))})
    masks = {c: sc.fixed_mask(std, a.rel) for c, (mean, std) in stats.items()}
    fixed_share = pd.DataFrame([{"camera": c, "frames": int(counts[c]), "fixed_share": float(masks[c].mean())} for c in big])
    print(fixed_share.to_string(index=False), flush=True)
    figure = sc.statistics_figure({c: (stats[c][0], stats[c][1], masks[c]) for c in big},
                                  "docs/data/figures/shortcut_camera_statistics.jpg")

    # ---- 2. variants ------------------------------------------------------------------------------------------------
    variants = {"original": base}
    t1 = time.perf_counter()
    variants["lowres"] = sc.embed(sc.VariantDataset(files, cache_root, "lowres", low=a.low),
                                  feats / f"dinov3_vits16_224_cls_lowres{a.low}.npy", workers=a.workers)
    print(f"lowres embeddings: {time.perf_counter() - t1:.0f} s", flush=True)
    masked_rows = np.flatnonzero(np.isin(cams, MASKED_CAMERAS))
    for variant in ("skyonly", "fixedonly"):
        t1 = time.perf_counter()
        ds = sc.VariantDataset([files[i] for i in masked_rows], cache_root, variant, masks=masks,
                               cameras=[cams[i] for i in masked_rows])
        part = sc.embed(ds, feats / f"dinov3_vits16_224_cls_{variant}.npy", workers=a.workers)
        full = np.zeros_like(base)
        full[masked_rows] = part
        variants[variant] = full
        print(f"{variant} embeddings ({len(masked_rows):,} frames): {time.perf_counter() - t1:.0f} s", flush=True)

    # ---- 3. probes --------------------------------------------------------------------------------------------------
    ndg = pd.read_parquet("data/near_duplicate_groups.parquet")
    dg = pd.read_parquet("data/duplicate_groups.parquet")
    group = pd.Series(m.sample_id.values, index=m.index, dtype=object)
    group[m.sample_id.isin(dg.sample_id)] = m.sample_id.map(dg.set_index("sample_id").dup_group.astype(str))
    key = "sample_id" if "sample_id" in ndg else "image_file"
    ndg_map = ndg.set_index(key).near_dup_group.astype(str)
    has_nd = m[key].isin(ndg_map.index)
    group[has_nd] = m.loc[has_nd, key].map(ndg_map)
    utc = m.utc.dt.tz_convert(None)
    day = utc.dt.strftime("%Y-%m-%d")
    is_series = m.dataset.isin(["eye2sky", "montenegro", "almeria", "b0268"]) & day.notna()
    group[is_series] = m.dataset[is_series] + ":" + day[is_series]
    week = ((utc - utc.min()).dt.days // 7)

    def run(task: str, rows: np.ndarray, labels: np.ndarray, test: np.ndarray, names: tuple[str, ...]) -> None:
        for v in names:
            r = sc.linear_probe(variants[v][rows], labels, test)
            r.update({"task": task, "variant": v})
            results.append(r)
            print(f"  {task:40s} {v:10s} acc {r['accuracy']:.3f}  balanced {r['balanced_accuracy']:.3f}  "
                  f"chance {r['chance']:.3f}  majority {r['majority']:.3f}  (train {r['n_train']:,} / test {r['n_test']:,})",
                  flush=True)

    results: list[dict] = []
    # dataset identity, balanced
    rows = sc.balanced_subsample(m.dataset.values, a.per_class, rng)
    run("dataset identity (11 datasets)", rows, m.dataset.values[rows], sc.holdout_random_groups(group.iloc[rows]),
        ("original", "lowres"))
    # the same pictures at two resolutions: SWIMSEG (600 px PNG) against SWINySEG daytime copies (300 px JPEG)
    swim = np.flatnonzero((m.dataset.values == "swimseg")
                          | ((m.dataset.values == "swinyseg") & m.image_file.str.contains("/d", regex=False).values))
    run("SWIMSEG vs SWINySEG-day (same camera)", swim, m.dataset.values[swim], sc.holdout_random_groups(group.iloc[swim]),
        ("original", "lowres"))
    # station identity: AURIC vs BARSE, split by day
    e2s = np.flatnonzero(m.dataset.values == "eye2sky")
    e2s_test = sc.holdout_last_blocks(day.iloc[e2s])
    run("Eye2Sky station (AURIC vs BARSE)", e2s, m.camera_id.values[e2s], e2s_test,
        ("original", "skyonly", "fixedonly", "lowres"))
    # hour of day
    run("Eye2Sky hour of day (UTC)", e2s, utc.dt.hour.values[e2s].astype(int), e2s_test, ("original", "skyonly", "fixedonly"))
    mon = np.flatnonzero(m.dataset.values == "montenegro")
    mon_test = sc.holdout_last_blocks(week.iloc[mon])
    run("Montenegro hour of day (UTC)", mon, utc.dt.hour.values[mon].astype(int), mon_test, ("original", "skyonly", "fixedonly"))
    # labels, for scale
    mon_label = m.source_label.str.split(",").str[0].values
    run("Montenegro primary class (5)", mon, mon_label[mon], mon_test, ("original", "skyonly", "fixedonly"))
    for ds, task in (("mgcd", "MGCD sky type (7)"), ("ccsn", "CCSN genus (11)"), ("swimcat", "SWIMCAT category (5)")):
        rows = np.flatnonzero(m.dataset.values == ds)
        run(task, rows, m.source_label.values[rows], sc.holdout_random_groups(group.iloc[rows]), ("original", "lowres"))

    probes = pd.DataFrame(results)[["task", "variant", "classes", "n_train", "n_test", "accuracy", "balanced_accuracy",
                                    "chance", "majority"]]
    probes.to_parquet("data/shortcut_probes.parquet", index=False)

    # ---- 4. report --------------------------------------------------------------------------------------------------
    report = sc.report_markdown(probes, fixed_share, SHORTCUTS, "figures/shortcut_camera_statistics.jpg")
    Path("docs/data/shortcuts_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: data/shortcut_probes.parquet, docs/data/shortcuts_report.md, {figure}",
          flush=True)


if __name__ == "__main__":
    main()
