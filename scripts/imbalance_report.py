"""Imbalance report over the manifest and the ceilometer records (P030).

    python scripts/imbalance_report.py [--workers 8]

Outputs: data/imbalance_tables.parquet (every distribution as rows), data/mask_cloud_fraction.parquet (per
segmentation image), docs/data/imbalance_report.md, docs/data/figures/imbalance.png. The ceilometer records are
cached at data/ceilometer_records.parquet (shared with P031).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data import imbalance as ib  # noqa: E402
from stratia.data.ceilometer import load_all  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402

DECISION = [
    "**Sources are sampled in proportion to the square root of their size** (table above). Natural sampling would make "
    "Eye2Sky 56 % of every epoch and the five smallest sources under 2 % together; uniform sampling would show the 25 "
    "B0268 frames and the 115 SWINSEG images thousands of times per epoch. The square root keeps the largest source "
    "near a third and lifts the small ones to a few percent. Rejected: natural (majority-camera collapse, the P029 "
    "shortcut) and uniform (over-fitting of tiny sources).",
    "**Within a source, classes are weighted in the loss by the class-balanced rule** (1 - beta) / (1 - beta^n) with "
    "beta = 0.999, which approaches uniform weighting for rare classes and natural weighting for common ones; CCSN's "
    "11 genera (ratio 2.4) and MGCD's 7 types (ratio 2.0) need it mildly, the ceilometer bins (none / low / mid / high) "
    "more. Rejected: class-uniform resampling, which duplicates rare images and leaks through near-duplicates.",
    "**Soft labels are never resampled**: Montenegro's rater distributions and merged conflict sets are used as "
    "targets as they are; balancing acts through the loss weight only.",
    "**Metrics are macro by default** (macro-F1 over genera and étages, balanced accuracy for oktas and CBH bins), "
    "reported next to the micro numbers, so a majority-class collapse is visible.",
    "**Evaluation is also reported by sun-zenith bin** (Eye2Sky, B0268) and, for the segmentation sets, by cloud-fraction "
    "bin, because P029 showed the hour of day is readable from the features and the overcast/clear extremes dominate "
    "several sets.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.perf_counter()
    paths = load_paths()
    data_root = Path(paths["data_root"])
    m = pd.read_parquet("data/manifest.parquet")
    sections: list[tuple[str, str, pd.DataFrame]] = []
    panels: list[tuple[str, pd.DataFrame]] = []
    rows = []

    def add(title: str, note: str, dist: pd.DataFrame, panel: bool = True, dataset: str = "", quantity: str = "") -> None:
        sections.append((title, note, dist))
        if panel:
            panels.append((title, dist))
        for _, r in dist.iterrows():
            rows.append({"dataset": dataset, "quantity": quantity, "value": r.value, "count": int(r["count"]), "share": r.share})

    # ---- native classes ------------------------------------------------------------------------------------------
    for ds, title in (("ccsn", "CCSN genus (11)"), ("mgcd", "MGCD sky type (7)"), ("swimcat", "SWIMCAT category (5)")):
        part = m[m.dataset == ds]
        add(title, "Native labels, all images.", ib.distribution(part.source_label), dataset=ds, quantity="class")
        if (part.official_split != "none").any():
            for split, sp in part.groupby("official_split"):
                add(f"{title}, official {split}", "Native labels in the published split.",
                    ib.distribution(sp.source_label), panel=False, dataset=ds, quantity=f"class:{split}")
    mon = m[m.dataset == "montenegro"]
    primary = mon.source_label.str.split(",").str[0]
    add("Montenegro primary altitude class (5)", "First class of the rater majority.", ib.distribution(primary),
        dataset="montenegro", quantity="class")
    etage_tokens = mon.source_label.str.split(",").explode()
    add("Montenegro altitude classes present (multi-label)",
        "Every class named in the majority set, counted once per image.",
        ib.distribution(etage_tokens), panel=False, dataset="montenegro", quantity="etage_present")
    def majority(dist_json: pd.Series) -> pd.Series:
        """The code with the largest share in a JSON distribution such as {"6": 0.56, "7": 0.44}."""
        return dist_json.map(lambda s: max(json.loads(s).items(), key=lambda kv: kv[1])[0] if isinstance(s, str) else np.nan)

    okt = majority(mon.oktas_dist)
    add("Montenegro total cloud cover N (oktas, majority)", "Majority rater code; 9 = sky obscured.",
        ib.distribution(okt, order=[str(k) for k in range(10)]), dataset="montenegro", quantity="oktas")
    h = majority(mon.h_dist)
    add("Montenegro cloud-base code h (majority)",
        "WMO code table 1600 (0 = lowest band ... 9 = above 2,500 m or no cloud; / = unknown).",
        ib.distribution(h), dataset="montenegro", quantity="h_code")

    # ---- segmentation masks ---------------------------------------------------------------------------------------
    seg = m[m.seg_file.notna()]
    with ProcessPoolExecutor(a.workers) as ex:
        jobs = [(d, str(data_root / f)) for d, f in zip(seg.dataset, seg.seg_file, strict=True)]
        res = list(ex.map(ib.mask_cloud_fraction, jobs, chunksize=64))
    frac = pd.DataFrame(res)
    frac.insert(0, "image_file", seg.image_file.values)
    frac.insert(0, "dataset", seg.dataset.values)
    frac.to_parquet("data/mask_cloud_fraction.parquet", index=False)
    for ds, part in frac.groupby("dataset"):
        pixel = pd.Series({"sky": 1 - part.cloud_fraction.mean(), "cloud": part.cloud_fraction.mean()})
        pix = pd.DataFrame({"value": pixel.index, "count": (pixel.values * 1000).round().astype(int), "share": pixel.values})
        add(f"{ds} mask pixels (sky / cloud, mean over images)", "Share of labelled sky pixels.", pix, panel=False,
            dataset=ds, quantity="pixels")
        add(f"{ds} cloud fraction per image", "Share of each image's labelled sky that is cloud.",
            ib.distribution(ib.cloud_fraction_bin(part.cloud_fraction), order=list(ib.CLOUD_FRACTION_BINS)),
            panel=ds in {"swinyseg", "almeria"}, dataset=ds, quantity="cloud_fraction")
    al = frac[frac.dataset == "almeria"]
    layers = pd.Series({"low": al.low.mean(), "mid": al.mid.mean(), "high": al.high.mean()})
    add("Almería layer pixels (share of labelled sky, mean over images)", "Low / mid / high cloud layer classes.",
        pd.DataFrame({"value": layers.index, "count": (layers.values * 1000).round().astype(int),
                      "share": layers.values}),
        dataset="almeria", quantity="layer_pixels")

    # ---- sun zenith -----------------------------------------------------------------------------------------------
    for ds in ("eye2sky", "b0268"):
        part = m[m.dataset == ds]
        add(f"{ds} sun-zenith bins (degrees)", "From time and site (P021).",
            ib.distribution(ib.sun_bin(part.sun_zenith_deg), order=list(ib.SUN_BINS)), panel=ds == "eye2sky",
            dataset=ds, quantity="sun_zenith")
    add("Montenegro hour (UTC)", "Hour of the frame.", ib.distribution(mon.utc.dt.hour), panel=False,
        dataset="montenegro", quantity="hour")

    # ---- ceilometer cloud-base bins -------------------------------------------------------------------------------
    base = data_root / "Eye2Sky" / "ceilometer" / "ceil" / "data"
    cl = load_all(base, "data/ceilometer_records.parquet")
    for code, part in cl.groupby("ceilometer"):
        ok = part[~part.qc_any]
        add(f"Ceilometer {code} lowest cloud base (QC-clean records)",
            "none = no cloud overhead; low < 2 km, mid 2-6 km, high > 6 km.",
            ib.distribution(ib.cbh_bin(ok.cbh1.values), order=["none", *ib.CBH_BINS]), dataset=f"ceilometer-{code}",
            quantity="cbh_bin")
        add(f"Ceilometer {code} layer count", "Number of cloud layers reported (QC-clean records).",
            ib.distribution(ok.n_layers, order=[0, 1, 2, 3, 4]), panel=False, dataset=f"ceilometer-{code}", quantity="layers")

    # ---- sources and sampling -------------------------------------------------------------------------------------
    source = m.dataset.replace({"swimseg": "swim-family", "swinseg": "swim-family", "swinyseg": "swim-family",
                                "swimcat": "swim-family", "shwimseg": "swim-family"})
    sources = ib.source_weights(source.value_counts().to_dict(), power=0.5).sort_values("images", ascending=False)
    cb = {}
    for ds, name in (("ccsn", "CCSN"), ("mgcd", "MGCD")):
        d = ib.distribution(m[m.dataset == ds].source_label)
        cb[name] = pd.DataFrame({"class": d.value, "images": d["count"],
                                 "weight": ib.class_balanced_weights(d["count"].values)})
    cbh_dist = ib.distribution(ib.cbh_bin(cl[~cl.qc_any].cbh1.values), order=["none", *ib.CBH_BINS])
    cb["ceilometer bins (both sites)"] = pd.DataFrame({"class": cbh_dist.value, "images": cbh_dist["count"],
                                                       "weight": ib.class_balanced_weights(cbh_dist["count"].values)})

    pd.DataFrame(rows).to_parquet("data/imbalance_tables.parquet", index=False)
    fig = ib.figure(panels, "docs/data/figures/imbalance.png")
    report = ib.report_markdown(sections, sources, cb, DECISION, "figures/imbalance.png")
    Path("docs/data/imbalance_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(sources.to_string(index=False), flush=True)
    for title, _, dist in sections:
        print(f"{title}: {ib.metrics_line(dist)}", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f} s: data/imbalance_tables.parquet, docs/data/imbalance_report.md, {fig}",
          flush=True)


if __name__ == "__main__":
    main()
