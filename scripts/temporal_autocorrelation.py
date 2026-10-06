"""Temporal autocorrelation of the time-series datasets (P027): similarity against time gap -> curves, decisions.

    python scripts/temporal_autocorrelation.py [--per-bin 5000] [--seed 0]

Inputs: data/manifest.parquet (timestamps), cache/features/dinov3_vits16_224_cls.npy and data/perceptual_hashes.parquet
(P025 instruments). Almería's times come from its file names (the manifest has none).
Outputs: data/temporal_curves.parquet (one row per camera and gap bin, plus baseline rows),
docs/data/temporal_report.md, docs/data/figures/temporal_autocorrelation.png.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data import temporal as tp  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402

SERIES = ("eye2sky-AURIC", "eye2sky-BARSE", "montenegro-lowcost", "almeria-Kontas")   # b0268: 25 frames in 5 min, too few

# Reviewed decisions (docs/phases/P027-temporal-autocorrelation.md): the unit a split may not cut through, per series.
BLOCKS = {
    "eye2sky (AURIC, BARSE)": "block = one calendar day, no buffer. Adjacent frames are the same scene (82-92 % at 30 s); "
                              "the same-scene share is below 1 % after 2-4 h; the mean cosine reaches the different-day level "
                              "between 8.5 and 17 h, and the night (10.5 h) lies in a bin already at that level; consecutive "
                              "days at the same hour are no more alike than any two days at the same hour. The sun adds about "
                              "0.02 cosine at the same hour on any day: no split removes it, and it is not leakage.",
    "eye2sky across stations": "same-moment frames at AURIC and BARSE are correlated through the shared sky (0.81 against the "
                               "0.70 different-day level) but are not the same scene (0.4 % above 0.97): a held-out station is a "
                               "legitimate out-of-camera test; holding out its days as well removes the shared-weather term.",
    "montenegro": "block = a contiguous run of at least 7 days; splits take whole blocks. The excess vanishes within a day "
                  "(0.03 at 8.5-17 h) but returns at one day (0.30) and stays at 0.09-0.17 up to six days: consecutive days "
                  "share weather and season, so single-day blocks would leak. Residual correlation between adjacent blocks "
                  "(pairs up to six days apart across a boundary) is accepted and reported; an 11-day buffer would remove it "
                  "at the cost of a sixth of the data.",
    "almeria-Kontas": "block = one calendar day, no buffer: frames come in bursts minutes apart within a day (100 % same scene "
                      "below 2 min), and the excess is 0.05 at one day.",
    "b0268": "25 frames in five minutes: one block; no split until the logger has run for days.",
    "mgcd, ccsn, swim family": "no timestamps: the P025 same-scene groups are the only blocking available.",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-bin", type=int, default=5000, help="pairs sampled per gap bin and camera")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    paths = load_paths()

    m = pd.read_parquet("data/manifest.parquet")
    emb = np.load(Path(paths["cache_root"]) / "features" / "dinov3_vits16_224_cls.npy")
    hashes = pd.read_parquet("data/perceptual_hashes.parquet")
    assert len(emb) == len(m) == len(hashes) and (hashes.image_file.values == m.image_file.values).all()
    ph0 = hashes.phash_0.values.astype(np.uint64)

    times = m.utc.copy()
    almeria = m.dataset == "almeria"
    times[almeria] = tp.times_from_filenames(m.loc[almeria, "image_file"]).dt.tz_localize("UTC").astype(times.dtype)

    results, structures, rows, decisions = {}, {}, [], []
    sorted_index, sorted_t = {}, {}
    for cam in SERIES:
        sel = np.flatnonzero((m.camera_id.values == cam) & times.notna().values)
        t = tp.seconds(times.iloc[sel])
        order = np.argsort(t, kind="stable")
        idx, t = sel[order], t[order]
        sorted_index[cam], sorted_t[cam] = idx, t
        structures[cam] = tp.day_structure(times.iloc[sel])
        e, h = emb[idx], ph0[idx]
        pairs = tp.pair_measures(tp.sample_pairs_by_gap(t, tp.DEFAULT_EDGES, a.per_bin, rng), e, h)
        c = tp.curve(pairs, tp.DEFAULT_EDGES)
        far = tp.pair_measures(tp.sample_pairs_by_gap(t, np.array([tp.DAY, np.inf]), 4 * a.per_bin, rng), e, h)
        same_hour = tp.pair_measures(tp.sample_same_hour_other_days(t, 4 * a.per_bin, rng), e, h)
        baselines = pd.DataFrame([tp.baseline_row(far, "different_day"), tp.baseline_row(same_hour, "same_hour_other_day")])
        d = tp.decide_gap(c, baselines.cos_mean.iloc[0], baselines.same_scene_frac.iloc[0])
        results[cam] = {"curve": c, "baselines": baselines, "decision": d}
        c2 = c.assign(camera=cam, baseline="", excess=d["excess"].to_numpy() if len(d["excess"]) else np.nan)
        rows += [c2, baselines.assign(camera=cam)]
        print(f"{cam}: {structures[cam]['days']} days, {len(pairs):,} pairs; adjacent cosine {c.cos_mean.iloc[0]:.3f}, "
              f"different-day {baselines.cos_mean.iloc[0]:.3f}, same-hour-other-day {baselines.cos_mean.iloc[1]:.3f}; "
              f"gap by excess {tp.fmt_gap(d['gap_by_excess_s'])}, by same-scene {tp.fmt_gap(d['gap_by_same_scene_s'])}",
              flush=True)

    # Eye2Sky across its two stations (15 km apart): how alike are AURIC and BARSE at the same moment?
    ia, ib = sorted_index["eye2sky-AURIC"], sorted_index["eye2sky-BARSE"]
    cross = tp.sample_pairs_by_gap(sorted_t["eye2sky-AURIC"], tp.CROSS_EDGES, a.per_bin, rng, t_b=sorted_t["eye2sky-BARSE"])
    cross = tp.pair_measures(cross, emb[ia], ph0[ia], emb[ib], ph0[ib])
    cc = tp.curve(cross, tp.CROSS_EDGES)
    far_cross = tp.pair_measures(tp.sample_pairs_by_gap(sorted_t["eye2sky-AURIC"], np.array([tp.DAY, np.inf]), 4 * a.per_bin, rng,
                                                        t_b=sorted_t["eye2sky-BARSE"]), emb[ia], ph0[ia], emb[ib], ph0[ib])
    cross_base = pd.DataFrame([tp.baseline_row(far_cross, "different_day")])
    dc = tp.decide_gap(cc, cross_base.cos_mean.iloc[0], cross_base.same_scene_frac.iloc[0])
    name = "eye2sky AURIC x BARSE (cross-station)"
    results[name] = {"curve": cc, "baselines": cross_base, "decision": dc}
    rows += [cc.assign(camera=name, baseline="", excess=dc["excess"].to_numpy() if len(dc["excess"]) else np.nan),
             cross_base.assign(camera=name)]
    print(f"{name}: same-moment cosine {cc.cos_mean.iloc[0]:.3f} (same-scene share {cc.same_scene_frac.iloc[0]:.1%}), "
          f"different-day {cross_base.cos_mean.iloc[0]:.3f}", flush=True)

    curves = pd.concat(rows, ignore_index=True)
    curves.to_parquet("data/temporal_curves.parquet", index=False)

    for cam in SERIES:
        d, s = results[cam]["decision"], structures[cam]
        decisions.append(f"**{cam}**: minimum gap {tp.fmt_gap(d['gap_s'])} (cosine excess: {tp.fmt_gap(d['gap_by_excess_s'])}; "
                         f"same-scene share: {tp.fmt_gap(d['gap_by_same_scene_s'])}); the series has {s['days']} days of "
                         f"{s['frames_per_day_median']:.0f} frames at {tp.fmt_gap(s['step_median_s'])}.")
    decisions += ["Splitting units decided after reviewing the curves:"] + [f"**{k}**: {v}" for k, v in BLOCKS.items()]
    figure = tp.plot_curves(results, "docs/data/figures/temporal_autocorrelation.png")
    report = tp.report_markdown(results, structures, decisions, "figures/temporal_autocorrelation.png")
    Path("docs/data/temporal_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: data/temporal_curves.parquet, docs/data/temporal_report.md, {figure}",
          flush=True)


if __name__ == "__main__":
    main()
