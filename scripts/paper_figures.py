"""P098 — the paper's figures, generated from committed artefacts (no hand-drawn numbers).

    python scripts/paper_figures.py        # writes paper/fig/architecture.png, genus_confusion.png, cbh_scatter.png

architecture.png   the block diagram of Fig. 1 (matplotlib; the only figure with no data behind it);
genus_confusion.png the two P093 confusion heatmaps (docs/data/figures/genus_confusion_{ccsn,mgcd}.png) side by side;
cbh_scatter.png    predicted against ceilometer base height on the validation days of the Gate G4 CBH head
                   (runs/<cbh_base_s0>/val_predictions.npz) with the P089 80 % conformal interval per pair.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stratia.model import cbh_bins as B  # noqa: E402

FIG = ROOT / "paper" / "fig"
CBH_RUN = "20261011T101553Z_cbh_head_s0_[P075_base]_e286609cd2"  # locked_test.MODELS["cbh_base_s0"]

INK = "#1f2a44"
BLUE = "#3b6ea5"
AMBER = "#c77d1c"
GREEN = "#3c8d5a"
GREY = "#8a94a6"


def _box(ax, xy, w, h, text, colour, fontsize=8.5, fc="white"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", linewidth=1.3, edgecolor=colour, facecolor=fc))  # noqa: E501
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK, linespacing=1.25)


def _arrow(ax, p, q, colour=GREY, style="-|>", lw=1.1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, linewidth=lw, color=colour, shrinkA=1, shrinkB=1))


def architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 3.9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.9)
    ax.axis("off")
    # inputs
    _box(ax, (0.15, 2.45), 1.35, 0.9, "sky frame\n512 × 512\n(full-frame resize)", BLUE)
    _box(ax, (0.15, 0.55), 1.35, 1.2, "camera calibration\n+ pose + Sun\n→ ray map 32 × 32\n+ Sun zenith\n(or unknown)", AMBER, fontsize=8)  # noqa: E501
    # backbone
    _box(ax, (1.95, 2.45), 1.75, 0.9, "DINOv3 ViT-S/16\nfrozen\ntokens 32 × 32 × 384", BLUE, fc="#eef3fa")
    _arrow(ax, (1.5, 2.9), (1.95, 2.9))
    # SkyRay
    _box(ax, (1.95, 0.7), 1.75, 0.9, "SkyRay encoder\nray direction, angle to Sun\n→ added to tokens;\ngeometry dropout → null token", AMBER, fontsize=7.8)  # noqa: E501
    _arrow(ax, (1.5, 1.15), (1.95, 1.15), colour=AMBER)
    # pooled grid and full grid
    _box(ax, (4.15, 2.75), 1.35, 0.6, "full grid\n32 × 32", GREY, fontsize=8)
    _box(ax, (4.15, 1.6), 1.35, 0.6, "pooled 4 × 4\n→ 8 × 8", GREY, fontsize=8)
    _arrow(ax, (3.7, 3.0), (4.15, 3.05))
    _arrow(ax, (3.7, 2.7), (4.15, 1.95))
    _arrow(ax, (3.7, 1.15), (4.15, 1.75), colour=AMBER)
    # heads
    hx, hw = 5.85, 2.55
    _box(ax, (hx, 2.85), hw, 0.85, "dense heads (DPT-style decoder, H/4)\nsky · cloud · glare · invalid\ncloud layers low · mid · high", GREEN, fontsize=7.6)  # noqa: E501
    _box(ax, (hx, 1.75), hw, 0.85, "global heads\ngenus (12, multi-label) · étage (3)\noktas 0–8 (CORAL ordinal)", GREEN, fontsize=7.6)  # noqa: E501
    _box(ax, (hx, 0.65), hw, 0.85, "CBH head\nzenith-prior attention over the grid\n24 log-height bins + no cloud", GREEN, fontsize=7.6)  # noqa: E501
    _box(ax, (hx, 0.02), hw, 0.5, "OOD score: Mahalanobis + energy", GREEN, fontsize=7.6)
    _arrow(ax, (5.5, 3.05), (hx, 3.27))
    _arrow(ax, (5.5, 1.9), (hx, 2.17))
    _arrow(ax, (5.5, 1.85), (hx, 1.07))
    _arrow(ax, (5.5, 1.8), (hx, 0.27))
    # outputs
    ox, ow = 8.8, 2.3
    _box(ax, (ox, 2.85), ow, 0.85, "sky parse + layers\n(per pixel)", INK, fontsize=8)
    _box(ax, (ox, 1.75), ow, 0.85, "genus set, étage set,\ncover in oktas", INK, fontsize=8)
    _box(ax, (ox, 0.65), ow, 0.85, "base height: median,\n80 % and 95 % conformal intervals", INK, fontsize=7.6)
    _box(ax, (ox, 0.02), ow, 0.5, "reliable / uncertain", INK, fontsize=8)
    for y in (3.27, 2.17, 1.07, 0.27):
        _arrow(ax, (hx + hw, y), (ox, y))
    ax.set_xlim(0, ox + ow + 0.1)
    ax.text((ox + ow) / 2, 3.8, "one ONNX graph · temperatures, conformal radii and OOD statistics folded in · runs inside the desktop application",  # noqa: E501
            ha="center", va="center", fontsize=7.8, color=GREY, style="italic")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def genus_confusion(out: Path) -> None:
    a = mpimg.imread(ROOT / "docs" / "data" / "figures" / "genus_confusion_ccsn.png")
    b = mpimg.imread(ROOT / "docs" / "data" / "figures" / "genus_confusion_mgcd.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [a.shape[1], b.shape[1]]})
    for ax, im in zip(axes, (a, b), strict=True):
        ax.imshow(im)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def cbh_scatter(out: Path) -> dict[str, float]:
    z = np.load(ROOT / "runs" / CBH_RUN / "val_predictions.npz")
    calib = json.loads((ROOT / "configs" / "calibration.json").read_text(encoding="utf-8"))
    r80 = float(calib["cbh_conformal_radius"]["80"])
    keep = z["cloud"] & (z["height_m"] > 0)
    probs, truth, station = z["probs"][keep], z["height_m"][keep], z["station"][keep]
    conditional = probs.copy()  # the point estimate of a cloudy pair is the median of the height part, as scored in P090
    conditional[:, B.NO_CLOUD] = 0.0
    med = B.median_height_m(conditional)
    lo, hi = B.interval_m(probs, lo_q=max(0.5 - r80, 0.001), hi_q=min(0.5 + r80, 0.999))
    mae = float(np.abs(med - truth).mean())
    cov = float(((truth >= lo) & (truth <= hi)).mean())
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    colours = {s: c for s, c in zip(sorted(set(station.tolist())), (BLUE, AMBER, GREEN, GREY), strict=False)}
    ax.vlines(truth, lo, hi, colors="#cfd4dc", linewidth=0.6, zorder=1)
    for s, c in colours.items():
        m = station == s
        ax.scatter(truth[m], med[m], s=7, color=c, alpha=0.75, linewidths=0, label=f"{s} (n = {int(m.sum())})", zorder=2)
    lim = (100, 13000)
    ax.plot(lim, lim, color=INK, linewidth=0.8, linestyle="--", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ceilometer base height (m)")
    ax.set_ylabel("STRATIA median (m)")
    ax.set_title(f"validation days, {len(truth):,} cloudy pairs; 80 % interval covers {cov:.2f}", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"n": int(keep.sum()), "mae_m": mae, "coverage_80": cov}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    architecture(FIG / "architecture.png")
    genus_confusion(FIG / "genus_confusion.png")
    facts = cbh_scatter(FIG / "cbh_scatter.png")
    print(f"figures -> {FIG}; cbh scatter: {facts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
