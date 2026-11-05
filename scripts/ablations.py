"""P091 — Ablation table assembled from the Stage H result tables and the P091 runs (validation data only).

Rows: geometry (SkyRay) on / off and the geometry-dropout rate, hard instead of set-valued and soft targets, the
hierarchy loss, uncertainty weighting, the Eye2Sky étage rows, single- vs multi-task training, LoRA vs frozen,
consumer-view synthesis and the regularisation study for the CBH head. Every number is read from
data/*.parquet; nothing is retrained here. Outputs: data/ablations.parquet, docs/data/ablations.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ablations.parquet"
DOC = ROOT / "docs" / "data" / "ablations.md"
METRICS = ["mean_etage_f1", "mean_genus_f1"]


def global_rows(g: pd.DataFrame, name: str, seed: int = 0) -> dict[str, float]:
    part = g[(g.name == name) & (g.seed == seed) & (g.split == "val")]
    if not len(part):
        return {}
    last = part.run_id.max()  # the latest run of that name
    part = part[part.run_id == last]
    out = {}
    for m in METRICS:
        v = part[(part.dataset == "all") & (part.metric == m)].value
        out[m] = float(v.iloc[0]) if len(v) else float("nan")
    for ds, m in (("montenegro", "c2_etage_ratio"), ("montenegro", "c2_oktas_ratio"), ("eye2sky_cbh", "lowest_etage_acc")):
        v = part[(part.dataset == ds) & (part.metric == m)].value
        out[f"{ds}/{m}"] = float(v.iloc[0]) if len(v) else float("nan")
    out["run_id"] = last
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    g = pd.read_parquet(ROOT / "data" / "stage1_global.parquet")
    rows = []
    base = global_rows(g, "indomain_e2s")
    seeds = [global_rows(g, n) for n in ("indomain_e2s", "indomain_e2s_s1", "indomain_e2s_s2")]
    seeds = [s for s in seeds if s]
    std = {m: float(np.std([s[m] for s in seeds])) for m in METRICS} if len(seeds) == 3 else {}

    def add(group, variant, r, note=""):
        rows.append({"group": group, "variant": variant, **{k: v for k, v in r.items()}, "note": note})

    add(
        "reference",
        "Gate G4 heads (geometry dropout 0.3, soft and set-valued targets, hierarchy loss 0.1, Eye2Sky étage rows)",
        base,
        f"seed std over 3 seeds: étage {std.get('mean_etage_f1', float('nan')):.4f}, genus {std.get('mean_genus_f1', float('nan')):.4f}",  # noqa: E501
    )
    add("geometry (SkyRay)", "no geometry: ray map and Sun always dropped (p_drop 1.0)", global_rows(g, "abl_nogeo"))
    add("geometry (SkyRay)", "geometry always given (p_drop 0.0)", global_rows(g, "abl_geoalways"))
    add("targets", "hard targets: first listed genus, majority étages, majority okta", global_rows(g, "abl_hard"))
    add("loss", "no hierarchy loss (lambda_h 0)", global_rows(g, "abl_noh"))
    add("loss", "uncertainty (Kendall) task weighting", global_rows(g, "abl_kendall"))
    add("Eye2Sky étage rows", "without the ceilometer étage rows (indomain)", global_rows(g, "indomain"))
    for fold in ("ccsn", "mgcd", "montenegro", "almeria"):
        with_e = global_rows(g, f"lodo_{fold}_e2s")
        without = global_rows(g, f"lodo_{fold}")
        for lab, r in (("with Eye2Sky étage rows", with_e), ("without", without)):
            part = g[
                (g.name == (f"lodo_{fold}_e2s" if lab.startswith("with ") else f"lodo_{fold}"))
                & (g.split == "val")
                & (g.dataset == fold)
                & (g.metric == "etage_f1")
            ]
            v = float(part[part.run_id == part.run_id.max()].value.iloc[0]) if len(part) else float("nan")
            rows.append(
                {
                    "group": "Eye2Sky étage rows (LODO)",
                    "variant": f"held-out {fold}, {lab}",
                    "held_out_etage_f1": v,
                    "run_id": r.get("run_id", ""),
                }
            )
    # regularisation study (P080) rows exist under reg_*: dropout, weight decay, EMA
    for name, lab in (
        ("reg_do0", "head dropout 0"),
        ("reg_do03", "head dropout 0.3"),
        ("reg_wd0", "weight decay 0"),
        ("reg_wd02", "weight decay 0.2"),
        ("reg_ema0", "no EMA"),
        ("reg_ema999", "EMA 0.999"),
    ):
        add("regularisation (P080)", lab, global_rows(g, name))
    # single- vs multi-task (P076) and LoRA (P077)
    try:
        j = pd.read_parquet(ROOT / "data" / "stage4_joint.parquet")
        for name in sorted(j.name.unique()):
            part = j[(j.name == name)]
            r = {
                m: float(part[part.metric.str.endswith(m)].value.iloc[0])
                for m in ("mean_etage_f1", "sky_miou", "cbh_mae_m", "layer_miou")
                if part.metric.str.endswith(m).any()
            }
            rows.append({"group": "multi-task (P076)", "variant": f"joint training: {name}", **r, "run_id": part.run_id.iloc[0]})
    except (OSError, KeyError):
        pass
    try:
        lo = pd.read_parquet(ROOT / "data" / "stage5_lora.parquet")
        for name in sorted(lo.name.unique()):
            part = lo[lo.name == name]
            r = {}
            for m in ("mean_etage_f1", "mean_genus_f1"):
                p = part[part.metric.str.endswith(m)]
                if len(p):
                    r[m] = float(p.after.iloc[0])
                    r[f"{m}_before"] = float(p.before.iloc[0])
            rows.append(
                {
                    "group": "partial unfreeze (P077)",
                    "variant": f"LoRA: {name} (after / before)",
                    **r,
                    "run_id": part.run_id.iloc[0],
                }
            )
    except (OSError, KeyError):
        pass
    # CBH head variants (P075) incl. consumer-view synthesis and geometry
    c = pd.read_parquet(ROOT / "data" / "cbh_head.parquet")
    for variant in c.variant.unique():
        part = c[c.variant == variant]
        part = part[part.seed == 0] if (part.seed == 0).any() else part
        r = part.iloc[-1]
        rows.append(
            {
                "group": "CBH head (P075)",
                "variant": variant,
                "cbh_mae_m": float(r.mae_m),
                "cbh_skill": float(r.skill),
                "cbh_coverage_80": float(r.coverage_80),
                "run_id": r.run_id,
            }
        )
    # dense heads: sky only vs with layers vs Kendall
    d = pd.read_parquet(ROOT / "data" / "dense_heads.parquet")
    for model_name in d.model.unique():
        part = d[(d.model == model_name) & (d.dataset == "all")].iloc[-1]
        rows.append(
            {
                "group": "dense heads (P074)",
                "variant": model_name,
                "sky_miou": float(part.miou),
                "layer_miou": float(part.layer_miou)
                if "layer_miou" in d.columns and part.layer_miou == part.layer_miou
                else float("nan"),
                "clear_sky_cloud_fraction": float(part.clear_sky_cloud_fraction),
                "run_id": part.run_id,
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    write_doc(df, base, std)
    print(f"{len(df)} ablation rows -> {OUT}, {DOC}")
    return 0


def _f(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def write_doc(df: pd.DataFrame, base: dict, std: dict) -> None:
    lines = [
        "# Ablations (P091, validation data)",
        "",
        "Every row is a validation number read from the Stage H result tables or the five P091 runs; the reference "
        "is the Gate G4 configuration. Global rows: mean étage macro-F1 over the sources and mean genus macro-F1 "
        "(CCSN, MGCD, SWIMCAT), the Montenegro C2 ratios and the Eye2Sky lowest-étage accuracy; one seed each, the "
        f"reference's spread over three seeds is étage ± {std.get('mean_etage_f1', float('nan')):.3f}, genus ± "
        f"{std.get('mean_genus_f1', float('nan')):.3f}. Backbone size was not ablated: only 224-px CLS caches exist for "
        "ViT-B/L (the P054 probes stand in); augmentation does not apply to heads trained on cached features (P080 "
        "covers regularisation). Not retrained here; the locked test (P090) was read once and is not touched.",
        "",
    ]
    glob = df[df.group.isin(["reference", "geometry (SkyRay)", "targets", "loss", "Eye2Sky étage rows", "regularisation (P080)"])]
    lines += [
        "## Global heads",
        "",
        "| Group | Variant | Étage F1 (Δ) | Genus F1 (Δ) | C2 étage ratio | C2 oktas ratio | Eye2Sky lowest-étage acc | Run |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in glob.itertuples(index=False):
        de = r.mean_etage_f1 - base.get("mean_etage_f1", float("nan"))
        dg = r.mean_genus_f1 - base.get("mean_genus_f1", float("nan"))
        lines.append(
            f"| {r.group} | {r.variant} | {_f(r.mean_etage_f1)} ({de:+.3f}) | {_f(r.mean_genus_f1)} ({dg:+.3f}) | "
            f"{_f(getattr(r, 'montenegro/c2_etage_ratio', float('nan')))} | {_f(getattr(r, 'montenegro/c2_oktas_ratio', float('nan')))} | "  # noqa: E501
            f"{_f(getattr(r, 'eye2sky_cbh/lowest_etage_acc', float('nan')))} | `{str(r.run_id)[:16]}` |"
        )
    lo = df[df.group == "Eye2Sky étage rows (LODO)"]
    if len(lo):
        lines += ["", "| LODO fold | Variant | Held-out étage F1 |", "|---|---|---|"]
        for r in lo.itertuples(index=False):
            lines.append(
                f"| {r.variant.split(',')[0]} | {','.join(r.variant.split(',')[1:]).strip()} | {_f(r.held_out_etage_f1)} |"
            )
    for group, cols in (
        ("multi-task (P076)", ["mean_etage_f1", "sky_miou", "layer_miou", "cbh_mae_m"]),
        ("partial unfreeze (P077)", ["mean_etage_f1", "mean_etage_f1_before", "mean_genus_f1", "mean_genus_f1_before"]),
        ("CBH head (P075)", ["cbh_mae_m", "cbh_skill", "cbh_coverage_80"]),
        ("dense heads (P074)", ["sky_miou", "layer_miou", "clear_sky_cloud_fraction"]),
    ):
        part = df[df.group == group]
        if not len(part):
            continue
        cols = [c for c in cols if c in part.columns]
        lines += ["", f"## {group}", "", "| Variant | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for r in part.itertuples(index=False):
            lines.append(f"| {r.variant} | " + " | ".join(_f(getattr(r, c)) for c in cols) + " |")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
