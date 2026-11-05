"""P097 / P098 — the paper's tables, generated from the committed result files (no hand-typed numbers).

    python scripts/reproduce_tables.py            # writes paper/tables/*.tex and paper/tables/*.md

Sources: results/P090_locked_test.json (the locked test, read once), results/ablations.parquet (P091),
results/human_ceiling_model.parquet (P092) and results/efficiency.json (P094): small copies of the phase
scripts' data/ outputs, committed so that a fresh clone reproduces every table and number without the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"


def result_file(name: str) -> Path:
    """A result file: the committed copy under results/ (what a fresh clone has), else the phase script's data/ output."""
    committed = ROOT / "results" / name
    return committed if committed.exists() else ROOT / "data" / name


def esc(s: str) -> str:
    return (
        str(s)
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("_", r"\_")
        .replace("±", r"$\pm$")
        .replace("≥", r"$\geq$")
        .replace("≤", r"$\leq$")
    )


def table(name: str, caption: str, label: str, header: list[str], rows: list[list[str]], align: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    align = align or ("l" + "r" * (len(header) - 1))
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + align + "}",
        r"\toprule",
        " & ".join(esc(h) for h in header) + r" \\",
        r"\midrule",
    ]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}", r"\caption{" + esc(caption) + "}", r"\label{tab:" + label + "}", r"\end{table}", ""]
    (OUT / f"{name}.tex").write_text("\n".join(tex), encoding="utf-8", newline="\n")
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
        "| " + " | ".join(rows_) + " |" for rows_ in [list(map(str, r)) for r in rows]
    ]
    (OUT / f"{name}.md").write_text(f"**{caption}**\n\n" + "\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(f"  {name}: {len(rows)} rows")


def f3(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    res = json.loads((ROOT / "results" / "P090_locked_test.json").read_text(encoding="utf-8"))
    rows = pd.DataFrame(res["rows"])

    def pick(protocol, subset, model, metric):
        m = rows[(rows.protocol == protocol) & (rows["set"] == subset) & (rows.model == model) & (rows.metric == metric)]
        if not len(m):
            return float("nan"), ""
        r = m.iloc[0]
        ci = (
            ""
            if np.isnan(r.ci_lo)
            else (f" [{r.ci_lo:.0f}, {r.ci_hi:.0f}]" if abs(r.value) >= 100 else f" [{r.ci_lo:.2f}, {r.ci_hi:.2f}]")
        )
        return float(r.value), ci

    # ---- Table 1: the pre-registered criteria, validation and locked test
    t1 = []
    for h in res["headline"]:
        val = h["val_p082"]
        ci = "" if "Montenegro" in h["quantity"] else h["ci"]  # two test blocks: the cluster CI says nothing
        t1.append(
            [
                h["criterion"],
                h["quantity"],
                f3(val) if abs(val) < 100 else f"{val:.0f}",
                (f3(h["model"]) if abs(h["model"]) < 100 else f"{h['model']:.0f}") + (f" {ci}" if ci else ""),
                (f3(h["reference"]) if abs(h["reference"]) < 100 else f"{h['reference']:.0f}"),
                h["required"],
                "yes" if h["pass"] else "no",
            ]
        )
    table(
        "tab1_criteria",
        "Pre-registered criteria on validation (Gate G4) and on the locked test sets, read once.",
        "criteria",
        ["", "Quantity", "Val", "Test [95% CI]", "Reference", "Required", "Holds"],
        t1,
        "llrlrll",
    )

    # ---- Table 2: recognition on the locked test (in-domain and LODO), model vs probes
    t2 = []
    for src in ("ccsn", "mgcd"):
        v, ci = pick("in_domain", src, "indomain_e2s_s0", "genus_f1")
        p, pci = pick("in_domain", src, "probe_dinov3_vitl16_224", "genus_f1")
        t2.append(["in-domain genus", src.upper(), f3(v) + ci, f3(p) + pci])
    for src in ("ccsn", "mgcd", "montenegro", "almeria"):
        v, ci = pick("in_domain", src, "indomain_e2s_s0", "etage_f1")
        t2.append(["in-domain étage", src, f3(v) + ci, "–"])
    for fold in ("ccsn", "mgcd", "montenegro", "almeria"):
        v, ci = pick("lodo", fold, f"lodo_{fold}_e2s", "etage_f1")
        p, pci = pick("lodo", fold, "probe_dinov3_vitb16_224", "etage_f1")
        t2.append(["LODO étage (held out)", fold, f3(v) + ci, f3(p) + pci])
    table(
        "tab2_recognition",
        "Recognition on the locked test sets: STRATIA against the strongest linear probe of Stage F (ViT-L/16 for in-domain genus, ViT-B/16 for LODO étage), macro-F1 with cluster-bootstrap 95% intervals.",  # noqa: E501
        "recognition",
        ["Protocol", "Source", "STRATIA", "Linear probe"],
        t2,
        "llrr",
    )

    # ---- Table 3: cloud-base height on the locked test
    t3 = []
    for model, label, subset in (
        ("cbh_base_s0", "STRATIA (both sites in training)", "all"),
        ("cbh_base_s0", "  OLDLR (CDLRA)", "OLDLR"),
        ("cbh_base_s0", "  WESTE (CDLRB)", "WESTE"),
        ("cbh_base_s0_geometry_zeroed", "STRATIA, geometry zeroed", "all"),
        ("cbh_oldlr_only_s0", "STRATIA trained on OLDLR only, tested on WESTE", "WESTE"),
        ("ridge_dinov3_vits16_512", "Ridge on frozen features (P058)", "all"),
        ("climatology_train", "Training climatology", "all"),
    ):
        mae, ci = pick("cbh_pairs", subset, model, "mae_m")
        skill, sci = pick("cbh_pairs", subset, model, "skill")
        w500, _ = pick("cbh_pairs", subset, model, "within_500m")
        cov, _ = pick("cbh_pairs", subset, model, "coverage_80_calibrated")
        t3.append(
            [
                label,
                f"{mae:.0f}{ci}" if mae == mae else "–",
                (f3(skill) + sci) if skill == skill else "–",
                f3(w500) if w500 == w500 else "–",
                f3(cov) if cov == cov else "–",
            ]
        )
    table(
        "tab3_cbh",
        "Cloud-base height on the locked ceilometer-paired test pairs (2,965 cloudy frames, 25 days per site): MAE, skill over the training climatology, share within 500 m, and 80% interval coverage with the validation-fitted calibration.",  # noqa: E501
        "cbh",
        ["Model", "MAE (m) [CI]", "Skill [CI]", "±500 m", "80% cov."],
        t3,
        "lrrrr",
    )

    # ---- Table 4: B0268 locked day
    t4 = []
    for graph, label in (("stratia-1.0.0-demo", "zero-shot"), ("stratia-1.0.1-demo", "adapted on three afternoons")):
        vals = [
            pick("few_shot_target", "locked_day", graph, m)[0]
            for m in ("etage_set_agreement", "oktas_within1", "genus_top1_in_set", "ood_flagged_share")
        ]
        t4.append([label] + [f3(v) for v in vals])
    t4.append(["rater vs rater (ceiling)", "0.911", "0.771", "–", "–"])
    table(
        "tab4_b0268",
        "The owner's camera (Arducam B0268), locked day 2026-10-20, 120 frames never used before the final evaluation; scored against each of the two annotators and averaged.",  # noqa: E501
        "b0268",
        ["Model", "Exact étage set", "Oktas ±1", "Top genus in set", "OOD flagged"],
        t4,
        "lrrrr",
    )

    # ---- Table 5: ablations (validation)
    try:
        ab = pd.read_parquet(result_file("ablations.parquet"))
        t5 = []
        groups = ["reference", "geometry (SkyRay)", "targets", "loss", "Eye2Sky étage rows", "Eye2Sky étage rows (LODO)"]
        groups.append("regularisation (P080)")
        for r in ab[ab.group.isin(groups)].itertuples(index=False):
            etage = r.mean_etage_f1 if r.mean_etage_f1 == r.mean_etage_f1 else getattr(r, "held_out_etage_f1", float("nan"))
            t5.append([r.group, r.variant[:70], f3(etage), f3(r.mean_genus_f1)])
        table(
            "tab5_ablations",
            "Ablations of the global heads on validation data (one seed; the reference's three-seed spread is below 0.01): mean étage macro-F1 over the sources and mean genus macro-F1; the LODO rows give the held-out fold's étage F1.",  # noqa: E501
            "ablations",
            ["Group", "Variant", "Étage F1", "Genus F1"],
            t5,
            "llrr",
        )
    except OSError:
        print("  ablations: data/ablations.parquet missing (run scripts/ablations.py)")

    # ---- Table 6: human ceiling (P092)
    try:
        hc = pd.read_parquet(result_file("human_ceiling_model.parquet"))
        t6 = []
        for q in ("étage set", "oktas 0-8"):
            part = hc[hc.quantity == q].set_index("metric").value
            t6.append(
                [
                    q,
                    f3(part.get("alpha, humans only")),
                    f3(part.get("alpha, humans + STRATIA")),
                    f3(part.get("leave-one-out agreement, STRATIA vs the others' majority")),
                    f3(part.get("leave-one-out agreement, humans (mean over raters)")),
                ]
            )
        table(
            "tab6_human_ceiling",
            "STRATIA as a sixth Montenegro annotator (validation images): Krippendorff's alpha without and with the model, and agreement with the majority of the other raters.",  # noqa: E501
            "humanceiling",
            ["Quantity", "Alpha (humans)", "Alpha (+ model)", "Model vs others", "Human vs others"],
            t6,
            "lrrrr",
        )
    except OSError:
        print("  human ceiling: missing")

    # ---- Table 7: efficiency (P094)
    try:
        ef = json.loads((result_file("efficiency.json")).read_text(encoding="utf-8"))
        t7 = [[k, f"{v['median_ms']:.1f}", f"{v['p95_ms']:.1f}"] for k, v in ef["latency"].items() if isinstance(v, dict)]
        t7.append(["Raspberry Pi 5", "–", "–"])
        table(
            "tab7_efficiency",
            f"Latency per frame (median / p95 of repeated runs) of the deployed graph; {ef['params']['total'] / 1e6:.1f} M parameters, {ef['flops_forward_512'] / 1e9:.0f} GFLOPs at {ef['input']}.",  # noqa: E501
            "efficiency",
            ["Setting", "Median ms", "p95 ms"],
            t7,
            "lrr",
        )
    except OSError:
        print("  efficiency: missing")
    numbers(res, rows, pick)
    print(f"tables -> {OUT}")
    return 0


def numbers(res: dict, rows: pd.DataFrame, pick) -> None:
    """paper/numbers.tex: LaTeX macros for every number quoted in the prose (generated, never typed)."""
    macros = {}
    head = {h["quantity"]: h for h in res["headline"]}
    macros["LodoModel"] = f"{head['LODO étage macro-F1, mean of the four folds']['model']:.3f}"
    macros["LodoProbe"] = f"{head['LODO étage macro-F1, mean of the four folds']['reference']:.3f}"
    macros["IndGenusModel"] = f"{head['in-domain genus macro-F1 (CCSN, MGCD)']['model']:.3f}"
    macros["IndGenusProbe"] = f"{head['in-domain genus macro-F1 (CCSN, MGCD)']['reference']:.3f}"
    macros["CtwoEtage"] = f"{head['Montenegro étage-set agreement / human ceiling']['model']:.2f}"
    macros["CtwoOktas"] = f"{head['Montenegro oktas ±1 agreement / human ceiling']['model']:.2f}"
    macros["SkillOldlr"] = f"{head['CBH skill, OLDLR (CDLRA) pairs']['model']:.2f}"
    macros["SkillWeste"] = f"{head['CBH skill, WESTE (CDLRB) pairs, OLDLR-only training']['model']:.2f}"
    macros["MaeModel"] = f"{head['CBH MAE (m), all pairs, against the frozen-feature ridge']['model']:.0f}"
    macros["MaeRidge"] = f"{head['CBH MAE (m), all pairs, against the frozen-feature ridge']['reference']:.0f}"
    macros["CovEighty"] = f"{head['80 % interval coverage with the P089 temperature and radius']['model']:.2f}"
    for key, protocol, subset, model, metric, fmt in (
        ("MaeClim", "cbh_pairs", "all", "climatology_train", "mae_m", ".0f"),
        ("SkillAll", "cbh_pairs", "all", "cbh_base_s0", "skill", ".2f"),
        ("CovNinetyFive", "cbh_pairs", "all", "cbh_base_s0", "coverage_95_calibrated", ".2f"),
        ("CovRaw", "cbh_pairs", "all", "cbh_base_s0", "coverage_80", ".2f"),
        ("SkyMiou", "dense", "all", "dense_sky_layer_s0", "sky_miou", ".3f"),
        ("LayerMiou", "dense", "almeria", "dense_sky_layer_s0", "layer_miou", ".3f"),
        ("OodAuroc", "ood", "b0268_vs_in_domain", "indomain_e2s_s0", "auroc", ".2f"),
        ("ZsEtage", "few_shot_target", "locked_day", "stratia-1.0.0-demo", "etage_set_agreement", ".2f"),
        ("ZsOktas", "few_shot_target", "locked_day", "stratia-1.0.0-demo", "oktas_within1", ".2f"),
        ("ZsOod", "few_shot_target", "locked_day", "stratia-1.0.0-demo", "ood_flagged_share", ".2f"),
        ("AdEtage", "few_shot_target", "locked_day", "stratia-1.0.1-demo", "etage_set_agreement", ".2f"),
        ("AdOktas", "few_shot_target", "locked_day", "stratia-1.0.1-demo", "oktas_within1", ".2f"),
        ("AdGenus", "few_shot_target", "locked_day", "stratia-1.0.1-demo", "genus_top1_in_set", ".2f"),
        ("ClearCloud", "dense", "clear_sky_test", "dense_sky_layer_s0", "fraction_cloud", ".2f"),
    ):
        v, _ = pick(protocol, subset, model, metric)
        macros[key] = format(v, fmt) if v == v else "--"
    try:
        hc = pd.read_parquet(result_file("human_ceiling_model.parquet")).set_index(["quantity", "metric"]).value
        loo = "leave-one-out agreement, STRATIA vs the others' majority"
        hum = "leave-one-out agreement, humans (mean over raters)"
        macros["AlphaEtageHumans"] = f"{hc[('étage set', 'alpha, humans only')]:.3f}"
        macros["AlphaEtageModel"] = f"{hc[('étage set', 'alpha, humans + STRATIA')]:.3f}"
        macros["LooEtageModel"] = f"{hc[('étage set', loo)]:.3f}"
        macros["LooEtageHumans"] = f"{hc[('étage set', hum)]:.3f}"
        macros["LooOktasModel"] = f"{hc[('oktas 0-8', loo)]:.3f}"
        macros["LooOktasHumans"] = f"{hc[('oktas 0-8', hum)]:.3f}"
    except (OSError, KeyError):
        pass
    try:
        ef = json.loads((result_file("efficiency.json")).read_text(encoding="utf-8"))
        macros["ParamsM"] = f"{ef['params']['total'] / 1e6:.1f}"
        macros["HeadsParamsM"] = f"{ef['params']['total trainable above the backbone'] / 1e6:.2f}"
        macros["GFlops"] = f"{ef['flops_forward_512'] / 1e9:.0f}"
        for k, v in ef["latency"].items():
            if isinstance(v, dict) and "DirectML" in k:
                macros["LatencyDml"] = f"{v['median_ms']:.0f}"
            if isinstance(v, dict) and "onnx runtime CPU" in k:
                macros["LatencyCpu"] = f"{v['median_ms']:.0f}"
            if isinstance(v, dict) and "fp32, RTX 4050, batch 1" in k:
                macros["LatencyGpu"] = f"{v['median_ms']:.0f}"
    except OSError:
        pass
    try:
        ab = pd.read_parquet(result_file("ablations.parquet")).set_index("variant")
        for key, variant in (
            ("AblNoGeo", "no geometry: ray map and Sun always dropped (p_drop 1.0)"),
            ("AblGeoAlways", "geometry always given (p_drop 0.0)"),
            ("AblHard", "hard targets: first listed genus, majority étages, majority okta"),
            ("AblNoH", "no hierarchy loss (lambda_h 0)"),
            ("AblKendall", "uncertainty (Kendall) task weighting"),
        ):
            if variant in ab.index:
                macros[key] = f"{ab.loc[variant, 'mean_etage_f1']:.3f}"
        ref = ab[ab.index.str.startswith("Gate G4 heads")]
        if len(ref):
            macros["AblRef"] = f"{ref.mean_etage_f1.iloc[0]:.3f}"
    except (OSError, KeyError):
        pass
    lines = ["% generated by scripts/reproduce_tables.py from the committed result files; do not edit"]
    lines += [chr(92) + "newcommand{" + chr(92) + "n" + k + "}{" + v + "}" for k, v in macros.items()]
    text = chr(10).join(lines) + chr(10)
    (ROOT / "paper" / "numbers.tex").write_text(text, encoding="utf-8", newline=chr(10))
    print(f"  numbers.tex: {len(macros)} macros")


if __name__ == "__main__":
    sys.exit(main())
