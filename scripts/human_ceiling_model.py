"""P092 — STRATIA as an extra Montenegro annotator (validation units only).

The Gate G4 global heads answer the Montenegro validation images the way the human raters did (an étage set and a
cover in oktas); Krippendorff's alpha is computed over the human raters with and without the model, and every rater
(the model included) is scored against the majority / median of the others (the P035 leave-one-out). Outputs:
data/human_ceiling_model.parquet, docs/data/human_ceiling_model.md, docs/data/figures/human_ceiling_model.png.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stratia.contract import ETAGE_CLASSES  # noqa: E402
from stratia.data.montenegro import load_annotations, load_items  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.labels import agreement as ag  # noqa: E402
from stratia.labels.ontology import load_ontology  # noqa: E402
from stratia.train import data as D  # noqa: E402
from stratia.train import global_eval as G  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

OUT = ROOT / "data" / "human_ceiling_model.parquet"
DOC = ROOT / "docs" / "data" / "human_ceiling_model.md"
FIG = ROOT / "docs" / "data" / "figures" / "human_ceiling_model.png"
MODEL_RATER = "STRATIA"


def locked_test_module():
    spec = importlib.util.spec_from_file_location("locked_test", ROOT / "scripts" / "locked_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_answers(rows: pd.DataFrame, pred: dict[str, np.ndarray], items: pd.DataFrame) -> pd.DataFrame:
    """The model's answers in the raters' coding: étage set 'low|mid', oktas 0-8, for the Montenegro rows."""
    mon = np.flatnonzero(rows.dataset.to_numpy() == "montenegro")
    files = rows.sample_id.iloc[mon].str.split(":", n=1).str[1].str.replace("Montenegro/", "", regex=False)
    item_of = items.set_index("image_file").item_id
    out = []
    for i, f in zip(mon, files, strict=True):
        if f not in item_of.index:
            continue
        e = pred["etage"][i] > 0.5
        out.append(
            {
                "item_id": item_of[f],
                "user_id": MODEL_RATER,
                "etage_set": "|".join(sorted(c for c, on in zip(ETAGE_CLASSES, e, strict=True) if on)),
                "oktas": int(pred["oktas"][i].argmax()),
            }
        )
    return pd.DataFrame(out)


def pairwise_with(long: pd.DataFrame, column: str, rater: str, tolerance: float | None) -> pd.DataFrame:
    """Agreement of `rater` with every other rater on the units both answered; and human-human pairs for scale."""
    table = ag.pivot(long, column)
    rows = []
    cols = [c for c in table.columns]
    for a in cols:
        for b in cols:
            if a >= b:
                continue
            both = table[[a, b]].dropna()
            if len(both) < 20:
                continue
            if tolerance is None:
                agree = float((both[a] == both[b]).mean())
            else:
                agree = float((np.abs(both[a].astype(float) - both[b].astype(float)) <= tolerance).mean())
            rows.append(
                {"a": a, "b": b, "n": len(both), "agreement": agree, "pair": "model-human" if rater in (a, b) else "human-human"}
            )
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.perf_counter()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="P092")
    a = ap.parse_args()
    LT = locked_test_module()
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows_all = D.global_rows(eye2sky_etage=True, include_test=True)
    sel = rows_all[rows_all.split == "val"].reset_index(drop=True)
    rows, ds = LT.global_dataset(sel, data_root, cache_root, "indomain_val")
    model = LT.load_global(LT.MODELS["indomain_e2s_s0"], int(ds.grid.shape[-1]), device)
    pred = G.predict(model, ds, np.arange(len(rows)), device)
    root = data_root / "Montenegro"
    ont = load_ontology()
    ann, items = load_annotations(root), load_items(root)
    derived = ag.derive(ann, ont)
    answers = model_answers(rows, pred, items)
    val_items = set(answers.item_id)
    humans = derived[derived.item_id.isin(val_items)][["item_id", "user_id", "etage_set", "oktas"]].copy()
    humans["user_id"] = humans.user_id.astype(str)
    both = pd.concat([humans, answers], ignore_index=True)
    print(
        f"{len(val_items)} validation images, {humans.user_id.nunique()} human raters, {len(humans)} human annotations",
        flush=True,
    )

    results = []
    for column, level, tol, label in (("etage_set", "nominal", None, "étage set"), ("oktas", "interval", 1.0, "oktas 0-8")):
        a_h = ag.krippendorff_alpha(ag.pivot(humans, column), level)
        a_m = ag.krippendorff_alpha(ag.pivot(both, column), level)
        kind = "nominal" if tol is None else "numeric"
        loo = ag.leave_one_out(ag.pivot(both, column), kind, tolerance=tol or 1.0)
        loo_h = ag.leave_one_out(ag.pivot(humans, column), kind, tolerance=tol or 1.0)
        pw = pairwise_with(both, column, MODEL_RATER, tol)
        model_row = loo[loo.rater == MODEL_RATER].iloc[0]
        human_rows = loo_h[loo_h.rater != "all"]
        results += [
            {"quantity": label, "metric": "alpha, humans only", "value": a_h},
            {"quantity": label, "metric": "alpha, humans + STRATIA", "value": a_m},
            {
                "quantity": label,
                "metric": "leave-one-out agreement, STRATIA vs the others' majority",
                "value": float(model_row.agreement),
                "n": int(model_row.units),
            },
            {
                "quantity": label,
                "metric": "leave-one-out agreement, humans (mean over raters)",
                "value": float(human_rows.agreement.mean()),
                "n": int(human_rows.units.sum()),
            },
            {
                "quantity": label,
                "metric": "leave-one-out agreement, humans (min / max)",
                "value": float(human_rows.agreement.min()),
                "value_hi": float(human_rows.agreement.max()),
            },
            {
                "quantity": label,
                "metric": "pairwise agreement, model-human (mean over raters)",
                "value": float(pw[pw.pair == "model-human"].agreement.mean()) if len(pw) else float("nan"),
            },
            {
                "quantity": label,
                "metric": "pairwise agreement, human-human (mean over pairs)",
                "value": float(pw[pw.pair == "human-human"].agreement.mean()) if len(pw) else float("nan"),
            },
        ]
        for r in loo.itertuples(index=False):
            results.append(
                {
                    "quantity": label,
                    "metric": f"leave-one-out agreement, rater {r.rater}",
                    "value": float(r.agreement),
                    "n": int(r.units),
                }
            )
    df = pd.DataFrame(results)
    df.to_parquet(OUT, index=False)
    with Run(
        __import__("omegaconf").OmegaConf.create(
            {"model": LT.MODELS["indomain_e2s_s0"], "split": "val", "n_items": len(val_items)}
        ),
        name="human_ceiling_model",
        phase=a.phase,
    ) as run:
        run.finish(
            **{
                f"{q}/{m}": float(v)
                for q, m, v in zip(df.quantity, df.metric, df.value, strict=True)
                if "rater" not in m and v == v
            }
        )
    figure(df)
    write_doc(df, len(val_items), humans.user_id.nunique(), run.run_id)
    print(f"done in {time.perf_counter() - t0:.0f} s -> {OUT}, {DOC}, {FIG}", flush=True)
    return 0


def figure(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, q in zip(axes, ("étage set", "oktas 0-8"), strict=True):
        part = df[(df.quantity == q) & df.metric.str.startswith("leave-one-out agreement, rater ")]
        part = part[~part.metric.str.endswith("rater all")].copy()
        part["rater"] = part.metric.str.replace("leave-one-out agreement, rater ", "", regex=False)
        part = part.sort_values("value")
        colours = ["#663300" if r == MODEL_RATER else "#B8B0A2" for r in part.rater]
        ax.barh(part.rater, part.value, color=colours)
        ax.set_xlim(0, 1)
        ax.set_xlabel("agreement with the majority / median of the other raters")
        a_h = float(df[(df.quantity == q) & (df.metric == "alpha, humans only")].value.iloc[0])
        a_m = float(df[(df.quantity == q) & (df.metric == "alpha, humans + STRATIA")].value.iloc[0])
        ax.set_title(f"{q}: alpha {a_h:.3f} humans, {a_m:.3f} with STRATIA", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def write_doc(df: pd.DataFrame, n_items: int, n_raters: int, run_id: str) -> None:
    lines = [
        "# STRATIA as an extra Montenegro annotator (P092)",
        "",
        f"The Gate G4 global heads (`indomain_e2s`, EMA) answer the {n_items} Montenegro validation images the way the "
        f"{n_raters} human raters did: an étage set from the altitude class and a cover in oktas. Krippendorff's alpha "
        "over the raters with and without the model, and each rater against the majority (étage set) or the median "
        "within ±1 okta (cover) of the others, as in P035. Validation units only; the locked test is P090. Run "
        f"`{run_id}`.",
        "",
        "| Quantity | Metric | Value | n |",
        "|---|---|---|---|",
    ]
    for r in df[~df.metric.str.startswith("leave-one-out agreement, rater ")].itertuples(index=False):
        v = f"{r.value:.3f}" + (f" / {r.value_hi:.3f}" if "value_hi" in df.columns and r.value_hi == r.value_hi else "")
        n = "" if "n" not in df.columns or r.n != r.n else int(r.n)
        lines.append(f"| {r.quantity} | {r.metric} | {v} | {n} |")
    lines += [
        "",
        f"![raters]({FIG.relative_to(DOC.parent).as_posix()})",
        "",
        "Reading: alpha is a chance-corrected agreement over the whole panel; adding a rater who agrees with the panel "
        "about as often as the humans do leaves it unchanged, a rater who disagrees systematically lowers it. The "
        "leave-one-out rows put the model on the same footing as each human: its agreement with the others' majority "
        "next to theirs.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
