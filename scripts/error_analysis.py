"""P093 — Error analysis and shortcut re-check on the validation rows.

Confusions of the Gate G4 global heads (native genus classes of CCSN and MGCD, the named pairs St/Sc, Ci/Cs, As/Ns;
étage presence per source), the Sun-glare cases (P088 corruption and the B0268 glare frames), a failure gallery of
the worst validation frames, and the P029 dataset-identity probe re-run on the model's pooled features. Outputs:
data/error_analysis.parquet, docs/data/error_analysis.md and figures under docs/data/figures/.
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
from stratia.data import shortcuts as sc  # noqa: E402
from stratia.data.image_cache import cache_path  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.train import data as D  # noqa: E402
from stratia.train import global_eval as G  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

OUT = ROOT / "data" / "error_analysis.parquet"
DOC = ROOT / "docs" / "data" / "error_analysis.md"
FIGS = ROOT / "docs" / "data" / "figures"
PAIRS = (("St", "Sc"), ("Ci", "Cs"), ("As", "Ns"))


def locked_test_module():
    spec = importlib.util.spec_from_file_location("locked_test", ROOT / "scripts" / "locked_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def confusion(truth: np.ndarray, hat: np.ndarray, names: list[str]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    cm = np.zeros((len(names), len(names)), dtype=np.int64)
    for t, h in zip(truth, hat, strict=True):
        if t in idx and h in idx:
            cm[idx[t], idx[h]] += 1
    return cm


def heatmap(cm: np.ndarray, names: list[str], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(0.55 * len(names) + 2, 0.5 * len(names) + 1.8))
    ax.imshow(row, cmap="YlOrBr", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7, color="black" if row[i, j] < 0.6 else "white")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (native class)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gallery(frames: list[tuple[str, str]], cache_root: Path, path: Path, cols: int = 4, tile: int = 300) -> None:
    from PIL import Image, ImageDraw

    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 44)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (file, caption) in enumerate(frames):
        try:
            im = Image.open(cache_path(cache_root, file)).convert("RGB")
        except OSError:
            continue
        im.thumbnail((tile, tile))
        x, y = (k % cols) * tile, (k // cols) * (tile + 44)
        sheet.paste(im, (x + (tile - im.width) // 2, y))
        for n, line in enumerate(caption.split("\n")[:3]):
            draw.text((x + 4, y + tile + 2 + 13 * n), line, fill="black")
    sheet.save(path, quality=88)


def main() -> int:
    t0 = time.perf_counter()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="P093")
    ap.add_argument("--gallery", type=int, default=16)
    a = ap.parse_args()
    LT = locked_test_module()
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows_all = D.global_rows(eye2sky_etage=True, include_test=True)
    sel = rows_all[rows_all.split == "val"].reset_index(drop=True)
    rows, ds = LT.global_dataset(sel, data_root, cache_root, "indomain_val")
    model = LT.load_global(LT.MODELS["indomain_e2s_s0"], int(ds.grid.shape[-1]), device)
    idx_all = np.arange(len(rows))
    pred = G.predict(model, ds, idx_all, device)
    man = (
        pd.read_parquet(ROOT / "data" / "manifest.parquet", columns=["sample_id", "image_file"]).set_index("sample_id").image_file
    )
    files = rows.sample_id.map(man)
    FIGS.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # ---- 1. genus confusions on the native classes of CCSN and MGCD
    pair_rows = []
    for src in ("ccsn", "mgcd"):
        m = (rows.source.to_numpy() == src) & rows.native_label.notna().to_numpy()
        part = rows[m]
        sets = D.native_class_sets(src, part.native_label.dropna().unique().tolist())
        cs = {k: v for k, v in sets.items() if v[0]}
        keep = part.native_label.isin(list(cs)).to_numpy()
        scores, names = D.native_scores(pred["genus"][m][keep], cs)
        truth = part.native_label.to_numpy()[keep]
        hat = np.array(names)[scores.argmax(1)]
        cm = confusion(truth, hat, list(names))
        heatmap(cm, list(names), f"{src.upper()} native classes, validation (rows = true)", FIGS / f"genus_confusion_{src}.png")
        acc = float((truth == hat).mean())
        records.append({"section": "genus", "dataset": src, "metric": "accuracy", "value": acc, "n": int(keep.sum())})
        for i, n in enumerate(names):
            tp = cm[i, i]
            records.append(
                {
                    "section": "genus",
                    "dataset": src,
                    "metric": f"recall {n}",
                    "value": float(tp / max(cm[i].sum(), 1)),
                    "n": int(cm[i].sum()),
                }
            )
            top = np.argsort(cm[i])[::-1]
            worst = next((names[j] for j in top if j != i and cm[i, j] > 0), None)
            if worst is not None:
                records.append(
                    {
                        "section": "genus",
                        "dataset": src,
                        "metric": f"most confused with, {n}",
                        "value": float(cm[i, names.index(worst)] / max(cm[i].sum(), 1)),
                        "n": int(cm[i].sum()),
                        "note": worst,
                    }
                )
        for p, q in PAIRS:
            if p in names and q in names:
                i, j = names.index(p), names.index(q)
                both = cm[i].sum() + cm[j].sum()
                cross = cm[i, j] + cm[j, i]
                pair_rows.append(
                    {
                        "section": "genus pair",
                        "dataset": src,
                        "metric": f"{p}/{q} swapped share",
                        "value": float(cross / max(both, 1)),
                        "n": int(both),
                        "note": f"{cm[i, j]} {p}->{q}, {cm[j, i]} {q}->{p}",
                    }
                )
        print(f"[genus] {src}: accuracy {acc:.3f} on {int(keep.sum())} rows; classes {list(names)}", flush=True)
    records += pair_rows

    # ---- 2. étage presence errors per source (exhaustive étage rows)
    full = (np.stack(rows.etage_w.to_numpy()) >= 1.0).all(1)
    for src in sorted(rows.source.unique()):
        m = full & (rows.source.to_numpy() == src)
        if not m.any():
            continue
        truth = np.stack(rows.etage.to_numpy())[m] > 0.5
        if not truth.any():                                     # SWIMCAT rows carry no étage label
            continue
        hat = pred["etage"][m] > 0.5
        for k, name in enumerate(ETAGE_CLASSES):
            t, h = truth[:, k], hat[:, k]
            records += [
                {"section": "etage", "dataset": src, "metric": f"{name}: true rate", "value": float(t.mean()), "n": int(m.sum())},
                {
                    "section": "etage",
                    "dataset": src,
                    "metric": f"{name}: predicted rate",
                    "value": float(h.mean()),
                    "n": int(m.sum()),
                },
                {
                    "section": "etage",
                    "dataset": src,
                    "metric": f"{name}: missed (true, not predicted)",
                    "value": float((t & ~h).sum() / max(t.sum(), 1)),
                    "n": int(t.sum()),
                },
                {
                    "section": "etage",
                    "dataset": src,
                    "metric": f"{name}: spurious (predicted, not true)",
                    "value": float((~t & h).sum() / max((~t).sum(), 1)),
                    "n": int((~t).sum()),
                },
            ]
        records.append(
            {
                "section": "etage",
                "dataset": src,
                "metric": "exact set agreement",
                "value": float((truth == hat).all(1).mean()),
                "n": int(m.sum()),
            }
        )

    # ---- 3. glare: the P088 corruption row and the B0268 glare frames
    rob = pd.read_parquet(ROOT / "data" / "robustness.parquet")
    clean = rob[rob.corruption.str.lower().str.contains("clean")]
    glare = rob[rob.corruption.str.lower().str.contains("glare")]
    if len(clean) and len(glare):
        for col in ("etage_f1", "genus_f1", "sky_miou", "clear_sky_cloud_fraction", "cbh_mae_m"):
            records.append(
                {
                    "section": "glare",
                    "dataset": "val (P088 corruption)",
                    "metric": f"{col}: clean -> sun glare",
                    "value": float(glare[col].iloc[0]),
                    "value_ref": float(clean[col].iloc[0]),
                }
            )
    try:
        b = pd.read_parquet(ROOT / "data" / "b0268_zero_shot_scores.parquet")
        b = b[b.day == "all"]
        for col in ("glare_fraction_strong_frames", "glare_fraction_other_frames"):
            v = b[col].dropna()
            if len(v):
                records.append(
                    {"section": "glare", "dataset": "b0268 (adaptation days)", "metric": col, "value": float(v.iloc[0])}
                )
    except (OSError, KeyError):
        pass

    # ---- 4. failure gallery: exhaustive-étage frames with the largest étage error, plus confident wrong genus
    err = np.abs(pred["etage"] - np.stack(rows.etage.to_numpy())).max(1)
    err[~full] = -1
    order = np.argsort(err)[::-1][: a.gallery]
    frames = []
    for i in order:
        t = [c for c, v in zip(ETAGE_CLASSES, np.stack(rows.etage.to_numpy())[i] > 0.5, strict=True) if v]
        h = [c for c, v in zip(ETAGE_CLASSES, pred["etage"][i] > 0.5, strict=True) if v]
        frames.append(
            (
                files.iloc[i],
                f"{rows.source.iloc[i]} | true {','.join(t) or '-'}\npred {','.join(h) or '-'} (max err {err[i]:.2f})\n"
                f"{str(rows.native_label.iloc[i])[:28]}",
            )
        )
    gallery(frames, cache_root, FIGS / "failure_gallery.jpg")
    records.append({"section": "gallery", "dataset": "val", "metric": "frames shown", "value": float(len(frames))})

    # ---- 5. dataset-identity probe on the model's pooled features vs the backbone CLS features (P029 re-check)
    logits = LT.global_logits(model, ds, idx_all, device)
    pooled = logits["pooled"]
    cls = FeatureCache(cache_root, "dinov3_vits16", 512).features(rows.sample_id, "cls")
    labels = rows.dataset.to_numpy().astype(str)
    counts = pd.Series(labels).value_counts()
    keep_ds = counts[counts >= 60].index.tolist()
    m = np.isin(labels, keep_ds)
    rng = np.random.default_rng(0)
    sub = np.flatnonzero(m)[sc.balanced_subsample(labels[m], int(min(counts[keep_ds])), rng)]
    test = np.array(sc.holdout_random_groups(rows.unit.astype(str).iloc[sub]), dtype=bool)
    # every dataset on both sides: a source whose validation rows form one or two units (Montenegro's temporal
    # blocks) would otherwise land whole in one fold and the probe could not learn it from any feature
    for dsn in keep_ds:
        here = labels[sub] == dsn
        if here.sum() and (test[here].all() or not test[here].any()):
            pick = rng.random(int(here.sum())) < 0.3
            test[np.flatnonzero(here)] = pick
    for name, x in (("model pooled features (512-d)", pooled), ("backbone CLS features (384-d)", cls)):
        r = sc.linear_probe(x[sub], labels[sub], test)
        records.append(
            {
                "section": "dataset identity",
                "dataset": ", ".join(keep_ds),
                "metric": name,
                "value": float(r["accuracy"]),
                "value_ref": float(r["chance"]),
                "n": int(r["n_test"]),
                "note": f"balanced accuracy {r['balanced_accuracy']:.3f}",
            }
        )
        print(f"[identity] {name}: accuracy {r['accuracy']:.3f} (chance {r['chance']:.3f})", flush=True)

    df = pd.DataFrame(records)
    df.to_parquet(OUT, index=False)
    with Run(
        __import__("omegaconf").OmegaConf.create({"model": LT.MODELS["indomain_e2s_s0"], "split": "val"}),
        name="error_analysis",
        phase=a.phase,
    ) as run:
        run.finish(n_rows=float(len(df)))
    write_doc(df, run.run_id)
    print(f"done in {time.perf_counter() - t0:.0f} s -> {OUT}, {DOC}", flush=True)
    return 0


def _f(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def write_doc(df: pd.DataFrame, run_id: str) -> None:
    lines = [
        "# Error analysis and shortcut re-check (P093)",
        "",
        "Gate G4 global heads (`indomain_e2s`, EMA) on the validation rows; the locked test numbers are in P090. "
        f"Run `{run_id}`.",
        "",
        "## Genus confusions (native classes)",
        "",
        "![CCSN](figures/genus_confusion_ccsn.png)",
        "",
        "![MGCD](figures/genus_confusion_mgcd.png)",
        "",
        "| Dataset | Class | Recall | n | Most confused with (share) |",
        "|---|---|---|---|---|",
    ]
    g = df[df.section == "genus"]
    for src in ("ccsn", "mgcd"):
        part = g[g.dataset == src]
        acc = part[part.metric == "accuracy"]
        if len(acc):
            lines.append(f"| {src.upper()} | all | accuracy {acc.value.iloc[0]:.3f} | {int(acc.n.iloc[0])} | |")
        for r in part[part.metric.str.startswith("recall ")].itertuples(index=False):
            cls = r.metric.replace("recall ", "")
            w = part[part.metric == f"most confused with, {cls}"]
            note = f"{w.note.iloc[0]} ({w.value.iloc[0]:.2f})" if len(w) else ""
            lines.append(f"| {src.upper()} | {cls} | {r.value:.3f} | {int(r.n)} | {note} |")
    lines += ["", "| Dataset | Pair | Swapped share of the pair's rows | n | Counts |", "|---|---|---|---|---|"]
    for r in df[df.section == "genus pair"].itertuples(index=False):
        lines.append(
            f"| {r.dataset.upper()} | {r.metric.replace(' swapped share', '')} | {r.value:.3f} | {int(r.n)} | {r.note} |"
        )
    lines += [
        "",
        "## Étage presence per source (exhaustive étage rows, threshold 0.5)",
        "",
        "| Source | Étage | True rate | Predicted rate | Missed | Spurious |",
        "|---|---|---|---|---|---|",
    ]
    e = df[df.section == "etage"]
    for src in sorted(e.dataset.unique()):
        part = e[e.dataset == src].set_index("metric").value
        for name in ETAGE_CLASSES:
            lines.append(
                f"| {src} | {name} | {_f(part.get(f'{name}: true rate'))} | {_f(part.get(f'{name}: predicted rate'))} | "
                f"{_f(part.get(f'{name}: missed (true, not predicted)'))} | {_f(part.get(f'{name}: spurious (predicted, not true)'))} |"  # noqa: E501
            )
        lines.append(f"| {src} | exact set | {_f(part.get('exact set agreement'))} | | | |")
    lines += ["", "## Sun glare", "", "| Where | Quantity | Value | Clean |", "|---|---|---|---|"]
    for r in df[df.section == "glare"].itertuples(index=False):
        ref = _f(getattr(r, "value_ref", float("nan")))
        lines.append(f"| {r.dataset} | {r.metric} | {_f(r.value)} | {ref} |")
    lines += [
        "",
        "## Failure gallery",
        "",
        "The validation frames with the largest étage error (exhaustive étage rows), true and predicted étages under each frame.",
        "",
        "![gallery](figures/failure_gallery.jpg)",
        "",
        "## Dataset identity from the model's features (P029 re-check)",
        "",
        "| Features | Datasets | Accuracy | Chance | n test | Note |",
        "|---|---|---|---|---|---|",
    ]
    for r in df[df.section == "dataset identity"].itertuples(index=False):
        lines.append(f"| {r.metric} | {r.dataset} | {r.value:.3f} | {_f(r.value_ref)} | {int(r.n)} | {r.note} |")
    lines += [
        "",
        "Attention maps were not produced: the heads pool an 8 × 8 token grid through a learned attention pool whose "
        "weights are not stored at inference; the sky-parse mask is the per-pixel view the paper shows instead.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
