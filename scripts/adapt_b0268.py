"""Few-shot adaptation of the global heads to the owner's B0268 (P084, reduced).

    python scripts/adapt_b0268.py --draw k1_d0 [--build-cache]        one few-shot draw
    python scripts/adapt_b0268.py --draw all                           k0_d0 (zero-shot baseline) and every k >= 1 draw
    python scripts/adapt_b0268.py --final                              train on every adaptation day for the demo model

Reduced form of the plan's LP-FT with source replay: the frozen DINOv3 ViT-S/16 features stay cached, only the
geometry encoder and the global heads (the P073 `GlobalModel`, initialised from the Gate G4 global run) are
trained. Each batch holds `--batch` B0268 training frames plus `--replay` times as many source rows (the training
rows of the in-domain split, uniformly drawn; the legacy lesson that 4x worked and 10x over-biased). LP phase: only
the three output layers (genus, étage, oktas) at `--lr-lp`; FT phase: every parameter at `--lr-ft`; fixed epochs
over the B0268 training frames, no early stopping and no model selection on the held-out days. Before and after,
the model is scored on the draw's held-out adaptation days against each rater of the P041 pair
(`stratia.train.b0268_eval.score_against_raters`, ceiling = the raters' pairwise agreement) and on the source
validation split (`score_dataset`); a drop of more than `--max-regression` in the mean étage or genus F1 fails the
draw. Outputs: data/b0268_adaptation.parquet, docs/data/b0268_adaptation.md, one P084 run per draw whose
checkpoints/best.pt is loadable by scripts/export_onnx.py --global-run.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from stratia.augment.dropout import GeometryDropout  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.train import b0268_eval as E  # noqa: E402
from stratia.train import data as D  # noqa: E402
from stratia.train.global_eval import GlobalModel, losses_for, predict, score_dataset  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

DEFAULT_INIT = "20261011T125754Z_global_indomain_e2s_s0_27d0d07c22"
OUT, DOC = Path("data/b0268_adaptation.parquet"), Path("docs/data/b0268_adaptation.md")
B0268_KEYS = ["n", "etage_set_agreement", "c2_etage_ratio", "oktas_within1", "c2_oktas_ratio", "oktas_mae", "oktas_bias",
              "genus_top1_in_set"]
SOURCE_KEYS = ["mean_etage_f1", "mean_genus_f1", "montenegro/c2_etage_ratio", "montenegro/c2_oktas_ratio"]


# ----------------------------------------------------------------------------------------------- model
def lp_parameters(model: GlobalModel) -> list[torch.nn.Parameter]:
    """The output layers of the three global heads: what the linear-probe phase trains."""
    heads = model.net.global_heads
    return [p for m in (heads.genus, heads.etage, heads.oktas) for p in m.parameters()]


def load_init(run_dir: Path, d_model: int, device: str) -> tuple[GlobalModel, dict]:
    """The Gate G4 global heads (EMA weights when the run kept them, as the export does)."""
    cfg = OmegaConf.load(run_dir / "config.yaml")
    model = GlobalModel(d_model, int(cfg["hidden"]), float(cfg["dropout"]), bool(cfg["kendall"]))
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    weights = dict(state["model"])
    if state.get("ema"):
        weights.update(state["ema"]["shadow"])
    model.load_state_dict(weights)
    # the keys calibrate.py / export_onnx.py read from a P073 global run (the adapted run is scored like one)
    keep = {"hidden": int(cfg["hidden"]), "dropout": float(cfg["dropout"]), "kendall": bool(cfg["kendall"]),
            "name": str(cfg["name"]), "lodo": cfg.get("lodo"), "eye2sky_etage": bool(cfg.get("eye2sky_etage", False))}
    return model.to(device), keep


def save_checkpoint(model: GlobalModel, path: Path, epochs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "ema": None, "step": 0,
                "epoch": epochs}, path)


# ----------------------------------------------------------------------------------------------- data
def b0268_caches(cache_root: Path, rows: pd.DataFrame, build: bool) -> tuple[FeatureCache, FeatureCache]:
    """The small B0268 caches (own keys, so the big caches and their fingerprints stay untouched)."""
    grid = FeatureCache(cache_root, "dinov3_vits16", D.SIZE, transform="ffr1-patches-b0268-pool4")
    cls = FeatureCache(cache_root, "dinov3_vits16", D.SIZE, transform="ffr1-b0268")
    ids, files = rows.sample_id.tolist(), rows.image_file.tolist()
    missing = [c.key for c in (grid, cls) if not (c.complete and set(c.index().sample_id) >= set(ids))]
    if missing:
        if not build:
            raise SystemExit(f"B0268 feature caches missing ({missing}); rerun with --build-cache (GPU, about a minute)")
        from stratia.features.backbones import FrozenBackbone

        backbone = FrozenBackbone("dinov3_vits16")
        grid.build(ids, files, backbone, cache_root, patches=True, batch=8, workers=0, pool=4)
        cls.build(ids, files, backbone, cache_root, patches=False, batch=16, workers=0)
        del backbone
        torch.cuda.empty_cache()
    return grid, cls


def dataset_for(rows: pd.DataFrame, grid: FeatureCache, cls: FeatureCache, geo: D.GeometryCache) -> D.GlobalDataset:
    return D.GlobalDataset(rows, grid.rows_for(rows.sample_id), cls.rows_for(rows.sample_id), grid.load("patches"),
                           cls.load("cls"), geo)


def ceiling(raters: str = "AB") -> dict[str, float]:
    c = pd.read_parquet("data/b0268_ceiling.parquet")
    c = c[c.raters == raters].set_index("quantity")
    return {"etage_set": float(c.loc["étage set", "leave_one_out"]), "oktas_within1": float(c.loc["oktas 0-8", "leave_one_out"])}


# ----------------------------------------------------------------------------------------------- evaluation
def evaluate(model: GlobalModel, ds_b: D.GlobalDataset, idx_b: np.ndarray, raters: dict, ceil: dict,
             ds_s: D.GlobalDataset, idx_s: np.ndarray, device: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """B0268 held-out days against each rater (per day + all) and the source validation split per dataset."""
    b0268 = pd.DataFrame()
    if len(idx_b):
        pred = predict(model, ds_b, idx_b, device)
        part = ds_b.rows.iloc[idx_b]
        b0268 = E.per_day_against_raters(pred, part.item_id, part.day, raters, ceil)
    pred_s = predict(model, ds_s, idx_s, device)
    part = ds_s.rows.iloc[idx_s].reset_index(drop=True)
    source: dict[str, float] = {}
    e, g = [], []
    for name in sorted(part.dataset.unique()):
        sel = (part.dataset == name).to_numpy()
        sc = score_dataset(part[sel].reset_index(drop=True), {k: v[sel] for k, v in pred_s.items()})
        for k, v in sc.items():
            source[f"{name}/{k}"] = float(v)
        if "etage_f1" in sc:
            e.append(sc["etage_f1"])
        if "genus_f1" in sc:
            g.append(sc["genus_f1"])
    source["mean_etage_f1"] = float(np.mean(e)) if e else float("nan")
    source["mean_genus_f1"] = float(np.mean(g)) if g else float("nan")
    return b0268, source


# ----------------------------------------------------------------------------------------------- training
def train(model: GlobalModel, ds_b: D.GlobalDataset, tr_b: np.ndarray, ds_s: D.GlobalDataset, tr_s: np.ndarray, a,
          device: str, rng: np.random.Generator, log) -> None:
    """LP then FT, fixed epochs over the B0268 training frames with source replay in every batch."""
    drop = GeometryDropout(a.p_drop, a.p_drop)
    per_epoch = max(1, math.ceil(len(tr_b) / a.batch))
    phases = [("lp", lp_parameters(model), a.lr_lp, a.epochs_lp), ("ft", list(model.parameters()), a.lr_ft, a.epochs_ft)]
    step = 0
    for name, params, lr, epochs in phases:
        if epochs <= 0 or not params:
            continue
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW([p for p in params], lr=lr, weight_decay=a.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * per_epoch)
        for epoch in range(epochs):
            model.train()
            order = rng.permutation(tr_b)
            losses = []
            for s in range(0, len(order), a.batch):
                bi = order[s:s + a.batch]
                si = rng.choice(tr_s, size=a.replay * len(bi), replace=len(tr_s) < a.replay * len(bi))
                part_b = D.collate([ds_b[int(i)] for i in bi])
                part_s = D.collate([ds_s[int(i)] for i in si])
                rm, mt, _, _ = drop.batch(part_s["ray_map"].numpy(), part_s["meta"].numpy(), rng)
                part_s["ray_map"], part_s["meta"] = torch.from_numpy(rm), torch.from_numpy(mt)
                batch = {k: torch.cat([part_b[k], part_s[k]]).to(device) for k in part_b}
                out = model(batch["tokens"], batch["cls"], batch["ray_map"], batch["meta"])
                parts = losses_for(out, batch, a.lambda_h)
                if model.weighting is not None:
                    total, _ = model.weighting({k: v for k, v in parts.items() if k != "hier"})
                    total = total + parts["hier"]
                else:
                    total = sum(parts.values())
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_([p for p in params], 1.0)
                opt.step()
                sched.step()
                losses.append(float(total.detach()))
                step += 1
            log(step, **{f"{name}/loss": float(np.mean(losses)), f"{name}/epoch": float(epoch + 1)})
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()


# ----------------------------------------------------------------------------------------------- one draw
def run_draw(draw: str, a, shared: dict, device: str) -> tuple[list[dict], dict]:
    rows_b = D.b0268_rows(draw=draw)
    geo_file = shared["cache_root"] / "geometry" / f"pool4_adapt_b0268_{len(rows_b)}.npz"
    geo_b = D.GeometryCache.build(rows_b, shared["data_root"], geo_file)
    ds_b = dataset_for(rows_b, shared["grid_b"], shared["cls_b"], geo_b)
    tr_b = np.flatnonzero(rows_b.split == "train")
    va_b = np.flatnonzero(rows_b.split == "val")
    if a.final:
        tr_b, va_b = np.flatnonzero(rows_b.split.isin(["train", "val"])), np.array([], dtype=int)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    model, init_cfg = load_init(ROOT / "runs" / a.init, shared["d_model"], device)
    train_days = sorted(rows_b.unit[tr_b].unique().tolist())
    shots = int(shared["few"][shared["few"].draw == draw].shots.iloc[0]) if not a.final else len(train_days)
    name = ("adapt_b0268_final" if a.final else f"adapt_b0268_{draw}") + (f"_{a.tag}" if a.tag != "base" else "")
    val_days = sorted(rows_b.unit[va_b].unique().tolist()) if len(va_b) else []
    # init_cfg's hidden / dropout / kendall sit at the top level: the export reads the run like a P073 global run
    cfg = OmegaConf.create({**vars(a), "draw": draw, "init_run": a.init, **init_cfg,
                            "n_train_b0268": int(len(tr_b)), "n_val_b0268": int(len(va_b)),
                            "train_days": train_days, "val_days": val_days,
                            "n_source_train": int(len(shared["tr_s"])), "shots": shots})
    records: list[dict] = []
    eval_args = (shared["raters"], shared["ceiling"], shared["ds_s"], shared["va_s"], device)
    with Run(cfg, name=name, phase=a.phase) as run:
        before_b, before_s = evaluate(model, ds_b, va_b, *eval_args)
        if len(tr_b):
            train(model, ds_b, tr_b, shared["ds_s"], shared["tr_s"], a, device, rng, run.log)
        after_b, after_s = evaluate(model, ds_b, va_b, *eval_args)
        save_checkpoint(model, run.dir / "checkpoints" / "best.pt", a.epochs_lp + a.epochs_ft)
        regression = max(before_s["mean_etage_f1"] - after_s["mean_etage_f1"],
                         before_s["mean_genus_f1"] - after_s["mean_genus_f1"])
        verdict = "ok" if regression <= a.max_regression else "source regression"
        summary = {"regression": regression, "verdict_ok": float(verdict == "ok")}
        for stage, b, s in (("zero_shot", before_b, before_s), ("adapted", after_b, after_s)):
            for _, r in b.iterrows():
                for k, v in r.items():
                    if k != "day" and isinstance(v, float) and not np.isnan(v):
                        records.append({"draw": draw, "shots": shots, "stage": stage, "scope": "b0268_val", "group": r.day,
                                        "metric": k, "value": v})
                        if r.day == "all" and stage == "adapted":
                            summary[k] = v
            for k, v in s.items():
                if not np.isnan(v):
                    records.append({"draw": draw, "shots": shots, "stage": stage, "scope": "source_val", "group": k.split("/")[0]
                                    if "/" in k else "all", "metric": k.split("/")[-1], "value": v})
            if stage == "adapted":
                summary["source_mean_etage_f1"] = after_s["mean_etage_f1"]
                summary["source_mean_genus_f1"] = after_s["mean_genus_f1"]
        run.finish(**summary)
    for r in records:
        r["run_id"] = run.run_id
        r["tag"] = a.tag
    return records, {"draw": draw, "shots": shots, "run_id": run.run_id, "verdict": verdict, "before_b": before_b,
                     "after_b": after_b, "before_s": before_s, "after_s": after_s, "n_train": int(len(tr_b)),
                     "n_val": int(len(va_b)), "train_days": train_days, "val_days": val_days}


# ----------------------------------------------------------------------------------------------- report
def sweep_table(df: pd.DataFrame, max_regression: float) -> list[str]:
    """One row per configuration tag: mean gain over the k >= 1 draws on the held-out days and the source change."""
    def mean_delta(part, scope, group, metric):
        p = part[(part.scope == scope) & (part.group == group) & (part.metric == metric)]
        w = p.pivot_table(index="draw", columns="stage", values="value")
        if "adapted" not in w or "zero_shot" not in w:
            return float("nan"), float("nan")
        d = w["adapted"] - w["zero_shot"]
        return float(d.mean()), float(w["adapted"].mean())

    rows = []
    for tag, part in df[(df.shots > 0) & (df.draw != "k0_d0")].groupby("tag"):
        e_gain, e_after = mean_delta(part, "b0268_val", "all", "etage_set_agreement")
        o_gain, o_after = mean_delta(part, "b0268_val", "all", "oktas_within1")
        g_gain, g_after = mean_delta(part, "b0268_val", "all", "genus_top1_in_set")
        se, _ = mean_delta(part, "source_val", "all", "mean_etage_f1")
        sg, _ = mean_delta(part, "source_val", "all", "mean_genus_f1")
        worst = part[(part.scope == "source_val") & (part.group == "all")].pivot_table(
            index=["draw", "metric"], columns="stage", values="value")
        worst_drop = float((worst["zero_shot"] - worst["adapted"]).max()) if len(worst) else float("nan")
        n_draws = part.draw.nunique()
        rows.append({"tag": tag, "draws": n_draws, "etage_after": e_after, "etage_gain": e_gain, "oktas_after": o_after,
                     "oktas_gain": o_gain, "genus_after": g_after, "genus_gain": g_gain, "source_etage_change": se,
                     "source_genus_change": sg, "worst_source_drop": worst_drop,
                     "passes": worst_drop <= max_regression})
    if not rows:
        return []
    t = pd.DataFrame(rows)
    lines = ["## Configurations (mean over the k >= 1 draws; held-out days against each rater, averaged)", "",
             "| Tag | Draws | Étage set after (gain) | Oktas ±1 after (gain) | Genus top-1 after (gain) | "
             "Source étage F1 change | Source genus F1 change | Worst source drop | Within the rule |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples(index=False):
        lines.append(f"| `{r.tag}` | {r.draws} | {r.etage_after:.3f} ({r.etage_gain:+.3f}) | "
                     f"{r.oktas_after:.3f} ({r.oktas_gain:+.3f}) | "
                     f"{r.genus_after:.3f} ({r.genus_gain:+.3f}) | {r.source_etage_change:+.3f} | {r.source_genus_change:+.3f} | "
                     f"{r.worst_source_drop:.3f} | {'yes' if r.passes else 'no'} |")
    return lines + [""]


def write_doc(results: list[dict], a, all_rows: pd.DataFrame | None = None) -> None:
    def f(v):
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"

    def row_of(df: pd.DataFrame, key: str):
        return None if not len(df) else df[df.day == "all"].iloc[0].get(key)

    lines = ["# B0268 few-shot adaptation (P084, reduced)", "",
             f"Heads-only LP-FT on cached DINOv3 ViT-S/16 features from the Gate G4 global run `{a.init}`; this report's "
             f"draw tables are configuration `{a.tag}`: LP {a.epochs_lp} epochs at {a.lr_lp:g} on the three output layers, "
             f"FT {a.epochs_ft} epochs at {a.lr_ft:g} on every head parameter; batch {a.batch} B0268 frames + {a.replay}x "
             f"source replay; geometry dropout {a.p_drop} on the source part; fixed epochs, no model selection on the "
             "held-out days. B0268 columns: the model against each rater of the P041 pair, averaged, on the draw's "
             "held-out adaptation days (`c2_*_ratio` = model / the raters' pairwise agreement). Source columns: the "
             f"in-domain validation split; a draw fails when the mean étage or genus F1 drops by more than "
             f"{a.max_regression:g}.", ""]
    if all_rows is not None and len(all_rows):
        lines += sweep_table(all_rows, a.max_regression)
    for r in results:
        lines += [f"## Draw `{r['draw']}` (k = {r['shots']}: train {r['train_days'] or 'none'} / {r['n_train']} frames; held out "
                  f"{r['val_days'] or 'none'} / {r['n_val']} frames; run `{r['run_id']}`; verdict **{r['verdict']}**)", "",
                  "| Scope | Metric | Before | After | Change |", "|---|---|---|---|---|"]
        for k in B0268_KEYS:
            b, c = row_of(r["before_b"], k), row_of(r["after_b"], k)
            if b is None and c is None:
                continue
            delta = None if b is None or c is None or np.isnan(b) or np.isnan(c) else c - b
            lines.append(f"| B0268 held-out | {k} | {f(b)} | {f(c)} | {f(delta)} |")
        for k in SOURCE_KEYS:
            b, c = r["before_s"].get(k), r["after_s"].get(k)
            if b is None and c is None:
                continue
            delta = None if b is None or c is None else c - b
            lines.append(f"| source val | {k} | {f(b)} | {f(c)} | {f(delta)} |")
        if len(r["after_b"]) > 1:
            lines += ["", "| Held-out day | étage set before | after | oktas ±1 before | after |", "|---|---|---|---|---|"]
            for _, rr in r["after_b"].iterrows():
                if rr.day == "all":
                    continue
                bb = r["before_b"][r["before_b"].day == rr.day].iloc[0]
                lines.append(f"| {rr.day} | {f(bb.get('etage_set_agreement'))} | {f(rr.get('etage_set_agreement'))} | "
                             f"{f(bb.get('oktas_within1'))} | {f(rr.get('oktas_within1'))} |")
        lines.append("")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def results_from_rows(df: pd.DataFrame, tag: str, max_regression: float) -> list[dict]:
    """`write_doc` records rebuilt from the results table for one configuration (`--doc-only`): the before / after
    tables and the verdicts come from the table, the train and held-out days from each run's config when its folder
    exists ("?" otherwise); the final run (draw k0_d0) comes last."""
    def table(p: pd.DataFrame, stage: str) -> pd.DataFrame:
        q = p[(p.stage == stage) & (p.scope == "b0268_val")]
        if not len(q):
            return pd.DataFrame(columns=["day"])
        w = q.pivot_table(index="group", columns="metric", values="value").reset_index().rename(columns={"group": "day"})
        w.columns.name = None
        return pd.concat([w[w.day == "all"], w[w.day != "all"].sort_values("day")], ignore_index=True)

    def source(p: pd.DataFrame, stage: str) -> dict:
        q = p[(p.stage == stage) & (p.scope == "source_val")]
        return {(m if g == "all" else f"{g}/{m}"): float(v) for g, m, v in zip(q.group, q.metric, q.value, strict=True)}

    out = []
    part = df[df.tag == tag]
    for draw in sorted(part.draw.unique(), key=lambda d: (d == "k0_d0", d)):
        p = part[part.draw == draw]
        before_s, after_s = source(p, "zero_shot"), source(p, "adapted")
        regression = max(before_s.get("mean_etage_f1", 0.0) - after_s.get("mean_etage_f1", 0.0),
                         before_s.get("mean_genus_f1", 0.0) - after_s.get("mean_genus_f1", 0.0))
        run_id = str(p.run_id.iloc[0])
        cfg_file = ROOT / "runs" / run_id / "config.yaml"
        cfg = OmegaConf.load(cfg_file) if cfg_file.exists() else None
        after_b = table(p, "adapted")
        n_val = int(after_b.loc[after_b.day == "all", "n"].iloc[0]) if "n" in after_b and len(after_b) else 0
        out.append({"draw": draw, "shots": int(p.shots.iloc[0]), "run_id": run_id,
                    "verdict": "ok" if regression <= max_regression else "source regression",
                    "before_b": table(p, "zero_shot"), "after_b": after_b, "before_s": before_s, "after_s": after_s,
                    "n_train": int(cfg.get("n_train_b0268", 0)) if cfg is not None else "?", "n_val": n_val,
                    "train_days": list(cfg.get("train_days", [])) if cfg is not None else "?",
                    "val_days": list(cfg.get("val_days", [])) if cfg is not None else "?"})
    return out


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    t0 = time.perf_counter()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", default="k1_d0", help="a few-shot draw id, or 'all'")
    ap.add_argument("--final", action="store_true", help="train on every adaptation day (the demo candidate)")
    ap.add_argument("--init", default=DEFAULT_INIT)
    ap.add_argument("--replay", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs-lp", type=int, default=5)
    ap.add_argument("--epochs-ft", type=int, default=10)
    ap.add_argument("--lr-lp", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--p-drop", type=float, default=0.3)
    ap.add_argument("--lambda-h", type=float, default=0.1)
    ap.add_argument("--max-regression", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase", default="P084")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--tag", default="base", help="configuration name kept in the results table (one row set per tag)")
    ap.add_argument("--doc-only", action="store_true",
                    help="rewrite docs/data/b0268_adaptation.md for --tag from data/b0268_adaptation.parquet; no training")
    a = ap.parse_args()
    if a.doc_only:
        df = pd.read_parquet(OUT)
        write_doc(results_from_rows(df, a.tag, a.max_regression), a, df)
        print(f"{DOC} rewritten for `{a.tag}` from {OUT}")
        return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = load_paths()
    data_root, cache_root = Path(paths["data_root"]), Path(paths["cache_root"])

    few = pd.read_parquet("data/splits/few_shot_target.parquet")
    draws = ["k0_d0"] if a.final else (sorted(few.draw.unique()) if a.draw == "all" else [a.draw])
    rows_all = D.b0268_rows()
    grid_b, cls_b = b0268_caches(cache_root, rows_all, a.build_cache)
    rows_s = D.global_rows(eye2sky_etage=True)
    grid_s = FeatureCache(cache_root, "dinov3_vits16", D.SIZE, transform="ffr1-patches-all-pool4")
    cls_s = FeatureCache(cache_root, "dinov3_vits16", D.SIZE)
    have = set(grid_s.index().sample_id)
    rows_s = rows_s[rows_s.sample_id.isin(have)].reset_index(drop=True)
    geo_s = D.GeometryCache.build(rows_s, data_root, cache_root / "geometry" / f"pool4_adapt_source_{len(rows_s)}.npz")
    ds_s = dataset_for(rows_s, grid_s, cls_s, geo_s)
    shared = {"data_root": data_root, "cache_root": cache_root, "grid_b": grid_b, "cls_b": cls_b, "few": few,
              "d_model": int(ds_s.grid.shape[-1]), "ds_s": ds_s, "tr_s": np.flatnonzero(rows_s.split == "train"),
              "va_s": np.flatnonzero(rows_s.split == "val"),
              "raters": E.rater_targets(pd.read_parquet("data/b0268_derived.parquet")), "ceiling": ceiling("AB")}
    print(f"{len(rows_all)} labelled B0268 frames; source rows {len(rows_s):,} ({len(shared['tr_s']):,} train, "
          f"{len(shared['va_s']):,} val); draws {draws}{' (final)' if a.final else ''}; {device}", flush=True)

    records, results = [], []
    for draw in draws:
        rec, res = run_draw(draw, a, shared, device)
        records += rec
        results.append(res)
        b = res["after_b"]
        line = "" if not len(b) else f"étage set {row(b, 'etage_set_agreement')} oktas ±1 {row(b, 'oktas_within1')}"
        print(f"  {draw}: {res['verdict']}; after: {line}; source étage F1 {res['after_s']['mean_etage_f1']:.3f} "
              f"(before {res['before_s']['mean_etage_f1']:.3f}) | {time.perf_counter() - t0:.0f} s", flush=True)
    df = pd.DataFrame(records)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        if "tag" not in old:
            old["tag"] = "base"
        keep = ~((old.tag == a.tag) & old.draw.isin(df.draw.unique()))
        df = pd.concat([old[keep], df], ignore_index=True)
    df.to_parquet(OUT, index=False)
    if not a.final and len(results):
        write_doc([r for r in results if r["draw"] != "k0_d0" or len(results) == 1], a, df)
    print(f"done in {time.perf_counter() - t0:.0f} s: {OUT}, {DOC}", flush=True)
    return 0


def row(df: pd.DataFrame, key: str) -> str:
    v = df[df.day == "all"].iloc[0].get(key)
    return "-" if v is None or np.isnan(v) else f"{v:.3f}"


if __name__ == "__main__":
    sys.exit(main())
