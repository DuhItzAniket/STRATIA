"""Scoring a model's B0268 predictions against the P041 rater targets (P083 zero-shot, P084 adaptation).

The targets are the per-frame soft targets of `stratia.labels.b0268_labels` (shares over the raters). A model is
scored the way criterion C2 scores Montenegro: exact agreement with the rater majority on the étage set (sigmoid
> 0.5 per étage) and total cover within ±1 okta (argmax of the oktas distribution against the majority okta), with
the raters' own agreement as the ceiling; plus the top-1 genus hit against the raters' union, the sky-parse cloud
fraction against the cover, the glare pixels against the raters' glare, and the OOD score.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..baselines import tasks as T
from ..contract import ETAGE_CLASSES, GENUS_CLASSES

N_GENERA = 10            # the first ten contract classes are genera; then 'clear' and 'contrail'


def majority_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """Per frame: the étage majority vector (share >= 0.5) and shares, the majority okta (argmax of the distribution,
    the lower okta on ties) and the mean okta, the genus union (any rater) and majority, clear and strong-glare
    majorities. Frames without usable annotations are dropped."""
    rows = []
    for t in targets.itertuples(index=False):
        if not isinstance(t.etage_share, str):
            continue
        es = json.loads(t.etage_share)
        gs = json.loads(t.genus_share) if isinstance(t.genus_share, str) else {}
        od = json.loads(t.oktas_dist) if isinstance(t.oktas_dist, str) else {}
        dist = np.zeros(9)
        for k, v in od.items():
            dist[int(k)] = v
        total = float(dist.sum())
        rows.append({"item_id": t.item_id, "day": t.day, "image_file": t.image_file,
                     "etage_majority": np.array([es.get(e, 0.0) >= 0.5 for e in ETAGE_CLASSES], dtype=bool),
                     "etage_share": np.array([es.get(e, 0.0) for e in ETAGE_CLASSES], dtype=np.float32),
                     "oktas_majority": int(dist.argmax()) if total > 0 else None,
                     "oktas_mean": float((dist * np.arange(9)).sum() / total) if total > 0 else None,
                     "genus_union": [g for g, s in gs.items() if s > 0], "genus_majority": [g for g, s in gs.items() if s >= 0.5],
                     "clear": bool((t.clear_share or 0.0) >= 0.5), "glare_strong": bool((t.glare_strong_share or 0.0) >= 0.5),
                     "cloud_share": t.cloud_share})
    return pd.DataFrame(rows)


def score(pred: dict[str, np.ndarray], truth: pd.DataFrame, ceiling: dict[str, float] | None = None,
          ood_threshold: float | None = None) -> dict[str, float]:
    """`pred` holds, aligned with `truth` rows: etage [n, 3] and genus [n, 12] probabilities, oktas [n, 9]
    probabilities; optionally cloud_fraction [n] (sky-parse cloud share of the valid sky), glare_fraction [n] and
    ood [n]. `ceiling` = the raters' leave-one-out agreement (etage_set, oktas_within1) for the C2-style ratios."""
    n = len(truth)
    out: dict[str, float] = {"n": float(n)}
    e_true = np.stack(truth.etage_majority.to_numpy()).astype(int)
    e_hat = (pred["etage"] > 0.5).astype(int)
    out["etage_set_agreement"] = float((e_true == e_hat).all(1).mean())
    out["etage_f1"] = T.multilabel_macro_f1(e_true, e_hat) if e_true.any() else float("nan")
    for i, e in enumerate(ETAGE_CLASSES):
        out[f"etage_{e}_acc"] = float((e_true[:, i] == e_hat[:, i]).mean())
        out[f"etage_{e}_pred_rate"] = float(e_hat[:, i].mean())
        out[f"etage_{e}_true_rate"] = float(e_true[:, i].mean())
    ok = truth.oktas_majority.notna().to_numpy()
    if ok.any():
        o_true = truth.oktas_majority.to_numpy()[ok].astype(int)
        o_hat = pred["oktas"][ok].argmax(1)
        out["oktas_within1"] = float((np.abs(o_hat - o_true) <= 1).mean())
        out["oktas_mae"] = float(np.abs(o_hat - o_true).mean())
        out["oktas_bias"] = float((o_hat - o_true).mean())
        out["oktas_pred_mean"] = float(o_hat.mean())
        out["oktas_true_mean"] = float(o_true.mean())
    top = [GENUS_CLASSES[k] for k in pred["genus"][:, :N_GENERA].argmax(1)]
    has = truth.genus_union.map(len).to_numpy() > 0
    if has.any():
        out["genus_top1_in_union"] = float(np.mean([t in u for t, u, h in zip(top, truth.genus_union, has, strict=True) if h]))
        maj = truth.genus_majority.map(len).to_numpy() > 0
        if maj.any():
            out["genus_top1_in_majority"] = float(np.mean([t in u for t, u, h in zip(top, truth.genus_majority, maj,
                                                                                      strict=True) if h]))
    if "cloud_fraction" in pred:
        cf = np.asarray(pred["cloud_fraction"], dtype=float)
        cover = truth.oktas_mean.to_numpy(dtype=float) / 8.0
        both = np.isfinite(cf) & np.isfinite(cover)
        if both.sum() > 2:
            out["cloud_fraction_mae_vs_cover"] = float(np.abs(cf[both] - cover[both]).mean())
            out["cloud_fraction_bias_vs_cover"] = float((cf[both] - cover[both]).mean())
            out["cloud_fraction_corr_vs_cover"] = float(np.corrcoef(cf[both], cover[both])[0, 1]) if cover[both].std() > 0 \
                else float("nan")
        low = ok & (truth.oktas_majority.fillna(9).to_numpy().astype(int) <= 1)
        if low.any():
            out["cloud_fraction_on_le1_okta"] = float(np.nanmean(cf[low]))
            out["n_le1_okta"] = float(low.sum())
    if "glare_fraction" in pred:
        gf = np.asarray(pred["glare_fraction"], dtype=float)
        strong = truth.glare_strong.to_numpy(dtype=bool)
        if strong.any():
            out["glare_fraction_strong_frames"] = float(np.nanmean(gf[strong]))
        if (~strong).any():
            out["glare_fraction_other_frames"] = float(np.nanmean(gf[~strong]))
    if "ood" in pred:
        ood = np.asarray(pred["ood"], dtype=float)
        out["ood_mean"] = float(np.nanmean(ood))
        out["ood_p90"] = float(np.nanpercentile(ood, 90))
        if ood_threshold is not None:
            out["ood_flagged_share"] = float(np.mean(ood > ood_threshold))
    if ceiling:
        out["c2_etage_ratio"] = out["etage_set_agreement"] / ceiling["etage_set"]
        if "oktas_within1" in out:
            out["c2_oktas_ratio"] = out["oktas_within1"] / ceiling["oktas_within1"]
    return out


def rater_targets(derived: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per rater, from the per-annotation table of `stratia.labels.b0268_labels.derive` (usable annotations only):
    item_id -> etage bool[3], oktas (int, None when obscured), genus (list; [] = no genus named; None = 'unsure'),
    glare_strong, clear. The model is scored against each rater and the raters' pairwise agreement is the ceiling,
    so neither side depends on how ties between two raters are broken."""
    out = {}
    for rater, part in derived[derived.usable.astype(bool)].groupby("user_id"):
        rows = []
        for r in part.itertuples(index=False):
            etages = r.etage_set.split("|") if isinstance(r.etage_set, str) and r.etage_set else []
            if isinstance(r.genus_set, str):
                genus = r.genus_set.split("|") if r.genus_set else []
            else:
                genus = None
            rows.append({"item_id": r.item_id, "etage": np.array([e in etages for e in ETAGE_CLASSES], dtype=bool),
                         "oktas": None if pd.isna(r.oktas) else int(r.oktas), "genus": genus,
                         "glare_strong": r.glare == "strong", "clear": r.extra == "clear"})
        out[str(rater)] = pd.DataFrame(rows).set_index("item_id")
    return out


def score_against_raters(pred: dict[str, np.ndarray], item_ids, raters: dict[str, pd.DataFrame],
                         ceiling: dict[str, float] | None = None) -> dict[str, float]:
    """The model against each rater on the frames `item_ids` (aligned with `pred`), averaged over raters:
    etage_set_agreement, oktas_within1 / oktas_mae / oktas_bias, genus_top1_in_set (the top genus is in that rater's
    set; frames where the rater named no genus or was unsure are skipped). Per-rater values are returned as
    `<metric>_<rater>`; `n_raters` counts the raters. With `ceiling` (the raters' own pairwise agreement) the
    C2-style ratios are added."""
    e_hat = pred["etage"] > 0.5
    o_hat = pred["oktas"].argmax(1)
    top = [GENUS_CLASSES[k] for k in pred["genus"][:, :N_GENERA].argmax(1)]
    per: dict[str, dict[str, float]] = {}
    for rater, table in raters.items():
        t = table.reindex(list(item_ids))
        ok = t.etage.notna().to_numpy()
        if not ok.any():
            continue
        e_true = np.stack(t.etage[ok].to_numpy()).astype(bool)
        m = {"etage_set_agreement": float((e_true == e_hat[ok]).all(1).mean())}
        okt = ok & t.oktas.notna().to_numpy()
        if okt.any():
            o_true = t.oktas[okt].to_numpy().astype(int)
            m["oktas_within1"] = float((np.abs(o_hat[okt] - o_true) <= 1).mean())
            m["oktas_mae"] = float(np.abs(o_hat[okt] - o_true).mean())
            m["oktas_bias"] = float((o_hat[okt] - o_true).mean())
        hits = [tp in s for tp, s, k in zip(top, t.genus, ok, strict=True) if k and isinstance(s, list) and s]
        if hits:
            m["genus_top1_in_set"] = float(np.mean(hits))
        per[rater] = m
    out: dict[str, float] = {"n": float(len(list(item_ids))), "n_raters": float(len(per))}
    metrics = sorted({k for m in per.values() for k in m})
    for metric in metrics:
        vals = [m[metric] for m in per.values() if metric in m]
        out[metric] = float(np.mean(vals))
        for rater, m in per.items():
            if metric in m:
                out[f"{metric}_{rater}"] = m[metric]
    if ceiling and "etage_set_agreement" in out:
        out["c2_etage_ratio"] = out["etage_set_agreement"] / ceiling["etage_set"]
    if ceiling and "oktas_within1" in out:
        out["c2_oktas_ratio"] = out["oktas_within1"] / ceiling["oktas_within1"]
    return out


def per_day_against_raters(pred: dict[str, np.ndarray], item_ids, days, raters: dict[str, pd.DataFrame],
                           ceiling: dict[str, float] | None = None) -> pd.DataFrame:
    """`score_against_raters` over all frames and per day (`days` aligned with `item_ids`)."""
    ids, dd = np.asarray(list(item_ids)), np.asarray(list(days))
    rows = [{"day": "all", **score_against_raters(pred, ids, raters, ceiling)}]
    for day in sorted(set(dd)):
        sel = dd == day
        rows.append({"day": day, **score_against_raters({k: np.asarray(v)[sel] for k, v in pred.items()}, ids[sel], raters,
                                                        ceiling)})
    return pd.DataFrame(rows)


def per_day(pred: dict[str, np.ndarray], truth: pd.DataFrame, ceiling: dict[str, float] | None = None,
            ood_threshold: float | None = None) -> pd.DataFrame:
    """`score` over all rows and per day; one row per group."""
    rows = [{"day": "all", **score(pred, truth, ceiling, ood_threshold)}]
    for day, part in truth.groupby("day", sort=True):
        sel = (truth.day == day).to_numpy()
        rows.append({"day": day, **score({k: np.asarray(v)[sel] for k, v in pred.items()}, part.reset_index(drop=True),
                                         ceiling, ood_threshold)})
    return pd.DataFrame(rows)
