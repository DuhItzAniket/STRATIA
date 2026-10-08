"""Montenegro soft labels and the human ceiling (P035).

Krippendorff's alpha (nominal, ordinal, interval) over a units x raters table with missing values, raw pairwise
agreement, and leave-one-rater-out agreement against the majority of the other raters: the bar a model is held to
(criterion C2 compares model-rater agreement with rater-rater agreement on étage and total cloud cover). The raw
codes (altitude class, N, Nh, h, C_L, C_M, C_H) are also translated through the ontology into the labels STRATIA
predicts (étage set, genus set, oktas, height band) so that the ceiling is measured on the same quantities.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from stratia.labels.ontology import Ontology

ETAGES = ("low", "mid", "high")
GENERA = ("Ci", "Cc", "Cs", "Ac", "As", "Ns", "Sc", "St", "Cu", "Cb")


# ------------------------------------------------------------------------------------------ Krippendorff's alpha


def krippendorff_alpha(table: pd.DataFrame, level: str = "nominal") -> float:
    """alpha over a units (rows) x raters (columns) table; NaN / None = missing. Values are categories (nominal),
    ordered categories (ordinal: the metric uses the value ranks' marginal counts) or numbers (interval)."""
    values = {}
    for _, row in table.iterrows():
        vals = [v for v in row.tolist() if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if len(vals) >= 2:
            values[len(values)] = vals
    if not values:
        return float("nan")
    cats = sorted({v for vals in values.values() for v in vals}, key=lambda x: (isinstance(x, str), x))
    index = {c: k for k, c in enumerate(cats)}
    m = len(cats)
    o = np.zeros((m, m))
    for vals in values.values():
        mu = len(vals)
        counts = Counter(index[v] for v in vals)
        for c, nc in counts.items():
            for k, nk in counts.items():
                o[c, k] += (nc * (nk - (1 if c == k else 0))) / (mu - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    delta = np.zeros((m, m))
    for c in range(m):
        for k in range(m):
            if c == k:
                continue
            if level == "nominal":
                delta[c, k] = 1.0
            elif level == "interval":
                delta[c, k] = (float(cats[c]) - float(cats[k])) ** 2
            elif level == "ordinal":
                lo, hi = min(c, k), max(c, k)
                delta[c, k] = (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
            else:
                raise ValueError(f"unknown level {level!r}")
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return float("nan") if d_e == 0 else float(1 - d_o / d_e)


def pairwise_agreement(table: pd.DataFrame, tolerance: float | None = None) -> float:
    """Mean over units of the share of rater pairs that agree (exactly, or within `tolerance` for numbers)."""
    shares = []
    for _, row in table.iterrows():
        vals = [v for v in row.tolist() if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if len(vals) < 2:
            continue
        pairs = [(a, b) for i, a in enumerate(vals) for b in vals[i + 1:]]
        agree = [abs(float(a) - float(b)) <= tolerance if tolerance is not None else a == b for a, b in pairs]
        shares.append(float(np.mean(agree)))
    return float(np.mean(shares)) if shares else float("nan")


def leave_one_out(table: pd.DataFrame, kind: str = "nominal", tolerance: float = 1.0, min_others: int = 2) -> pd.DataFrame:
    """Each rater against the majority (nominal) or median (numeric, within `tolerance`) of the other raters of the
    same unit, over units where at least `min_others` others answered. Returns one row per rater and a 'all' row."""
    rows = []
    hits_all, n_all = 0, 0
    for rater in table.columns:
        hits, n = 0, 0
        for _, row in table.iterrows():
            v = row[rater]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            others = [row[c] for c in table.columns if c != rater]
            others = [x for x in others if x is not None and not (isinstance(x, float) and np.isnan(x))]
            if len(others) < min_others:
                continue
            if kind == "nominal":
                top, count = Counter(others).most_common(1)[0]
                if count * 2 <= len(others):          # no majority among the others: the unit does not count
                    continue
                hit = v == top
            else:
                hit = abs(float(v) - float(np.median([float(x) for x in others]))) <= tolerance
            hits += int(hit)
            n += 1
        if n:
            rows.append({"rater": str(rater), "units": n, "agreement": hits / n})
        hits_all += hits
        n_all += n
    rows.append({"rater": "all", "units": n_all, "agreement": hits_all / n_all if n_all else float("nan")})
    return pd.DataFrame(rows)


def pivot(long: pd.DataFrame, column: str, unit: str = "item_id", rater: str = "user_id") -> pd.DataFrame:
    """units x raters table of one coded variable (missing stays NaN)."""
    return long.pivot_table(index=unit, columns=rater, values=column, aggfunc="first")


# ------------------------------------------------------------------------------------------ derived labels


def derive(ann: pd.DataFrame, ont: Ontology) -> pd.DataFrame:
    """Per annotation: étage set (from the altitude class), genus set (from the C_L / C_M / C_H codes), oktas (N
    0-8; None when obscured or missing), obscured flag, height band (h code without '/')."""
    spec = ont.datasets["montenegro"]["altitude_class"]
    out = ann[["item_id", "user_id"]].copy()
    etage, genus, oktas, obscured, hband, cloud = [], [], [], [], [], []
    for r in ann.itertuples(index=False):
        ac = spec.get(r.altitude_class) if isinstance(r.altitude_class, str) else None
        e = set(ac.get("etages", [])) if ac else None
        g = set()
        any_code = False
        for table, code in (("CL", r.CL), ("CM", r.CM), ("CH", r.CH)):
            if isinstance(code, str) and code != "/":
                any_code = True
                g |= set(ont.genera_for_code(table, code) or [])
        n = r.N if isinstance(r.N, str) else None
        etage.append("|".join(sorted(e)) if e is not None else None)
        genus.append("|".join(sorted(g)) if (any_code or (ac and ac.get("extra") == "clear")) else None)
        oktas.append(int(n) if n is not None and n.isdigit() and int(n) <= 8 else None)
        obscured.append(n == "9")
        hband.append(r.h if isinstance(r.h, str) and r.h != "/" else None)
        cloud.append(None if n is None else (n != "0"))
    out["etage_set"], out["genus_set"], out["oktas"], out["obscured"], out["h_band"], out["cloud"] = (
        etage, genus, oktas, obscured, hband, cloud)
    return out


def soft_targets(derived: pd.DataFrame) -> pd.DataFrame:
    """Per item: share of raters naming each étage and each genus, oktas and height-band distributions, counts."""
    rows = []
    for item, part in derived.groupby("item_id"):
        e_rows = part.etage_set.dropna()
        g_rows = part.genus_set.dropna()
        etage_share = {e: float(np.mean([e in s.split("|") for s in e_rows])) for e in ETAGES} if len(e_rows) else None
        genus_share = {g: float(np.mean([g in s.split("|") for s in g_rows])) for g in GENERA} if len(g_rows) else None
        okt = part.oktas.dropna()
        h = part.h_band.dropna()
        rows.append({"item_id": item, "n_raters": int(part.user_id.nunique()),
                     "etage_share": json.dumps(etage_share) if etage_share else None,
                     "genus_share": json.dumps(genus_share) if genus_share else None,
                     "oktas_dist": json.dumps({str(int(k)): float(v) for k, v in okt.value_counts(normalize=True).sort_index().items()})
                     if len(okt) else None,
                     "obscured_share": float(part.obscured.mean()),
                     "h_band_dist": json.dumps({k: float(v) for k, v in h.value_counts(normalize=True).sort_index().items()})
                     if len(h) else None,
                     "cloud_share": float(part.cloud.dropna().mean()) if part.cloud.notna().any() else None})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ ceiling table


def ceiling_table(ann: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    """alpha, raw pairwise agreement and leave-one-rater-out agreement for the raw codes and the derived labels."""
    rows = []

    def add(name, table, level, kind, tolerance=None, note=""):
        loo = leave_one_out(table, kind=kind, tolerance=tolerance if tolerance is not None else 0.0)
        rows.append({"quantity": name, "level": level, "units": int((table.notna().sum(axis=1) >= 2).sum()),
                     "alpha": krippendorff_alpha(table, level), "pairwise": pairwise_agreement(table, tolerance),
                     "leave_one_out": float(loo[loo.rater == "all"].agreement.iloc[0]), "note": note})

    for col in ("altitude_class", "CL", "CM", "CH"):
        add(f"raw {col}", pivot(ann, col), "nominal", "nominal")
    n_num = ann.assign(Nn=ann.N.map(lambda v: int(v) if isinstance(v, str) and v.isdigit() and int(v) <= 8 else np.nan))
    add("raw N (oktas 0-8)", pivot(n_num, "Nn"), "interval", "numeric", 1.0, "pairwise and leave-one-out within ±1 okta")
    add("raw N (oktas 0-8), ordinal", pivot(n_num, "Nn"), "ordinal", "numeric", 1.0)
    h_num = ann.assign(hn=ann.h.map(lambda v: int(v) if isinstance(v, str) and v.isdigit() else np.nan))
    add("raw h (height code 0-9)", pivot(h_num, "hn"), "ordinal", "numeric", 0.0, "exact code; '/' excluded")
    add("raw h (height code 0-9), ±1 band", pivot(h_num, "hn"), "ordinal", "numeric", 1.0)
    add("étage set", pivot(derived, "etage_set"), "nominal", "nominal", note="exact set from the altitude class")
    for e in ETAGES:
        t = pivot(derived.assign(flag=derived.etage_set.map(lambda s, e=e: (e in s.split("|")) if isinstance(s, str) else np.nan)), "flag")
        add(f"étage {e} present", t, "nominal", "nominal")
    add("genus set", pivot(derived, "genus_set"), "nominal", "nominal", note="exact set from C_L / C_M / C_H")
    for g in GENERA:
        t = pivot(derived.assign(flag=derived.genus_set.map(lambda s, g=g: (g in s.split("|")) if isinstance(s, str) else np.nan)), "flag")
        add(f"genus {g} present", t, "nominal", "nominal")
    add("cloud present (N > 0)", pivot(derived.assign(c=derived.cloud.map(lambda v: np.nan if v is None else float(v))), "c"),
        "nominal", "nominal")
    return pd.DataFrame(rows)


def report_markdown(ceiling: pd.DataFrame, per_rater: pd.DataFrame, n_items: int, n_ann: int, decision: list[str]) -> str:
    def f(x):
        return "n/a" if pd.isna(x) else f"{x:.3f}"

    lines = ["# Montenegro soft labels and human ceiling (P035)", "",
             f"{n_ann:,} annotations by 9 raters on {n_items:,} images (P016), translated through `configs/ontology.yaml` "
             "into the labels STRATIA predicts. alpha = Krippendorff's alpha (1 = perfect, 0 = chance); pairwise = share of "
             "rater pairs that agree; leave-one-out = a rater against the majority (or median) of the other raters, the bar "
             "a model is compared with (criterion C2).", "",
             "| Quantity | Level | Units | alpha | Pairwise | Leave-one-out | Note |", "|---|---|---|---|---|---|---|"]
    for _, r in ceiling.iterrows():
        lines.append(f"| {r.quantity} | {r.level} | {int(r.units):,} | {f(r.alpha)} | {f(r.pairwise)} | {f(r.leave_one_out)} | {r.note} |")
    lines += ["", "## Raters against the others (étage set and total cover)", "",
              "| Rater | Units (étage) | Étage agreement | Units (N) | N within ±1 okta |", "|---|---|---|---|---|"]
    for _, r in per_rater.iterrows():
        lines.append(f"| {r.rater} | {int(r.units_etage):,} | {f(r.etage)} | {int(r.units_n):,} | {f(r.n)} |")
    lines += ["", "## Decisions", ""] + [f"- {d}" for d in decision]
    return "\n".join(lines) + "\n"
