"""B0268 rater sheets -> derived labels, agreement and soft targets (P041).

The owner's Arducam B0268 frames were labelled per image with the sheet of `DATA/B0268/labels/` (one row per
frame: usable, oktas 0-9, the genera of each étage as drop-down combinations, the étage of the lowest cloud, Sun
glare, obstruction, notes). This module reads such sheets (CSV copies, one per rater), translates every row into
the labels STRATIA predicts (étage set, genus set, oktas, obscured, cloud, clear / contrail), measures the raters'
agreement with the P035 machinery (Krippendorff's alpha, pairwise, leave-one-rater-out) and writes per-frame soft
targets in the Montenegro format (P035), so training and evaluation treat both cameras alike.

Rules (the sheet's guide tab is authoritative):
* `usable = no` leaves every other column empty: the row counts for nothing but the usable share.
* A genus combination names co-present genera (Cu+Sc = {Cu, Sc}); 'none' = no cloud of that étage; 'unsure' = cloud
  of that étage present, genus unknown: the étage counts as present and the rater's genus set is unknown (not
  partial), so it is left out of the genus shares, never guessed.
* 'contrail' in the high column is the extra class, not a genus (the ontology keeps it apart); the high étage counts
  as present.
* oktas 9 = sky obscured (no cover estimate); oktas 0 with every column 'none' = clear.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import agreement as ag

RATER_COLUMNS = ("usable", "oktas", "low_cloud", "mid_cloud", "high_cloud", "lowest_base_guess", "sun_glare",
                 "obstruction", "notes")
VOCABULARY = {
    "usable": ("yes", "no"),
    "oktas": tuple(str(i) for i in range(10)),
    "low_cloud": ("none", "Cu", "Sc", "St", "Cb", "Cu+Sc", "unsure"),
    "mid_cloud": ("none", "Ac", "As", "Ns", "Ac+As", "unsure"),
    "high_cloud": ("none", "Ci", "Cs", "Cc", "Ci+Cs", "contrail", "unsure"),
    "lowest_base_guess": ("low", "mid", "high", "none"),
    "sun_glare": ("none", "mild", "strong"),
    "obstruction": ("none", "raindrops", "dirt", "object"),
}
ETAGE_COLUMNS = {"low": "low_cloud", "mid": "mid_cloud", "high": "high_cloud"}
GENERA_OF = {"Cu": ["Cu"], "Sc": ["Sc"], "St": ["St"], "Cb": ["Cb"], "Cu+Sc": ["Cu", "Sc"],
             "Ac": ["Ac"], "As": ["As"], "Ns": ["Ns"], "Ac+As": ["Ac", "As"],
             "Ci": ["Ci"], "Cs": ["Cs"], "Cc": ["Cc"], "Ci+Cs": ["Ci", "Cs"], "contrail": []}
GLARE_RANK = {"none": 0, "mild": 1, "strong": 2}


# ------------------------------------------------------------------------------------------------- loading
def load_sheet(csv_path: str | Path, rater: str, frames: pd.DataFrame) -> pd.DataFrame:
    """One rater's CSV (idx 1..n + the sheet columns) joined with the frame list (frame_id, day, file) as a long
    table with item_id = frame_id and user_id = rater. Values are validated against the sheet's vocabulary."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in ("idx", *RATER_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    df["idx"] = df.idx.astype(int)
    if sorted(df.idx) != list(range(1, len(frames) + 1)):
        raise ValueError(f"{csv_path}: expected idx 1..{len(frames)}, got {len(df)} rows")
    for col, allowed in VOCABULARY.items():
        bad = df[(df[col] != "") & ~df[col].isin(allowed)]
        if len(bad):
            raise ValueError(f"{csv_path}: {col} has values outside the vocabulary: {sorted(bad[col].unique())[:5]}")
    if (df.usable == "").any():
        raise ValueError(f"{csv_path}: 'usable' must be yes or no on every row ({int((df.usable == '').sum())} blank)")
    unusable = df.usable == "no"
    filled = df.loc[unusable, ["oktas", "low_cloud", "mid_cloud", "high_cloud", "lowest_base_guess", "sun_glare"]]
    if (filled != "").any().any():
        raise ValueError(f"{csv_path}: an unusable row carries labels")
    if (df.loc[~unusable, ["oktas", "low_cloud", "mid_cloud", "high_cloud"]] == "").any().any():
        raise ValueError(f"{csv_path}: a usable row misses oktas or a genus column")
    out = df.merge(frames.reset_index(drop=True).assign(idx=np.arange(1, len(frames) + 1)), on="idx", how="left")
    out["item_id"] = out.frame_id
    out["user_id"] = rater
    return out


def load_frames(labels_dir: str | Path) -> pd.DataFrame:
    """The frame list of the labelling folder (`B0268_frames.csv`: frame_id, day, local time, Sun, file)."""
    return pd.read_csv(Path(labels_dir) / "B0268_frames.csv", dtype={"frame_id": str, "day": str, "file": str})


def load_raters(labels_dir: str | Path, raters: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    frames = load_frames(labels_dir)
    parts = [load_sheet(Path(labels_dir) / f"B0268_labels_{r}.csv", r, frames) for r in raters]
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------------------------------------- derivation
def derive_row(r) -> dict:
    """One annotation as the labels STRATIA predicts (None = unknown / not applicable)."""
    if r.usable != "yes":
        return {"etage_set": None, "genus_set": None, "oktas": None, "obscured": False, "h_band": None, "cloud": None,
                "extra": None, "contrail": None, "glare": None, "obstruction": r.obstruction or "none", "usable": False}
    etages, genera, unsure, contrail = [], set(), False, False
    for etage, col in ETAGE_COLUMNS.items():
        v = getattr(r, col)
        if v == "none":
            continue
        etages.append(etage)
        if v == "unsure":
            unsure = True
        elif v == "contrail":
            contrail = True
        else:
            genera |= set(GENERA_OF[v])
    oktas = int(r.oktas)
    obscured = oktas == 9
    clear = oktas == 0 and not etages
    extra = "clear" if clear else ("contrail" if contrail and not genera else None)   # the P034 'extra' class: exclusive
    genus_set = None if unsure else "|".join(sorted(genera))
    return {"etage_set": "|".join(sorted(etages, key=["low", "mid", "high"].index)),
            "genus_set": genus_set, "oktas": None if obscured else oktas, "obscured": obscured, "h_band": None,
            "cloud": None if obscured else (oktas > 0 or bool(etages)), "extra": extra,
            "contrail": contrail,                                                      # a contrail next to genera counts too
            "glare": r.sun_glare, "obstruction": r.obstruction or "none", "usable": True}


def derive(long: pd.DataFrame) -> pd.DataFrame:
    """Per annotation: the P035 derived columns (étage set, genus set, oktas, obscured, h_band, cloud) plus extra,
    glare, obstruction and usable."""
    out = long[["item_id", "user_id"]].copy()
    rows = [derive_row(r) for r in long.itertuples(index=False)]
    for k in rows[0]:
        out[k] = [row[k] for row in rows]
    return out


# ------------------------------------------------------------------------------------------------- targets
def soft_targets(derived: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Per frame: the P035 shares and distributions (étage, genus, oktas, obscured, cloud) from the usable
    annotations, plus the clear and contrail shares, the glare and obstruction shares and the usable share."""
    usable = derived[derived.usable]
    base = ag.soft_targets(usable) if len(usable) else pd.DataFrame(columns=["item_id"])
    rows = []
    for item, part in derived.groupby("item_id"):
        u = part[part.usable]
        rows.append({"item_id": item, "n_annotators": int(part.user_id.nunique()), "usable_share": float(part.usable.mean()),
                     "clear_share": float((u.extra == "clear").mean()) if len(u) else None,
                     "contrail_share": float(u.contrail.astype(bool).mean()) if len(u) else None,
                     "glare_strong_share": float((u.glare == "strong").mean()) if len(u) else None,
                     "glare_any_share": float(u.glare.isin(["mild", "strong"]).mean()) if len(u) else None,
                     "obstruction_share": float((part.obstruction != "none").mean())})
    extra = pd.DataFrame(rows)
    out = extra.merge(base, on="item_id", how="left")
    meta = frames.rename(columns={"frame_id": "item_id"})[["item_id", "day", "file"]]
    out = out.merge(meta, on="item_id", how="left")
    out["image_file"] = "B0268/" + out.file
    return out.drop(columns=["file"])


def majority_sets(targets: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Hard labels for the P034 label table: étages and genera named by at least `threshold` of the raters, cloud
    presence by the cloud share, clear by the clear share. With two raters the default threshold makes the sets the
    UNION of the pair (one vote suffices); the label table is a convenience view, training and evaluation use the
    shares and the per-rater labels. Rows without usable annotations give None."""
    rows = []
    for t in targets.itertuples(index=False):
        if t.etage_share is None or (isinstance(t.etage_share, float) and np.isnan(t.etage_share)):
            rows.append({"item_id": t.item_id, "etage_set": None, "genus_set": None, "cloud_present": None, "extra": None})
            continue
        etages = [e for e, s in json.loads(t.etage_share).items() if s >= threshold]
        genus = json.loads(t.genus_share) if isinstance(t.genus_share, str) else {}
        genera = [g for g, s in genus.items() if s >= threshold]
        cloud = None if t.cloud_share is None or (isinstance(t.cloud_share, float) and np.isnan(t.cloud_share)) \
            else bool(t.cloud_share >= threshold)
        clear = t.clear_share is not None and t.clear_share >= threshold
        contrail = t.contrail_share is not None and t.contrail_share >= threshold and not genera
        extra = "clear" if clear else ("contrail" if contrail else None)
        rows.append({"item_id": t.item_id, "etage_set": [] if clear else etages,
                     "genus_set": None if (clear or contrail or not genera) else genera,
                     "cloud_present": False if clear else cloud, "extra": extra})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- agreement
def agreement_table(long: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    """alpha, pairwise and leave-one-rater-out agreement for the raw sheet columns and the derived labels, over
    the usable annotations. With two raters leave-one-out equals pairwise agreement."""
    usable_ids = derived[derived.usable].index
    raw = long.loc[usable_ids]
    der = derived.loc[usable_ids]
    rows = []

    def add(name, table, level, kind, tolerance=None, note=""):
        loo = ag.leave_one_out(table, kind=kind, tolerance=tolerance if tolerance is not None else 0.0, min_others=1)
        rows.append({"quantity": name, "level": level, "units": int((table.notna().sum(axis=1) >= 2).sum()),
                     "alpha": ag.krippendorff_alpha(table, level), "pairwise": ag.pairwise_agreement(table, tolerance),
                     "leave_one_out": float(loo[loo.rater == "all"].agreement.iloc[0]), "note": note})

    okt = raw.assign(okt=pd.to_numeric(raw.oktas, errors="coerce").where(lambda s: s <= 8))
    add("oktas 0-8", ag.pivot(okt, "okt"), "interval", "numeric", 1.0, "pairwise and leave-one-out within ±1 okta")
    add("oktas 0-8, exact", ag.pivot(okt, "okt"), "interval", "numeric", 0.0)
    for col in ("low_cloud", "mid_cloud", "high_cloud", "lowest_base_guess", "obstruction"):
        add(f"raw {col}", ag.pivot(raw, col), "nominal", "nominal")
    glare = raw.assign(gl=raw.sun_glare.map(GLARE_RANK))
    add("raw sun_glare", ag.pivot(glare, "gl"), "ordinal", "numeric", 0.0, "none < mild < strong")
    add("étage set", ag.pivot(der, "etage_set"), "nominal", "nominal", note="exact set from the three genus columns")
    for e in ag.ETAGES:
        add(f"étage {e} present", ag._flag_table(der, "etage_set", e), "nominal", "nominal")
    add("genus set", ag.pivot(der, "genus_set"), "nominal", "nominal", note="exact set; 'unsure' rows excluded")
    for g in ag.GENERA:
        add(f"genus {g} present", ag._flag_table(der, "genus_set", g), "nominal", "nominal")
    cloud = der.assign(c=der.cloud.map(lambda v: np.nan if v is None else float(v)))
    add("cloud present (oktas > 0)", ag.pivot(cloud, "c"), "nominal", "nominal")
    clear = der.assign(c=np.where(der.cloud.isna(), np.nan, (der.extra == "clear").astype(float)))   # obscured: unknown
    add("clear", ag.pivot(clear, "c"), "nominal", "nominal")
    return pd.DataFrame(rows)


def per_rater(derived: pd.DataFrame) -> pd.DataFrame:
    """Each rater against the others on the étage set (exact) and total cover (±1 okta)."""
    der = derived[derived.usable]
    loo_e = ag.leave_one_out(ag.pivot(der, "etage_set"), "nominal", min_others=1)
    loo_n = ag.leave_one_out(ag.pivot(der, "oktas"), "numeric", tolerance=1.0, min_others=1)
    return (loo_e.rename(columns={"units": "units_etage", "agreement": "etage"})
            .merge(loo_n.rename(columns={"units": "units_n", "agreement": "n"}), on="rater", how="outer").fillna(0))


def report_markdown(primary: pd.DataFrame, per_rater_primary: pd.DataFrame, extra: pd.DataFrame | None,
                    raters: tuple[str, ...], extra_raters: tuple[str, ...], n_frames: int, days: dict[str, int],
                    decision: list[str]) -> str:
    def f(x):
        return "n/a" if pd.isna(x) else f"{x:.3f}"

    lines = ["# B0268 labels and rater agreement (P041)", "",
             f"{n_frames:,} frames of the owner's Arducam B0268 (sky logger, 4656 x 3496, zenith mount, site BLR01) over "
             + ", ".join(f"{d} ({n})" for d, n in sorted(days.items()))
             + f", labelled per image by {len(raters)} annotators ({', '.join(raters)}) with the sheet of "
             "`DATA/B0268/labels/` and translated into the labels STRATIA predicts by `stratia/labels/b0268_labels.py`. "
             "alpha = Krippendorff's alpha (1 = perfect, 0 = chance); pairwise = share of rater pairs that agree; "
             "leave-one-out = a rater against the majority (or median) of the others, the bar a model is compared with "
             "(the B0268 counterpart of criterion C2).", "",
             f"## Annotators {' and '.join(raters)} (the P041 pair)", "",
             "| Quantity | Level | Units | alpha | Pairwise | Leave-one-out | Note |", "|---|---|---|---|---|---|---|"]
    for _, r in primary.iterrows():
        lines.append(f"| {r.quantity} | {r.level} | {int(r.units):,} | {f(r.alpha)} | {f(r.pairwise)} | "
                     f"{f(r.leave_one_out)} | {r.note} |")
    lines += ["", "### Raters against the others (étage set and total cover)", "",
              "| Rater | Units (étage) | Étage agreement | Units (oktas) | Oktas within ±1 |", "|---|---|---|---|---|"]
    for _, r in per_rater_primary.iterrows():
        lines.append(f"| {r.rater} | {int(r.units_etage):,} | {f(r.etage)} | {int(r.units_n):,} | {f(r.n)} |")
    if extra is not None and len(extra):
        lines += ["", f"## All annotators ({', '.join(raters + extra_raters)}; sensitivity, not the protocol pair)", "",
                  "| Quantity | Level | Units | alpha | Pairwise | Leave-one-out |", "|---|---|---|---|---|---|"]
        for _, r in extra.iterrows():
            lines.append(f"| {r.quantity} | {r.level} | {int(r.units):,} | {f(r.alpha)} | {f(r.pairwise)} | "
                         f"{f(r.leave_one_out)} |")
    lines += ["", "## Decisions", ""] + [f"- {d}" for d in decision]
    return "\n".join(lines) + "\n"
