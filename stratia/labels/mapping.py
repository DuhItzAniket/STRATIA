"""Dataset → ontology mapping (P034): every native label becomes a set-valued STRATIA label or is explicitly excluded.

One row per manifest sample (`data/labels.parquet`, joined on `sample_id`):
    genus_set           JSON list of genera the picture is known to contain; null when the genus is unknown
    genus_alternatives  True when the list names alternatives (a merged class such as MGCD "altocumulus" = Ac or Cc,
                        or a conflict merge) rather than genera known to be co-present
    extra               'clear' | 'contrail' | null
    etage_set           JSON list of étages present (from the genera, from Montenegro's altitude classes, or from
                        Almería's layer masks); null when unknown
    cloud_present       True / False / null
    label_source        'native class' | 'rater majority' | 'mask' | 'none'
    excluded_reason     why a sample carries no genus set (null when it carries one)
    label_conflict      True when P026 found the same picture under two labels and the sets were merged
Nothing is invented: a class whose genera the dataset does not name gives `genus_set` null and `cloud_present`
True, a SWIMCAT patch gives clear / cloud only, Eye2Sky and B0268 carry no image-level class yet.
"""

from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from stratia.labels.ontology import Ontology

CLASS_DATASETS = ("ccsn", "mgcd", "swimcat")
MASK_DATASETS = ("swimseg", "swinseg", "swinyseg", "shwimseg", "almeria")
MIN_LAYER_SHARE = 0.01       # an Almería layer counts as present when it covers at least 1 % of the labelled sky
COLUMNS = ["sample_id", "dataset", "genus_set", "genus_alternatives", "extra", "etage_set", "cloud_present",
           "label_source", "excluded_reason", "label_conflict"]


def _json(values) -> str | None:
    return None if values is None else json.dumps(sorted(values))


def map_class(ont: Ontology, dataset: str, label: str) -> dict:
    """A dataset's native class as genus set, extra class, étage set and cloud presence."""
    cls = ont.dataset_class(dataset, label)
    genera = cls.get("genera")
    extra = cls.get("extra")
    if genera is None:                                   # cloud of unnamed genus
        return {"genus_set": None, "genus_alternatives": False, "extra": None, "etage_set": None, "cloud_present": True,
                "excluded_reason": f"{dataset} class {label!r}: genus not named by the dataset"}
    if not genera:                                       # clear sky or contrail
        return {"genus_set": [], "genus_alternatives": False, "extra": extra, "etage_set": [],
                "cloud_present": extra != "clear", "excluded_reason": None}
    alternatives = bool(cls.get("alternatives"))
    etages = ont.etages_of(genera)
    known_etages = len(etages) == 1 or not alternatives          # alternatives spanning two étages: étage unknown
    return {"genus_set": list(genera), "genus_alternatives": alternatives, "extra": None,
            "etage_set": sorted(etages) if known_etages else None, "cloud_present": True, "excluded_reason": None}


def _majority_code(dist_json) -> str | None:
    if not isinstance(dist_json, str):
        return None
    d = json.loads(dist_json)
    return max(d.items(), key=lambda kv: kv[1])[0] if d else None


def map_montenegro(ont: Ontology, altitude_classes: str, cl_dist, cm_dist, ch_dist) -> dict:
    """Montenegro: étages from the majority altitude classes, genera from the majority C_L / C_M / C_H codes."""
    spec = ont.datasets["montenegro"]["altitude_class"]
    etages, extra, genera_hint = set(), None, []
    for token in (altitude_classes or "").split(","):
        token = token.strip()
        if not token or token not in spec:
            continue
        etages |= set(spec[token].get("etages", []))
        extra = extra or spec[token].get("extra")
        genera_hint += spec[token].get("genera", [])
    genera, alternatives, unknown = set(), False, False
    for table, dist in (("CL", cl_dist), ("CM", cm_dist), ("CH", ch_dist)):
        code = _majority_code(dist)
        if code is None:
            continue
        named = ont.genera_for_code(table, code)
        if named is None:
            unknown = True                                   # that part of the sky was not visible
            continue
        genera |= set(named)
        alternatives = alternatives or bool(ont.wmo_codes[table][str(code)].get("alternatives"))
    if extra == "clear" and not etages:
        return {"genus_set": [], "genus_alternatives": False, "extra": "clear", "etage_set": [], "cloud_present": False,
                "excluded_reason": None}
    if not genera:
        return {"genus_set": None, "genus_alternatives": False, "extra": None, "etage_set": sorted(etages) if etages else None,
                "cloud_present": True if etages else None,
                "excluded_reason": "montenegro: no genus named by the majority codes" + (" (not visible)" if unknown else "")}
    return {"genus_set": sorted(genera), "genus_alternatives": alternatives or bool(genera_hint), "extra": None,
            "etage_set": sorted(etages | ont.etages_of(genera)), "cloud_present": True, "excluded_reason": None}


def map_mask_row(dataset: str, cloud_fraction: float, low: float, mid: float, high: float) -> dict:
    """Segmentation datasets: cloud presence from the mask; Almería's layers give an étage set."""
    present = None if cloud_fraction != cloud_fraction else bool(cloud_fraction > MIN_LAYER_SHARE)
    etages = None
    if dataset == "almeria" and low == low:
        etages = [e for e, share in (("low", low), ("mid", mid), ("high", high)) if share >= MIN_LAYER_SHARE]
    return {"genus_set": None, "genus_alternatives": False, "extra": "clear" if present is False else None,
            "etage_set": [] if present is False else etages, "cloud_present": present,
            "excluded_reason": f"{dataset}: mask labels only (no genus)"}


def apply_mapping(manifest: pd.DataFrame, ont: Ontology, mask_fractions: pd.DataFrame | None = None,
                  conflicts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Labels for every manifest row; `mask_fractions` is P030's per-image table (dataset, image_file, cloud_fraction,
    low, mid, high); `conflicts` is P026's pair table (i, j manifest positions, kind, conflict, dataset_i)."""
    rows = []
    frac = None
    if mask_fractions is not None:
        frac = mask_fractions.set_index("image_file")
    for r in manifest.itertuples(index=False):
        base = {"sample_id": r.sample_id, "dataset": r.dataset, "label_source": "none", "label_conflict": False}
        if r.dataset in CLASS_DATASETS and isinstance(r.source_label, str):
            base.update(map_class(ont, r.dataset, r.source_label), label_source="native class")
        elif r.dataset == "montenegro":
            base.update(map_montenegro(ont, r.source_label, r.cl_dist, r.cm_dist, r.ch_dist), label_source="rater majority")
        elif r.dataset in MASK_DATASETS and frac is not None and r.image_file in frac.index:
            f = frac.loc[r.image_file]
            base.update(map_mask_row(r.dataset, f.cloud_fraction, f.low, f.mid, f.high), label_source="mask")
        else:
            base.update({"genus_set": None, "genus_alternatives": False, "extra": None, "etage_set": None,
                         "cloud_present": None, "excluded_reason": f"{r.dataset}: no image-level label"})
        rows.append(base)
    labels = pd.DataFrame(rows)
    if conflicts is not None and len(conflicts):
        _merge_conflicts(labels, conflicts)
    labels["genus_set"] = labels.genus_set.map(_json)
    labels["etage_set"] = labels.etage_set.map(_json)
    return labels[COLUMNS]


def _merge_conflicts(labels: pd.DataFrame, conflicts: pd.DataFrame) -> None:
    """Exact and copy pairs with different labels (P026): the pictures are the same, so each member takes the union
    of the genus sets as alternatives and is flagged; same-scene disagreements are left alone."""
    hard = conflicts[conflicts.conflict & conflicts.kind.isin(["exact", "copy"])]
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for i, j in zip(hard.i.values, hard.j.values, strict=True):
        a, b = find(int(i)), find(int(j))
        if a != b:
            parent[a] = b
    groups = defaultdict(list)
    for x in set(hard.i.values.tolist()) | set(hard.j.values.tolist()):
        groups[find(int(x))].append(int(x))
    for members in groups.values():
        union, any_known = set(), False
        for m in members:
            g = labels.at[m, "genus_set"]
            if g is not None:
                union |= set(g)
                any_known = True
        for m in members:
            if any_known:
                labels.at[m, "genus_set"] = sorted(union)
                labels.at[m, "genus_alternatives"] = True
                labels.at[m, "etage_set"] = None if len(labels.at[m, "genus_set"]) > 1 else labels.at[m, "etage_set"]
                labels.at[m, "extra"] = None if union else labels.at[m, "extra"]
            labels.at[m, "label_conflict"] = True
            labels.at[m, "label_source"] = "conflict merge"


def summary(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, part in labels.groupby("dataset", sort=True):
        cloudy = part.cloud_present.map(lambda v: v is True or v == 1)
        rows.append({"dataset": ds, "rows": len(part), "with_genus_set": int(part.genus_set.notna().sum()),
                     "alternatives": int(part.genus_alternatives.sum()), "with_etage_set": int(part.etage_set.notna().sum()),
                     "clear": int((part.extra == "clear").sum()), "contrail": int((part.extra == "contrail").sum()),
                     "cloud_unknown_genus": int((part.genus_set.isna() & cloudy).sum()),
                     "no_label": int(part.cloud_present.isna().sum()), "conflicts_merged": int(part.label_conflict.sum())})
    return pd.DataFrame(rows)


def mapping_table(ont: Ontology) -> pd.DataFrame:
    """Every native class of the class datasets with what it maps to (for the report)."""
    rows = []
    for ds in CLASS_DATASETS:
        for label, cls in ont.datasets[ds]["classes"].items():
            genera = cls.get("genera")
            rows.append({"dataset": ds, "native_class": label,
                         "maps_to": ("cloud, genus unknown" if genera is None else cls.get("extra") or
                                     (" or ".join(genera) if cls.get("alternatives") else " + ".join(genera))),
                         "etages": "" if genera is None else ", ".join(sorted(ont.etages_of(genera))) if genera else "none",
                         "note": cls.get("note", "")})
    for label, cls in ont.datasets["montenegro"]["altitude_class"].items():
        genera = " (" + " or ".join(cls["genera"]) + ")" if cls.get("genera") else ""
        rows.append({"dataset": "montenegro", "native_class": label,
                     "maps_to": cls.get("extra") or "étage " + ", ".join(cls["etages"]) + genera,
                     "etages": ", ".join(cls.get("etages", [])) or "none",
                     "note": "plus genera from the majority CL/CM/CH codes"})
    return pd.DataFrame(rows)


def report_markdown(table: pd.DataFrame, summ: pd.DataFrame, reasons: pd.Series, decision: list[str]) -> str:
    lines = ["# Label mapping report (P034)", "",
             "Every native label mapped through `configs/ontology.yaml` (P033) to a set-valued STRATIA label or excluded "
             "for a stated reason; output `data/labels.parquet` (one row per manifest sample).", "",
             "## Native classes and what they map to", "", "| Dataset | Native class | Maps to | Étages | Note |",
             "|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r.dataset} | {r.native_class} | {r.maps_to} | {r.etages} | {r.note} |")
    lines += ["", "## Coverage per dataset", "",
              "| Dataset | Rows | Genus set | of which alternatives | Étage set | Clear | Contrail | Cloud, genus unknown "
              "| No label | Conflicts merged |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        lines.append(f"| {r.dataset} | {r.rows:,} | {r.with_genus_set:,} | {r.alternatives:,} | {r.with_etage_set:,} | "
                     f"{r.clear:,} | {r.contrail:,} | {r.cloud_unknown_genus:,} | {r.no_label:,} | {r.conflicts_merged:,} |")
    lines += ["", "## Exclusion reasons", "", "| Reason | Rows |", "|---|---|"]
    for reason, n in reasons.items():
        lines.append(f"| {reason} | {n:,} |")
    lines += ["", "## Decisions", ""] + [f"- {d}" for d in decision]
    return "\n".join(lines) + "\n"
