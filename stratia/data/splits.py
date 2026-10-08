"""Split generator (P038): units, stratified assignment, protocols, verification and hashes, per docs/splits.md.

A unit is a connected component of "same group" (exact and near-duplicate groups) and "same block" (temporal
blocks). Units are assigned to train / val / test inside each source by a greedy rule that fills the split whose
stratum deficit the unit reduces most, so native-label shares stay balanced; conflict-merged images never enter
test. Leave-one-dataset-out folds and the held-out-station protocol are derived from the same tables, and
`verify` proves the guarantees before anything is written.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "splits.yaml"
SPLITS = ("train", "val", "test")


def load_split_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def source_of(dataset: pd.Series, config: dict) -> pd.Series:
    return dataset.map(lambda d: config.get("sources", {}).get(d, d))


# ------------------------------------------------------------------------------------------------- units


def block_keys(manifest: pd.DataFrame, config: dict) -> pd.Series:
    """Temporal block key per sample (None where the dataset's rule is 'group')."""
    rules = config["blocks"]
    utc = pd.to_datetime(manifest.utc, utc=True)
    keys = pd.Series([None] * len(manifest), index=manifest.index, dtype=object)
    for ds, part in manifest.groupby("dataset"):
        rule = rules.get(ds, rules.get("default", {"kind": "group"}))
        t = utc.loc[part.index]
        if rule["kind"] == "day":
            day = t.dt.strftime("%Y-%m-%d")
            keys.loc[part.index] = (f"{ds}:day:" + day).where(t.notna(), None)
        elif rule["kind"] == "days":
            start = t.min().normalize()
            idx = ((t - start).dt.days // int(rule["length_days"])).astype("Int64")
            keys.loc[part.index] = (f"{ds}:block:" + idx.astype(str)).where(t.notna(), None)
    return keys


def unit_ids(n: int, memberships: list[pd.Series]) -> np.ndarray:
    """Connected components over `n` samples, linking samples that share a non-null key in any of the series."""
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for keys in memberships:
        first = {}
        for pos, key in enumerate(keys.to_numpy()):
            if key is None or (isinstance(key, float) and np.isnan(key)):
                continue
            if key in first:
                a, b = find(first[key]), find(pos)
                if a != b:
                    parent[a] = b
            else:
                first[key] = pos
    roots = np.array([find(i) for i in range(n)])
    _, ids = np.unique(roots, return_inverse=True)
    return ids


# ------------------------------------------------------------------------------------------------- assignment


def assign_units(units: pd.DataFrame, fractions: dict, seed: int = 0) -> dict[int, str]:
    """Greedy stratified assignment of one source's units. `units` has columns unit, size, strata (Counter of
    stratum -> images) and test_ok (bool). Returns unit -> split."""
    rng = np.random.default_rng(seed)
    shuffled = units.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
    order = shuffled.sort_values("size", ascending=False, kind="stable")
    total = Counter()
    for c in units.strata:
        total.update(c)
    target = {s: {k: fractions[s] * v for k, v in total.items()} for s in SPLITS}
    current = {s: Counter() for s in SPLITS}
    out = {}
    for row in order.itertuples(index=False):
        choices = SPLITS if row.test_ok else ("train", "val")
        best, best_score = None, None
        for s in choices:
            score = sum(n * max(0.0, target[s][k] - current[s][k]) / max(target[s][k], 1.0) for k, n in row.strata.items())
            score += 1e-6 * (fractions[s] * row.size)           # tie-break towards the larger split
            if best_score is None or score > best_score:
                best, best_score = s, score
        out[int(row.unit)] = best
        current[best].update(row.strata)
    return out


def in_domain_split(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    """`table` has sample_id, source, unit, stratum, test_ok per sample; returns it with a `split` column."""
    out = table.copy()
    out["split"] = None
    fractions = config["fractions"]
    for k, (_source, part) in enumerate(table.groupby("source", sort=True)):
        rows = []
        for unit, g in part.groupby("unit"):
            rows.append({"unit": unit, "size": len(g), "strata": Counter(g.stratum.astype(str)),
                         "test_ok": bool(g.test_ok.all())})
        assignment = assign_units(pd.DataFrame(rows), fractions, seed=int(config.get("seed", 0)) + k)
        out.loc[part.index, "split"] = part.unit.map(assignment).values
    return out


def lodo_folds(in_domain: pd.DataFrame, config: dict) -> pd.DataFrame:
    """One row per (fold, sample): role 'test' for the held-out source, 'train' for train + val samples of the
    other sources whose unit has no member of the held-out source."""
    rows = []
    unit_sources = in_domain.groupby("unit").source.agg(set)
    for fold in config["lodo"]["folds"]:
        test = in_domain[in_domain.source == fold]
        tainted = {u for u, srcs in unit_sources.items() if fold in srcs}
        train = in_domain[(in_domain.source != fold) & in_domain.split.isin(["train", "val"]) & ~in_domain.unit.isin(tainted)]
        rows.append(pd.DataFrame({"fold": fold, "sample_id": test.sample_id.values, "role": "test"}))
        rows.append(pd.DataFrame({"fold": fold, "sample_id": train.sample_id.values, "role": "train"}))
    return pd.concat(rows, ignore_index=True)


def held_out_station(manifest: pd.DataFrame, in_domain: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Variants of the station protocol: same_days (train camera vs test camera, all days) and disjoint_days (the
    test camera on the in-domain val + test days, the train camera on the in-domain train days)."""
    spec = config["held_out_station"]
    part = (manifest[manifest.dataset == spec["dataset"]].drop(columns=["split"], errors="ignore")   # the manifest's own
            .merge(in_domain[["sample_id", "split"]], on="sample_id", how="left"))                     # empty split column
    rows = []
    for variant in spec["variants"]:
        if variant == "same_days":
            train = part[part.camera_id == spec["train_camera"]]
            test = part[part.camera_id == spec["test_camera"]]
        else:
            train = part[(part.camera_id == spec["train_camera"]) & (part.split == "train")]
            test = part[(part.camera_id == spec["test_camera"]) & part.split.isin(["val", "test"])]
        rows.append(pd.DataFrame({"variant": variant, "sample_id": train.sample_id.values, "role": "train"}))
        rows.append(pd.DataFrame({"variant": variant, "sample_id": test.sample_id.values, "role": "test"}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------------------- verification


def verify(in_domain: pd.DataFrame, lodo: pd.DataFrame, station: pd.DataFrame, manifest: pd.DataFrame,
           group_keys: list[pd.Series], blocks: pd.Series) -> list[str]:
    """The guarantees of docs/splits.md section 3; an empty list means all hold."""
    problems = []
    split_of = in_domain.set_index("sample_id").split
    sid = manifest.sample_id.values
    for name, keys in [("group", k) for k in group_keys] + [("block", blocks)]:
        df = pd.DataFrame({"key": keys.to_numpy(), "split": split_of.reindex(sid).values})
        df = df[df.key.notna()]
        bad = df.groupby("key").split.nunique()
        bad = bad[bad > 1]
        if len(bad):
            problems.append(f"{name} key crosses splits: {len(bad)} keys, e.g. {bad.index[0]!r}")
    if (in_domain.split.isna()).any():
        problems.append(f"{int(in_domain.split.isna().sum())} samples without a split")
    conflict_in_test = in_domain[(in_domain.split == "test") & ~in_domain.test_ok]
    if len(conflict_in_test):
        problems.append(f"{len(conflict_in_test)} test samples carry a flag that excludes them from test")
    src = in_domain.set_index("sample_id").source
    for fold, part in lodo.groupby("fold"):
        train_src = src.reindex(part[part.role == "train"].sample_id).unique()
        if fold in train_src:
            problems.append(f"LODO fold {fold}: held-out source in training")
        if set(src.reindex(part[part.role == "test"].sample_id).unique()) != {fold}:
            problems.append(f"LODO fold {fold}: test set is not exactly the held-out source")
    day = pd.to_datetime(manifest.utc, utc=True).dt.strftime("%Y-%m-%d")
    day_of = pd.Series(day.values, index=manifest.sample_id)
    cam_of = pd.Series(manifest.camera_id.values, index=manifest.sample_id)
    for variant, part in station.groupby("variant"):
        tr, te = part[part.role == "train"].sample_id, part[part.role == "test"].sample_id
        if set(cam_of.reindex(tr)) & set(cam_of.reindex(te)):
            problems.append(f"station {variant}: a camera appears in train and test")
        if variant == "disjoint_days" and set(day_of.reindex(tr)) & set(day_of.reindex(te)):
            problems.append("station disjoint_days: a date appears in train and test")
    return problems


def sha256_of_ids(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def split_hashes(in_domain: pd.DataFrame, lodo: pd.DataFrame, station: pd.DataFrame) -> dict:
    out = {}
    for (source, split), part in in_domain.groupby(["source", "split"]):
        out[f"in_domain/{source}/{split}"] = {"n": len(part), "sha256": sha256_of_ids(part.sample_id)}
    for (fold, role), part in lodo.groupby(["fold", "role"]):
        out[f"lodo/{fold}/{role}"] = {"n": len(part), "sha256": sha256_of_ids(part.sample_id)}
    for (variant, role), part in station.groupby(["variant", "role"]):
        out[f"held_out_station/{variant}/{role}"] = {"n": len(part), "sha256": sha256_of_ids(part.sample_id)}
    return out


# ------------------------------------------------------------------------------------------------- report


def summary_tables(in_domain: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = in_domain.pivot_table(index="source", columns="split", values="sample_id", aggfunc="count", fill_value=0)
    counts["units"] = in_domain.groupby("source").unit.nunique()
    counts["images"] = in_domain.groupby("source").size()
    strata = (in_domain.groupby(["source", "split", "stratum"]).size().rename("n").reset_index())
    strata["share"] = strata.n / strata.groupby(["source", "split"]).n.transform("sum")
    return counts.reset_index(), strata


def report_markdown(counts: pd.DataFrame, strata: pd.DataFrame, lodo: pd.DataFrame, station: pd.DataFrame, problems: list[str],
                    hashes: dict, notes: list[str]) -> str:
    lines = ["# Splits report (P038)", "",
             "Generated by `scripts/make_splits.py` from `configs/splits.yaml` (design: `docs/splits.md`). "
             "Files under `data/splits/`; hashes of the sorted sample ids per file below.", "",
             "## In-domain blocked splits", "", "| Source | Images | Units | Train | Val | Test | Test share |",
             "|---|---|---|---|---|---|---|"]
    for _, r in counts.iterrows():
        test = int(r.get("test", 0))
        lines.append(f"| {r.source} | {int(r.images):,} | {int(r.units):,} | {int(r.get('train', 0)):,} | "
                     f"{int(r.get('val', 0)):,} | {test:,} | {test / r.images:.1%} |")
    lines += ["", "### Stratum shares per split (largest strata)", "", "| Source | Stratum | Train | Val | Test |",
              "|---|---|---|---|---|"]
    wide = strata.pivot_table(index=["source", "stratum"], columns="split", values="share", fill_value=0.0).reset_index()
    for _source, part in wide.groupby("source"):
        top = part.assign(m=part[["train", "val", "test"]].max(axis=1)).sort_values("m", ascending=False).head(6)
        for _, r in top.iterrows():
            lines.append(f"| {r.source} | {r.stratum} | {r.train:.1%} | {r.val:.1%} | {r.test:.1%} |")
    lines += ["", "## Leave-one-dataset-out folds", "", "| Fold (held out) | Train images | Test images |", "|---|---|---|"]
    for fold, part in lodo.groupby("fold"):
        lines.append(f"| {fold} | {int((part.role == 'train').sum()):,} | {int((part.role == 'test').sum()):,} |")
    lines += ["", "## Held-out station (Eye2Sky)", "", "| Variant | Train frames | Test frames |", "|---|---|---|"]
    for variant, part in station.groupby("variant"):
        lines.append(f"| {variant} | {int((part.role == 'train').sum()):,} | {int((part.role == 'test').sum()):,} |")
    ok = ["- All guarantees hold (no group or block crosses a split; no conflict-merged image in test; held-out sources "
          "absent from LODO training; cameras and, in the disjoint variant, dates separated in the station protocol)."]
    lines += ["", "## Verification", ""] + ([f"- FAILED: {p}" for p in problems] or ok)
    lines += ["", "## Hashes (SHA-256 of sorted sample ids)", "", "| File | Images | SHA-256 |", "|---|---|---|"]
    for name, h in hashes.items():
        lines.append(f"| {name} | {h['n']:,} | `{h['sha256'][:16]}…` |")
    lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"
