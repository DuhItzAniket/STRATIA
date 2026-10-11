"""Multi-dataset, multi-task batch sampling (P069).

A STRATIA training set is a union of datasets with different sizes and label types. Three problems and the policy
for each:

* **Big datasets drown small ones.** Datasets are drawn with probability p_d ∝ n_d^tau (tau = 1 is proportional,
  0 is uniform; the default 0.5 is the usual square-root balance), and inside a dataset rows are drawn uniformly
  with replacement. The expected share of every dataset per epoch is therefore known in advance and is what the
  tests check (`expected_shares`).
* **Per-task batches.** Each batch belongs to one task group (`global`, `dense`, `cbh`) so a batch holds tensors of
  one shape (dense batches carry masks and a 32 x 32 token grid, global ones a cls vector, CBH ones an 8 x 8 grid)
  and the loss code is simple. Groups are drawn with configurable probabilities; a dataset belongs to the groups
  its tasks name.
* **Missing labels.** Every row carries a `label_mask` (one flag per head) built from which targets it has; the
  losses (P064) are masked per sample, so a batch can mix a CCSN row (genus only) with a Montenegro row (genus
  shares + étage + oktas) without any special casing.

`TaskBatchSampler` is a torch `Sampler` of index batches over a `MultiDataset`-style table (one row per sample with
`dataset`, `group` and the label flags); it is deterministic for a seed and an epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HEADS = ("genus", "etage", "oktas", "sky_parse", "layer", "cbh")
GROUPS = ("global", "dense", "cbh")
GROUP_OF_HEAD = {"genus": "global", "etage": "global", "oktas": "global", "sky_parse": "dense", "layer": "dense",
                 "cbh": "cbh"}


@dataclass
class SamplerConfig:
    tau: float = 0.5                                   # dataset temperature: p_d ∝ n_d^tau
    group_probs: dict = field(default_factory=lambda: {"global": 0.5, "dense": 0.3, "cbh": 0.2})
    batch_sizes: dict = field(default_factory=lambda: {"global": 64, "dense": 16, "cbh": 64})
    batches_per_epoch: int = 500
    min_share: float = 0.0                             # floor on a dataset's probability inside its group


def label_mask(row: pd.Series | dict) -> np.ndarray:
    """One bool per head from a row's label columns: has_genus, has_etage, has_oktas, has_sky_parse, has_layer, has_cbh."""
    return np.array([bool(row.get(f"has_{h}", False)) for h in HEADS], dtype=bool)


def dataset_probs(counts: pd.Series, tau: float, min_share: float = 0.0) -> pd.Series:
    """p_d ∝ n_d^tau over the datasets in `counts`, floored at min_share and renormalised."""
    w = counts.astype(float) ** tau
    p = w / w.sum()
    if min_share > 0:
        for _ in range(len(p)):                      # floor the small ones, rescale the rest, repeat until stable
            low = p < min_share - 1e-12
            if not low.any() or low.all():
                break
            p[low] = min_share
            rest = ~low
            p[rest] = p[rest] * (1.0 - min_share * int(low.sum())) / p[rest].sum()
    return p


def expected_shares(table: pd.DataFrame, cfg: SamplerConfig) -> pd.DataFrame:
    """Expected share of samples per (group, dataset) per epoch under the config (for the stats check)."""
    rows = []
    total = sum(cfg.group_probs[g] * cfg.batch_sizes[g] for g in cfg.group_probs)
    for g, pg in cfg.group_probs.items():
        part = table[table.group_mask.map(lambda m, g=g: g in m)]
        if not len(part):
            continue
        p = dataset_probs(part.dataset.value_counts(), cfg.tau, cfg.min_share)
        for d, pd_ in p.items():
            rows.append({"group": g, "dataset": d, "share": pg * cfg.batch_sizes[g] * pd_ / total,
                         "p_in_group": pd_, "n": int((part.dataset == d).sum())})
    return pd.DataFrame(rows)


class TaskBatchSampler:
    """Yields (group, indices) batches. `table` needs columns `dataset` and `group_mask` (a set of groups the row
    serves, from its tasks); indices are positions in `table`."""

    def __init__(self, table: pd.DataFrame, cfg: SamplerConfig, seed: int = 0):
        self.table = table.reset_index(drop=True)
        self.cfg = cfg
        self.seed = seed
        self.epoch = 0
        self.groups = [g for g in cfg.group_probs if any(g in m for m in self.table.group_mask)]
        self.group_p = np.array([cfg.group_probs[g] for g in self.groups], dtype=float)
        self.group_p /= self.group_p.sum()
        self.pools: dict[str, dict[str, np.ndarray]] = {}
        self.pool_p: dict[str, tuple[list[str], np.ndarray]] = {}
        for g in self.groups:
            idx = self.table.index[self.table.group_mask.map(lambda m, g=g: g in m)]
            part = self.table.loc[idx]
            probs = dataset_probs(part.dataset.value_counts(), cfg.tau, cfg.min_share)
            self.pools[g] = {d: part.index[part.dataset == d].to_numpy() for d in probs.index}
            self.pool_p[g] = (list(probs.index), probs.to_numpy())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.cfg.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        for _ in range(self.cfg.batches_per_epoch):
            g = self.groups[rng.choice(len(self.groups), p=self.group_p)]
            names, p = self.pool_p[g]
            bs = self.cfg.batch_sizes[g]
            picks = rng.choice(len(names), size=bs, p=p)
            idx = np.concatenate([rng.choice(self.pools[g][names[k]], size=int(c))
                                  for k, c in zip(*np.unique(picks, return_counts=True), strict=True)])
            rng.shuffle(idx)
            yield g, idx.tolist()


def sampling_stats(sampler: TaskBatchSampler, epochs: int = 1) -> pd.DataFrame:
    """Observed share per (group, dataset) over `epochs` epochs, next to the expected share."""
    counts: dict[tuple[str, str], int] = {}
    total = 0
    for e in range(epochs):
        sampler.set_epoch(e)
        for g, idx in sampler:
            for d in sampler.table.dataset.to_numpy()[idx]:
                counts[(g, d)] = counts.get((g, d), 0) + 1
            total += len(idx)
    obs = pd.DataFrame([{"group": g, "dataset": d, "observed": c / total} for (g, d), c in counts.items()])
    exp = expected_shares(sampler.table, sampler.cfg)
    return exp.merge(obs, on=["group", "dataset"], how="left").fillna({"observed": 0.0})


def group_mask_from_tasks(tasks) -> set[str]:
    """Dataset tasks (configs/datasets.yaml) -> the sampler groups the dataset can serve."""
    out = set()
    for t in tasks:
        if t in ("genus", "genus_soft", "sky_type", "oktas", "cbh_class", "wmo_codes", "etage"):
            out.add("global")
        if t in ("sky_parse", "layer"):
            out.add("dense")
        if t in ("cbh",):
            out.add("cbh")
    return out
