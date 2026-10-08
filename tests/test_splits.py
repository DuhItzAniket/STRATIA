from collections import Counter

import numpy as np
import pandas as pd

from stratia.data import splits as sp

CONFIG = {
    "version": 1, "seed": 0, "fractions": {"train": 0.7, "val": 0.1, "test": 0.2},
    "sources": {"swimseg": "swim", "swinyseg": "swim"},
    "blocks": {"eye2sky": {"kind": "day", "shared_across_cameras": True}, "montenegro": {"kind": "days", "length_days": 7},
               "default": {"kind": "group"}},
    "stratify": {"ccsn": "source_label"}, "exclude_from_test": {"flags": ["label_conflict"]},
    "lodo": {"folds": ["ccsn", "swim"]},
    "held_out_station": {"dataset": "eye2sky", "train_camera": "eye2sky-AURIC", "test_camera": "eye2sky-BARSE",
                         "variants": ["same_days", "disjoint_days"]},
}


def _manifest() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for k in range(300):                                                   # CCSN: 3 classes
        rows.append({"dataset": "ccsn", "camera_id": "ccsn", "utc": pd.NaT, "source_label": "ABC"[k % 3]})
    rows += [{"dataset": "swimseg", "camera_id": "wsi", "utc": pd.NaT, "source_label": None}] * 100
    rows += [{"dataset": "swinyseg", "camera_id": "wsi", "utc": pd.NaT, "source_label": None}] * 100
    for day in range(10):                                                  # Eye2Sky: two stations, 10 days, 20 frames each
        for cam in ("AURIC", "BARSE"):
            for f in range(20):
                rows.append({"dataset": "eye2sky", "camera_id": f"eye2sky-{cam}", "source_label": None,
                             "utc": pd.Timestamp("2022-04-01", tz="UTC") + pd.Timedelta(days=day, minutes=f)})
    for k in range(140):                                                   # Montenegro: 70 days, 2 frames a day
        rows.append({"dataset": "montenegro", "camera_id": "mon", "source_label": "Low clouds",
                     "utc": pd.Timestamp("2025-10-03", tz="UTC") + pd.Timedelta(days=k // 2, hours=k % 2)})
    m = pd.DataFrame(rows)
    m["sample_id"] = m.dataset + ":" + m.index.astype(str)
    m["image_file"] = m.sample_id
    rng.shuffle  # noqa: B018  (deterministic data; the generator's own seed is what is under test)
    return m


def _table(m):
    source = sp.source_of(m.dataset, CONFIG)
    # groups: CCSN pairs (0,1), (2,3); a SWIM copy pair across swimseg/swinyseg
    g = pd.Series([None] * len(m), dtype=object)
    g[0] = g[1] = "near:a"
    g[2] = g[3] = "near:b"
    g[300] = g[400] = "near:swim"
    blocks = sp.block_keys(m, CONFIG)
    units = sp.unit_ids(len(m), [g, blocks])
    test_ok = np.ones(len(m), bool)
    test_ok[0] = False                                                      # a conflict merge: never in test
    stratum = m.source_label.fillna("all").astype(str)
    table = pd.DataFrame({"sample_id": m.sample_id, "dataset": m.dataset, "source": source, "unit": units, "stratum": stratum,
                          "test_ok": test_ok})
    return table, g, blocks


def test_units_join_groups_and_blocks():
    m = _manifest()
    table, g, blocks = _table(m)
    u = table.unit.values
    assert u[0] == u[1] and u[2] == u[3] and u[0] != u[2] and u[300] == u[400]
    e = m.dataset == "eye2sky"
    first_day = e & (pd.to_datetime(m.utc, utc=True).dt.day == 1)
    assert len(set(u[first_day.values])) == 1                                # both stations, all frames of day 1: one unit
    assert blocks[first_day.values].nunique() == 1 and blocks[first_day.values].iloc[0] == "eye2sky:day:2022-04-01"
    mon = (m.dataset == "montenegro").values
    assert len(set(u[mon])) == 10                                           # 70 days in 7-day blocks
    assert blocks[mon].iloc[0] == "montenegro:block:0" and blocks[mon].iloc[-1] == "montenegro:block:9"
    assert blocks[(m.dataset == "ccsn").values].isna().all()


def test_assignment_balances_strata_and_respects_exclusions():
    m = _manifest()
    table, g, blocks = _table(m)
    out = sp.in_domain_split(table, CONFIG)
    assert out.split.notna().all()
    ccsn = out[out.source == "ccsn"]
    shares = ccsn.groupby(["split", "stratum"]).size().unstack(fill_value=0)
    test_share = len(ccsn[ccsn.split == "test"]) / len(ccsn)
    assert 0.15 <= test_share <= 0.25
    for s in ("train", "test"):
        row = shares.loc[s]
        assert (row / row.sum()).max() < 0.4                                # the three classes stay close to 1/3 each
    assert out.loc[0, "split"] != "test" and out.loc[0, "split"] == out.loc[1, "split"]   # excluded; unit stays whole
    mon = out[out.source == "montenegro"]
    assert set(mon.groupby("unit").split.nunique()) == {1} and mon.split.value_counts().get("test", 0) in {2 * 14, 3 * 14}
    assert sp.verify(out, sp.lodo_folds(out, CONFIG), sp.held_out_station(m, out, CONFIG), m, [g], blocks) == []
    again = sp.in_domain_split(table, CONFIG)
    assert (again.split == out.split).all()                                 # deterministic


def test_lodo_station_verify_and_hashes():
    m = _manifest()
    table, g, blocks = _table(m)
    out = sp.in_domain_split(table, CONFIG)
    lodo = sp.lodo_folds(out, CONFIG)
    swim = lodo[lodo.fold == "swim"]
    assert set(out.set_index("sample_id").source.reindex(swim[swim.role == "test"].sample_id)) == {"swim"}
    assert "swim" not in set(out.set_index("sample_id").source.reindex(swim[swim.role == "train"].sample_id))
    station = sp.held_out_station(m, out, CONFIG)
    dj = station[station.variant == "disjoint_days"]
    day = pd.to_datetime(m.set_index("sample_id").utc, utc=True).dt.strftime("%Y-%m-%d")
    assert not (set(day.reindex(dj[dj.role == "train"].sample_id)) & set(day.reindex(dj[dj.role == "test"].sample_id)))
    same = station[station.variant == "same_days"]
    assert (same.role == "train").sum() == 200 and (same.role == "test").sum() == 200
    broken = out.copy()
    broken.loc[1, "split"] = "test" if broken.loc[1, "split"] != "test" else "train"   # splits a group
    problems = sp.verify(broken, lodo, station, m, [g], blocks)
    assert any("group key crosses" in p for p in problems)
    h = sp.split_hashes(out, lodo, station)
    assert "in_domain/ccsn/test" in h and h["lodo/swim/test"]["n"] == 200 and len(h["lodo/swim/test"]["sha256"]) == 64
    assert sp.sha256_of_ids(["b", "a"]) == sp.sha256_of_ids(["a", "b"])
    counts, strata = sp.summary_tables(out)
    text = sp.report_markdown(counts, strata, lodo, station, [], h, ["note"])
    assert "| ccsn |" in text and "All guarantees hold" in text and "- note" in text
    assert isinstance(Counter(out.split), Counter)
