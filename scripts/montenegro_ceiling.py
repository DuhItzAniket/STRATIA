"""Montenegro soft labels and human ceiling (P035).

    python scripts/montenegro_ceiling.py

Reads the raw annotations (P016 loader), derives étage / genus / oktas / height-band labels per rater through the
ontology, writes data/montenegro_targets.parquet (per-image soft targets), data/montenegro_ceiling.parquet and
docs/data/montenegro_ceiling.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.montenegro import load_annotations, load_items  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.labels import agreement as ag  # noqa: E402
from stratia.labels.ontology import load_ontology  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    root = Path(load_paths()["data_root"]) / "Montenegro"
    ont = load_ontology()
    ann, items = load_annotations(root), load_items(root)
    derived = ag.derive(ann, ont)
    targets = ag.soft_targets(derived).merge(items[["item_id", "image_file"]], on="item_id", how="left")
    targets["image_file"] = "Montenegro/" + targets.image_file
    targets.to_parquet("data/montenegro_targets.parquet", index=False)
    ceiling = ag.ceiling_table(ann, derived)
    ceiling.to_parquet("data/montenegro_ceiling.parquet", index=False)
    loo_e = ag.leave_one_out(ag.pivot(derived, "etage_set"), "nominal")
    n_num = ann.assign(Nn=ann.N.map(lambda v: int(v) if isinstance(v, str) and v.isdigit() and int(v) <= 8 else float("nan")))
    loo_n = ag.leave_one_out(ag.pivot(n_num, "Nn"), "numeric", tolerance=1.0)
    per_rater = loo_e.rename(columns={"units": "units_etage", "agreement": "etage"}).merge(
        loo_n.rename(columns={"units": "units_n", "agreement": "n"}), on="rater", how="outer").fillna(0)
    print(ceiling.to_string(index=False), flush=True)
    print(per_rater.to_string(index=False), flush=True)
    c = ceiling.set_index("quantity")
    e_loo, e_alpha = c.loc["étage set", "leave_one_out"], c.loc["étage set", "alpha"]
    n_loo, n_alpha = c.loc["raw N (oktas 0-8)", "leave_one_out"], c.loc["raw N (oktas 0-8)", "alpha"]
    decision = [
        f"**Human ceiling for criterion C2:** a rater agrees with the majority of the other raters on the étage set in "
        f"{e_loo:.1%} of images (alpha {e_alpha:.2f}) and on total cloud cover within ±1 okta in {n_loo:.1%} (interval alpha "
        f"{n_alpha:.2f}). A model's agreement with the rater majority is compared with these numbers; 90 % of them is the "
        "C2 bar.",
        f"**Genus from the codes is the weakest signal** (exact genus set alpha {c.loc['genus set', 'alpha']:.2f}); per-genus "
        "presence is used as a soft target (share of raters), never as a hard label.",
        f"**Height code h** (ordinal alpha {c.loc['raw h (height code 0-9)', 'alpha']:.2f}; ±1 band leave-one-out "
        f"{c.loc['raw h (height code 0-9), ±1 band', 'leave_one_out']:.1%}) stays a local soft evaluation target; the ceilometer "
        "is the CBH reference (P031, P036).",
        "Targets per image: `data/montenegro_targets.parquet` (étage and genus shares, oktas and height-band distributions, "
        "obscured share, cloud share); training uses them as distributions (P030: no resampling of soft labels).",
    ]
    report = ag.report_markdown(ceiling, per_rater, len(targets), len(ann), decision)
    Path("docs/data/montenegro_ceiling.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: data/montenegro_targets.parquet, data/montenegro_ceiling.parquet, "
          "docs/data/montenegro_ceiling.md", flush=True)


if __name__ == "__main__":
    main()
