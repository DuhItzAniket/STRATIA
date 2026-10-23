"""B0268 zero-shot evaluation of the exported model against the P041 rater labels (P083).

    python scripts/eval_b0268.py [--model models/stratia-1.0.0-demo] [--draw k0_d0] [--roles val]

Runs the exported ONNX graph (the demo model: Gate G4 heads + P089 calibration folded in) on the labelled B0268
frames of the given few-shot draw and roles (default: the adaptation days of the zero-shot draw; the locked test
day is never touched here), with the nominal B0268 camera, a zenith pose and the manifest's Sun, and scores it
with `stratia.train.b0268_eval`: primarily against each rater of the P041 pair, averaged (the way the ceiling,
the raters' own agreement, is measured), and secondarily against the pair's majority and mean cover (cloud
fraction, glare, OOD). Writes data/b0268_zero_shot.parquet (per-frame predictions), data/b0268_zero_shot_scores.parquet
and docs/data/b0268_zero_shot.md, and registers a P083 run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from stratia import contract as C  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.geometry.cameras import camera_for  # noqa: E402
from stratia.geometry.pose import Pose  # noqa: E402
from stratia.geometry.raymap import meta_vector, ray_map  # noqa: E402
from stratia.model import cbh_bins as B  # noqa: E402
from stratia.train import b0268_eval as E  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

OUT_ROWS, OUT_SCORES, DOC = Path("data/b0268_zero_shot.parquet"), Path("data/b0268_zero_shot_scores.parquet"), \
    Path("docs/data/b0268_zero_shot.md")
PRIMARY_KEYS = ["n", "n_raters", "etage_set_agreement", "etage_set_agreement_A", "etage_set_agreement_B", "c2_etage_ratio",
                "oktas_within1", "oktas_within1_A", "oktas_within1_B", "c2_oktas_ratio", "oktas_mae", "oktas_bias",
                "genus_top1_in_set"]
SECONDARY_KEYS = ["n", "etage_set_agreement", "etage_f1", "oktas_within1", "genus_top1_in_union", "cloud_fraction_mae_vs_cover",
                  "cloud_fraction_corr_vs_cover", "cloud_fraction_on_le1_okta", "glare_fraction_strong_frames",
                  "glare_fraction_other_frames", "ood_mean", "ood_flagged_share"]


def frames_for(draw: str, roles: list[str]) -> pd.DataFrame:
    """Manifest rows of the B0268 frames with the given roles in the few-shot draw, with the frame id."""
    few = pd.read_parquet("data/splits/few_shot_target.parquet")
    few = few[(few.draw == draw) & few.role.isin(roles)]
    m = pd.read_parquet("data/manifest.parquet")
    m = m[m.sample_id.isin(few.sample_id)].merge(few[["sample_id", "day", "role"]], on="sample_id")
    m["item_id"] = m.image_file.str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0]
    return m.sort_values("utc").reset_index(drop=True)


def load_inputs(rows: pd.DataFrame, data_root: Path, card: dict) -> dict[str, np.ndarray]:
    import cv2

    size = int(card["input"]["height"])
    mean, std = np.asarray(card["input"]["mean"], np.float32), np.asarray(card["input"]["std"], np.float32)
    camera = camera_for("b0268", int(rows.width.iloc[0]), int(rows.height.iloc[0]))
    pose = Pose(R=np.eye(3), source="declared", calib_id="nominal")
    images, rms, metas = [], [], []
    for r in rows.itertuples(index=False):
        bgr = cv2.imdecode(np.fromfile(str(data_root / r.image_file), dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        images.append(((rgb.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1))
        rms.append(ray_map(camera, pose, float(r.sun_azimuth_deg), float(r.sun_zenith_deg), out_size=size, patch=C.PATCH_SIZE))
        metas.append(meta_vector(float(r.sun_zenith_deg)))
    return {"image": np.stack(images).astype(np.float32), "ray_map": np.stack(rms).astype(np.float32),
            "meta": np.stack(metas).astype(np.float32)}


def run_graph(onnx_path: Path, inputs: dict[str, np.ndarray], batch: int, providers: list[str]) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    names = [o.name for o in sess.get_outputs()]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in names}
    n = len(inputs["image"])
    for s in range(0, n, batch):
        feed = {k: v[s:s + batch] for k, v in inputs.items()}
        for k, v in zip(names, sess.run(None, feed), strict=True):
            parts[k].append(v)
    return {k: np.concatenate(v) for k, v in parts.items()}


def predictions(out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Contract outputs -> the arrays `b0268_eval` consumes, plus per-frame extras."""
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    sky = out["sky_parse_logits"].argmax(1)                         # [n, h, w]: invalid, sky, cloud, sun_glare
    valid = sky != 0
    denom = np.maximum(valid.sum((1, 2)), 1)
    cloud = (sky == 2).sum((1, 2)) / denom
    glare = (sky == 3).sum((1, 2)) / denom
    cbh = out["cbh_probs"]
    return {"etage": sig(out["etage_logits"]), "genus": sig(out["genus_logits"]), "oktas": out["oktas_probs"],
            "cloud_fraction": cloud, "glare_fraction": glare, "ood": out["ood_score"][:, 0],
            "cbh_median_m": B.median_height_m(cbh), "p_cloud_overhead": 1.0 - cbh[:, B.K_BINS]}


def ceiling_from(raters: str = "AB") -> dict[str, float] | None:
    c = pd.read_parquet("data/b0268_ceiling.parquet")
    c = c[c.raters == raters].set_index("quantity")
    if not len(c):
        return None
    return {"etage_set": float(c.loc["étage set", "leave_one_out"]), "oktas_within1": float(c.loc["oktas 0-8", "leave_one_out"])}


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _table(df: pd.DataFrame, keys: list[str]) -> list[str]:
    present = [k for k in keys if k in df.columns]
    lines = ["| Day | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    for _, r in df.iterrows():
        lines.append(f"| {r.day} | " + " | ".join(_fmt(r.get(k)) for k in present) + " |")
    return lines


def write_doc(primary: pd.DataFrame, secondary: pd.DataFrame, primary_all: pd.DataFrame | None, card: dict, draw: str,
              roles: list[str], ceiling: dict, n_test_excluded: int, run_id: str, ms_per_frame: float) -> None:
    onnx_file = card["onnx"]["file"] if isinstance(card.get("onnx"), dict) and "file" in card["onnx"] else "stratia.onnx"
    lines = ["# B0268 zero-shot evaluation (P083)", "",
             f"Model `{card['model_name']}` {card['model_version']} (`{onnx_file}`, git `{card['git_sha'][:10]}`; the Gate G4 "
             "heads with the P089 calibration folded in) on the labelled B0268 frames of draw "
             f"`{draw}`, roles {roles} (the locked test day, {n_test_excluded} frames, is not evaluated: P090). Nominal B0268 "
             f"camera, zenith pose, Sun from the manifest; ONNX Runtime on the CPU, {ms_per_frame:.0f} ms per frame. "
             f"Run `{run_id}`.", "",
             "## Against each rater of the P041 pair (mean of A and B)", "",
             "The model is scored against every rater separately and the scores are averaged, the way the ceiling is "
             f"measured (one rater against the other: étage set {ceiling['etage_set']:.3f}, oktas within ±1 "
             f"{ceiling['oktas_within1']:.3f}); `c2_*_ratio` = model / raters, as in criterion C2. No tie between the two "
             "raters is broken anywhere in this table.", ""]
    lines += _table(primary, PRIMARY_KEYS)
    lines += ["", "## Against the pair's majority and mean cover (cloud fraction, glare, OOD)", "",
              "Truth = étages named by at least one of the two raters, the lower okta on their ties, the union of their "
              "genera; cover = the raters' mean okta / 8. Use this table for the sky-parse, glare and OOD columns; the "
              "étage and okta columns here are the union / lower-okta view, the table above is the honest one.", ""]
    lines += _table(secondary, SECONDARY_KEYS)
    if primary_all is not None:
        lines += ["", "## Sensitivity: against each of the four raters (A, B, C, D), averaged", ""]
        lines += _table(primary_all, [k for k in PRIMARY_KEYS if not k.endswith(("_A", "_B"))])
    lines += ["", "Columns: `etage_set_agreement` = exact étage set (sigmoid > 0.5 per étage) against a rater's set; "
              "`oktas_within1` = argmax okta within ±1 of a rater's okta; `genus_top1_in_set` = the top genus is one that "
              "rater named (frames without a named genus skipped); `cloud_fraction_*` = sky-parse cloud share of the valid "
              "sky against the raters' mean cover; `cloud_fraction_on_le1_okta` = that share on frames the raters put at "
              "0-1 okta (the research design's no-collapse check asks for < 0.20 on clear frames); `glare_fraction_*` = "
              "sky-parse glare share on frames the raters called strong glare and on the others; `ood_flagged_share` = "
              "frames above the model card's OOD threshold."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    t0 = time.perf_counter()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/stratia-1.0.0-demo")
    ap.add_argument("--draw", default="k0_d0")
    ap.add_argument("--roles", nargs="+", default=["val"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--phase", default="P083")
    ap.add_argument("--name", default="b0268_zero_shot")
    ap.add_argument("--no-geometry", action="store_true",
                    help="zero the ray map and meta (unknown geometry, as the contract allows): an H4 check; outputs get "
                         "the suffix nogeo and the report is not written")
    ap.add_argument("--suffix", default=None,
                    help="write the parquet outputs under this suffix, skip the P083 report and name the run with it "
                         "(for checks of other graphs or roles; the default outputs are the P083 evaluation)")
    a = ap.parse_args()
    suffix = a.suffix or ("nogeo" if a.no_geometry else None)
    out_rows, out_scores = OUT_ROWS, OUT_SCORES
    if suffix:
        out_rows = OUT_ROWS.with_name(f"b0268_zero_shot_{suffix}.parquet")
        out_scores = OUT_SCORES.with_name(f"b0268_zero_shot_scores_{suffix}.parquet")
        a.name += f"_{suffix}"
    data_root = Path(load_paths()["data_root"])
    model_dir = Path(a.model)
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    rows = frames_for(a.draw, a.roles)
    few = pd.read_parquet("data/splits/few_shot_target.parquet")
    n_test = int(((few.draw == a.draw) & (few.role == "test")).sum())
    targets = pd.read_parquet("data/b0268_targets.parquet")
    truth = E.majority_targets(targets).set_index("item_id").reindex(rows.item_id).reset_index()
    assert truth.etage_majority.notna().all(), "a frame without rater targets"
    raters = E.rater_targets(pd.read_parquet("data/b0268_derived.parquet"))
    print(f"{len(rows)} frames, days {sorted(rows.day.unique())}, roles {a.roles}; {n_test} test frames excluded; raters "
          f"{sorted(raters)}", flush=True)

    inputs = load_inputs(rows, data_root, card)
    if a.no_geometry:
        inputs["ray_map"][:] = 0.0
        inputs["meta"][:] = 0.0
    t1 = time.perf_counter()
    out = run_graph(model_dir / "stratia.onnx", inputs, a.batch, ["CPUExecutionProvider"])
    ms = (time.perf_counter() - t1) * 1000 / len(rows)
    pred = predictions(out)
    ceiling = ceiling_from("AB")
    threshold = float(card["ood_threshold"]) if card.get("ood_threshold") is not None else None
    primary = E.per_day_against_raters(pred, rows.item_id, rows.day, raters, ceiling)
    secondary = E.per_day(pred, truth, ceiling, threshold)
    primary_all = None
    if Path("data/b0268_derived_all.parquet").exists():
        raters_all = E.rater_targets(pd.read_parquet("data/b0268_derived_all.parquet"))
        primary_all = E.per_day_against_raters(pred, rows.item_id, rows.day, raters_all, ceiling_from("ABCD"))

    cfg = OmegaConf.create({"model": str(model_dir), "model_version": card["model_version"], "draw": a.draw, "roles": a.roles,
                            "n_frames": len(rows), "provider": "CPUExecutionProvider", "batch": a.batch,
                            "geometry": "zeros (unknown)" if a.no_geometry else "nominal B0268 camera, zenith pose, manifest Sun",
                            "truth": "per-rater mean of A and B; secondary: majority / mean cover"})
    with Run(cfg, name=a.name, phase=a.phase) as run:
        summary = {k: v for k, v in primary.iloc[0].to_dict().items() if isinstance(v, float) and not np.isnan(v)}
        for k in ("cloud_fraction_on_le1_okta", "cloud_fraction_corr_vs_cover", "glare_fraction_strong_frames", "ood_mean",
                  "ood_flagged_share"):
            v = secondary.iloc[0].get(k)
            if isinstance(v, float) and not np.isnan(v):
                summary[k] = v
        run.finish(ms_per_frame=ms, **summary)
    per_frame = rows[["sample_id", "item_id", "day", "role", "utc", "sun_zenith_deg"]].copy()
    per_frame["etage_majority"] = [json.dumps([bool(x) for x in v]) for v in truth.etage_majority]
    per_frame["oktas_majority"] = truth.oktas_majority.to_numpy()
    per_frame["genus_union"] = [json.dumps(v) for v in truth.genus_union]
    for k in ("cloud_fraction", "glare_fraction", "ood", "cbh_median_m", "p_cloud_overhead"):
        per_frame[k] = pred[k]
    per_frame["etage_probs"] = [json.dumps([round(float(x), 4) for x in v]) for v in pred["etage"]]
    per_frame["oktas_argmax"] = pred["oktas"].argmax(1)
    per_frame["genus_top1"] = [C.GENUS_CLASSES[k] for k in pred["genus"][:, :E.N_GENERA].argmax(1)]
    per_frame["run_id"] = run.run_id
    per_frame.to_parquet(out_rows, index=False)
    scores = [primary.assign(truth="per-rater mean AB"), secondary.assign(truth="majority AB")]
    if primary_all is not None:
        scores.append(primary_all.assign(truth="per-rater mean ABCD"))
    pd.concat(scores, ignore_index=True).assign(run_id=run.run_id).to_parquet(out_scores, index=False)
    if not suffix:
        write_doc(primary, secondary, primary_all, card, a.draw, a.roles, ceiling, n_test, run.run_id, ms)
    print(primary.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    print(secondary[["day", "oktas_bias", "cloud_fraction_corr_vs_cover", "ood_flagged_share"]].to_string(index=False),
          flush=True)
    print(f"done in {time.perf_counter() - t0:.0f} s ({ms:.0f} ms per frame on the CPU): {out_rows}, {out_scores}"
          f"{', ' + str(DOC) if not suffix else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
