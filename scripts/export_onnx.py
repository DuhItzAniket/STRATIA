"""Export the deployed STRATIA model to ONNX with its model card (P096; FP32, dynamic batch, opset 17).

    python scripts/export_onnx.py --out models/stratia-1.0.0-demo --version 1.0.0-demo \
        [--frames DATA/B0268/BLR01/2026-10-18 --n-frames 4]
    python scripts/export_onnx.py --dummy --out tests/fixtures/stratia_dummy

The trained export composes the Gate G4 heads (P073 global `indomain_e2s`, P074 dense, P075 CBH base) with the
frozen DINOv3 ViT-S/16 backbone and the P089 calibration (see stratia/export/onnx_model.py). It writes
`<out>/stratia.onnx`, `<out>/model_card.json`, `<out>/export_report.json` and `docs/data/onnx_export.md`, and
registers a P096 run. `--frames` adds a parity check and an output preview on a few real B0268 frames (sky-logger
sidecars give the Sun; the nominal B0268 camera and a zenith pose give the ray map). The dummy export writes a
small untrained graph with the same interface for CloudScope's contract test (P065).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stratia import contract as C  # noqa: E402
from stratia.export import onnx_model as X  # noqa: E402

DEFAULT_RUNS = {
    "global_run": "20261011T125754Z_global_indomain_e2s_s0_27d0d07c22",
    "cbh_run": "20261011T101553Z_cbh_head_s0_[P075_base]_e286609cd2",
    "dense_run": "20261011T150404Z_dense_sky_layer_s0_dcd14f4833",
}
CALIBRATION = ROOT / "configs" / "calibration.json"


# ----------------------------------------------------------------------------------------------- real frames
def b0268_geometry():
    """The nominal B0268 camera (configs/cameras.yaml) and a zenith-pointing pose."""
    from stratia.geometry.cameras import camera_for
    from stratia.geometry.pose import Pose

    camera = camera_for("b0268", 4656, 3496)
    return camera, Pose(R=np.eye(3), source="declared", calib_id="nominal")


def load_frames(folder: Path, n: int, size: int, mean, std) -> tuple[dict[str, np.ndarray], list[dict]]:
    """n evenly spaced sky-logger frames of a day folder as contract inputs, plus per-frame facts."""
    import cv2

    from stratia.geometry.raymap import meta_vector, ray_map

    jpgs = sorted(folder.glob("*.jpg"))
    if not jpgs:
        raise SystemExit(f"no .jpg frames in {folder}")
    pick = [jpgs[i] for i in np.linspace(0, len(jpgs) - 1, min(n, len(jpgs))).astype(int)]
    camera, pose = b0268_geometry()
    images, rms, metas, facts = [], [], [], []
    for p in pick:
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
        images.append(x.transpose(2, 0, 1))
        side = p.with_suffix(".json")
        sun = json.loads(side.read_text(encoding="utf-8"))["sun"] if side.exists() else {}
        az, el = sun.get("azimuth_deg"), sun.get("apparent_elevation_deg", sun.get("elevation_deg"))
        zen = None if el is None else 90.0 - float(el)
        rms.append(ray_map(camera, pose, az, zen, out_size=size, patch=C.PATCH_SIZE) if w == camera.width else
                   np.zeros((4, size // C.PATCH_SIZE, size // C.PATCH_SIZE), np.float32))
        metas.append(meta_vector(zen))
        facts.append({"file": p.name, "source_size": [w, h], "sun_azimuth_deg": az, "sun_zenith_deg": zen})
    inputs = {"image": np.stack(images), "ray_map": np.stack(rms).astype(np.float32), "meta": np.stack(metas).astype(np.float32)}
    return inputs, facts


def pooling_error(size: int = C.DEFAULT_INPUT_SIZE) -> dict:
    """Angle between the in-graph pooled ray map and a ray map computed directly at the 8 x 8 patch (B0268 nominal)."""
    from stratia.geometry.raymap import ray_map

    camera, pose = b0268_geometry()
    az, zen = 135.0, 45.0
    fine = ray_map(camera, pose, az, zen, out_size=size, patch=C.PATCH_SIZE)
    coarse = ray_map(camera, pose, az, zen, out_size=size, patch=C.PATCH_SIZE * X.GRID_POOL)
    pooled = X.pool_ray_map(torch.from_numpy(fine)[None])[0].numpy()
    both = (coarse[3] > 0.5) & (pooled[3] > 0.5)
    dots = np.clip((coarse[:3] * pooled[:3]).sum(0)[both], -1, 1)
    deg = np.degrees(np.arccos(dots))
    return {"patches_valid_direct": int((coarse[3] > 0.5).sum()), "patches_valid_pooled": int((pooled[3] > 0.5).sum()),
            "valid_flag_disagreements": int(((coarse[3] > 0.5) != (pooled[3] > 0.5)).sum()),
            "angle_deg_max": float(deg.max()) if deg.size else 0.0, "angle_deg_mean": float(deg.mean()) if deg.size else 0.0}


def preview(outputs: dict[str, np.ndarray], facts: list[dict], threshold: float) -> list[dict]:
    """Human-readable outputs per real frame (a preview, not an evaluation: P083 does that with labels)."""
    from stratia.model import cbh_bins as B

    rows = []
    centres = B.centres_m()
    for i, f in enumerate(facts):
        genus = 1 / (1 + np.exp(-outputs["genus_logits"][i]))
        top = np.argsort(-genus)[:3]
        cbh = outputs["cbh_probs"][i]
        p_cloud = float(1 - cbh[B.K_BINS])
        median_bin = int(np.searchsorted(np.cumsum(cbh[: B.K_BINS]) / max(p_cloud, 1e-6), 0.5)) if p_cloud > 0.5 else None
        sky = outputs["sky_parse_logits"][i].argmax(0)
        rows.append({"file": f["file"], "ood_score": round(float(outputs["ood_score"][i, 0]), 3),
                     "uncertain": bool(outputs["ood_score"][i, 0] > threshold),
                     "oktas_argmax": int(outputs["oktas_probs"][i].argmax()),
                     "etage_probs": [round(float(x), 2) for x in 1 / (1 + np.exp(-outputs["etage_logits"][i]))],
                     "genus_top3": [(C.GENUS_CLASSES[k], round(float(genus[k]), 2)) for k in top],
                     "p_cloud_overhead": round(p_cloud, 2),
                     "cbh_median_m": None if median_bin is None else int(centres[min(median_bin, B.K_BINS - 1)]),
                     "sky_parse_fraction": {c: round(float((sky == k).mean()), 3) for k, c in enumerate(C.SKY_PARSE_CLASSES)}})
    return rows


# ----------------------------------------------------------------------------------------------- report
def write_report(path: Path, card: dict, info: dict) -> None:
    lines = [
        "# ONNX export (P096)", "",
        f"`{info['onnx_file']}`: {info['onnx_mb']:.1f} MB, SHA-256 `{info['onnx_sha256']}`, opset {X.OPSET}, dynamic batch, "
        f"FP32. Model card `{info['card_file']}` (stratia-contract v{card['contract_version']}, model "
        f"`{card['model_name']}` {card['model_version']}, git `{card['git_sha'][:10]}`), also embedded in the ONNX "
        f"metadata under `{X.METADATA_KEY}`.", "",
        "Graph: frozen DINOv3 ViT-S/16 (`" + card["backbone"] + "`) at 512 px; dense heads on the 32 x 32 token grid; "
        "global and cloud-base heads with their own geometry encoders on the grid average-pooled to 8 x 8 (as trained); "
        "the contract's 32 x 32 ray map is pooled to 8 x 8 in the graph; P089 temperatures and OOD statistics are folded "
        "in (`calibration.folded_into_graph = true`), so hosts apply nothing but the threshold.", "",
        "## Provenance", "",
        "| Part | Run | Validation summary |", "|---|---|---|",
    ]
    for k in ("global_run", "cbh_run", "dense_run"):
        s = card["head_summaries"].get(k.replace("_run", "_summary"), {})
        summary = ", ".join(f"{a} {b:.3f}" if isinstance(b, float) and not float(b).is_integer() else f"{a} {int(b)}"
                            for a, b in s.items())
        lines.append(f"| {k.replace('_run', '')} | `{card['trained_from'][k]}` | {summary} |")
    t = card["calibration"]["temperatures"]
    lines += ["", f"Calibration (P089): temperatures genus {t['genus']:.4f}, étage {t['etage']:.1f}, oktas {t['oktas']:.1f}, "
              f"CBH {t['cbh']:.4f}; OOD threshold {card['ood_threshold']:.4f}; CBH conformal radii "
              + ", ".join(f"{k} % {v:.3f}" for k, v in card["calibration"]["cbh_conformal_radius_cdf"].items()) + " (CDF units).",
              "", "## Parity (PyTorch FP32 vs ONNX Runtime CPU)", "",
              "| Output | Shape (batch of 4) | Random inputs: max abs | max rel | Real frames: max abs | max rel |",
              "|---|---|---|---|---|---|"]
    real = info.get("parity_real", {})
    for k, v in info["parity_random"].items():
        r = real.get(k, {})
        lines.append(f"| `{k}` | {tuple(v['shape'])} | {v['max_abs']:.2e} | {v['max_rel']:.2e} | "
                     + (f"{r['max_abs']:.2e} | {r['max_rel']:.2e} |" if r else "– | – |"))
    lines += ["", f"Contract check (names, shapes, finite values, probability sums): "
              f"{'passed' if not info['contract_problems'] else '; '.join(info['contract_problems'])}. "
              f"Half of every parity batch has an all-zero ray map and meta (unknown geometry).", "",
              f"ONNX Runtime CPU latency, batch 1, median of {info['latency_runs']}: {info['latency_ms']:.0f} ms "
              f"({info['cpu']}).", "",
              "## Ray-map pooling", "",
              "The global and CBH heads were trained on ray maps computed directly at the 8 x 8 patch; the graph pools the "
              "contract's 32 x 32 map instead (mean of the valid unit directions, re-normalised; valid when at least half of "
              "the sub-patches are). On the nominal B0268 camera (zenith pose, Sun at 45° zenith): "
              f"{info['pooling']['patches_valid_direct']} / {info['pooling']['patches_valid_pooled']} valid patches "
              f"(direct / pooled), {info['pooling']['valid_flag_disagreements']} flag disagreements, direction error "
              f"mean {info['pooling']['angle_deg_mean']:.3f}°, max {info['pooling']['angle_deg_max']:.3f}° "
              "(the CBH prior's sigma is 20°).", ""]
    if info.get("preview"):
        lines += ["## Preview on B0268 frames (not an evaluation)", "",
                  f"Frames from `{info['frames_folder']}` through the exported graph with the nominal B0268 camera, a zenith "
                  "pose and the sidecar's Sun. The model has never seen this camera; P083 (zero-shot, with labels) is the "
                  "evaluation. Shown only to confirm the graph produces sensible, finite outputs on real frames.", "",
                  "| Frame | OOD score | Uncertain | Oktas | Étage p(low, mid, high) | Genus top-3 | p(cloud overhead) | "
                  "CBH median | Sky-parse fractions |", "|---|---|---|---|---|---|---|---|---|"]
        for r in info["preview"]:
            lines.append(f"| {r['file']} | {r['ood_score']} | {'yes' if r['uncertain'] else 'no'} | {r['oktas_argmax']} | "
                         f"{r['etage_probs']} | {r['genus_top3']} | {r['p_cloud_overhead']} | "
                         f"{'–' if r['cbh_median_m'] is None else str(r['cbh_median_m']) + ' m'} | {r['sky_parse_fraction']} |")
        lines.append("")
    lines += ["## Licences", ""] + [f"- {s}" for s in card["licences"]] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--version", default="1.0.0-demo")
    ap.add_argument("--dummy", action="store_true", help="export the small untrained fixture instead of the trained model")
    ap.add_argument("--global-run", default=DEFAULT_RUNS["global_run"])
    ap.add_argument("--cbh-run", default=DEFAULT_RUNS["cbh_run"])
    ap.add_argument("--dense-run", default=DEFAULT_RUNS["dense_run"])
    ap.add_argument("--calibration", type=Path, default=CALIBRATION)
    ap.add_argument("--frames", type=Path, default=None, help="a sky-logger day folder for a real-frame parity check")
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--no-report", action="store_true", help="do not write docs/data/onnx_export.md nor register a run")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    size = C.DEFAULT_INPUT_SIZE
    t0 = time.time()
    if a.dummy:
        model, facts = X.dummy_model(), {"patch_size": C.PATCH_SIZE, "mean": [0.0] * 3, "std": [1.0] * 3}
        # the dummy folds its own (made-up) temperatures into the graph: the card says so, like the real one
        temperatures = {k: float(getattr(model, f"t_{k}")) for k in ("genus", "etage", "oktas", "cbh")}
        trained = {"calibration": {"temperatures": temperatures, "cbh_conformal_radius": {"80": 0.3, "95": 0.45},
                                   "ood_threshold": 1.0, "fitted_on": "untrained dummy: made-up values"}}
        name, threshold = "stratia_dummy.onnx", 1.0
    else:
        backbone, facts = X.hf_backbone("dinov3_vits16")
        parts = X.TrainedParts(a.global_run, a.cbh_run, a.dense_run, a.calibration)
        model, trained = X.load_trained(parts, backbone, d_model=facts["dim"])
        name, threshold = "stratia.onnx", float(trained["calibration"]["ood_threshold"])
    training_hash = None if a.dummy else X.manifest_hash()
    card = X.model_card(a.version, facts, trained, size, ood_threshold=threshold, training_hash=training_hash, dummy=a.dummy)
    onnx_path = X.export_onnx(model, a.out / name, size, metadata={X.METADATA_KEY: json.dumps(card)})
    card_path = X.write_model_card(card, a.out / "model_card.json")
    session = X.onnx_session(onnx_path)
    ex = X.example_inputs(4, size, seed=1)
    outputs = X.run_onnx(session, ex)
    problems = X.check_contract(outputs, 4, size)
    info = {"onnx_file": (onnx_path.relative_to(ROOT) if onnx_path.is_relative_to(ROOT) else onnx_path).as_posix(),
            "card_file": (card_path.relative_to(ROOT) if card_path.is_relative_to(ROOT) else card_path).as_posix(),
            "onnx_mb": onnx_path.stat().st_size / 1e6, "onnx_sha256": X.file_sha256(onnx_path),
            "parity_random": X.parity(model, session, ex), "contract_problems": problems,
            "pooling": pooling_error(size), "export_seconds": round(time.time() - t0, 1)}
    one = {k: v[:1] for k, v in ex.items()}
    times = []
    for _ in range(5):
        t = time.perf_counter()
        X.run_onnx(session, one)
        times.append((time.perf_counter() - t) * 1000)
    info.update(latency_ms=float(np.median(times)), latency_runs=len(times), cpu=platform.processor())
    if a.frames is not None:
        inputs, frame_facts = load_frames(a.frames, a.n_frames, size, facts["mean"], facts["std"])
        info["frames_folder"] = a.frames.as_posix()
        info["parity_real"] = X.parity(model, session, inputs)
        info["preview"] = preview(X.run_onnx(session, inputs), frame_facts, threshold)
        info["contract_problems"] += X.check_contract(X.run_onnx(session, inputs), len(frame_facts), size)
    (a.out / "export_report.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    worst = max(v["max_abs"] for v in info["parity_random"].values())
    print(f"{onnx_path} ({info['onnx_mb']:.1f} MB) parity max abs {worst:.2e}; contract "
          f"{'OK' if not info['contract_problems'] else info['contract_problems']}; latency {info['latency_ms']:.0f} ms")
    if a.dummy or a.no_report:
        return 0 if not info["contract_problems"] else 1
    write_report(ROOT / "docs" / "data" / "onnx_export.md", card, info)
    from omegaconf import OmegaConf

    from stratia.utils.run import Run

    cfg = OmegaConf.create({"version": a.version, "global_run": a.global_run, "cbh_run": a.cbh_run, "dense_run": a.dense_run,
                            "calibration": str(a.calibration), "opset": X.OPSET, "size": size})
    with Run(cfg, "onnx_export", "P096", data_hash=(training_hash or "")[:16]) as run:
        run.finish("done" if not info["contract_problems"] else "failed: contract", parity_max_abs=worst,
                   onnx_mb=info["onnx_mb"], latency_ms=info["latency_ms"],
                   pooling_angle_deg_max=info["pooling"]["angle_deg_max"])
        (run.dir / "export_report.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return 0 if not info["contract_problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
