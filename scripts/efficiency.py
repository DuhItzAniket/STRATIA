"""P094 — Efficiency: parameters, FLOPs and latency of the deployed graph.

Parameters per component of the exported model, FLOPs of one 512 x 512 forward pass (torch's flop counter),
latency of the PyTorch model on the RTX 4050 (fp32 and autocast fp16) and of the ONNX graph on the laptop CPU
(ONNX Runtime), and the DirectML numbers from CloudScope's bench when its binary is present. The Raspberry Pi 5
column stays empty until a Pi exists. Outputs: data/efficiency.json, docs/data/efficiency.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stratia.export import onnx_model as X  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

OUT = ROOT / "data" / "efficiency.json"
DOC = ROOT / "docs" / "data" / "efficiency.md"
GLOBAL_RUN = "20261011T125754Z_global_indomain_e2s_s0_27d0d07c22"
CBH_RUN = "20261011T101553Z_cbh_head_s0_[P075_base]_e286609cd2"
DENSE_RUN = "20261011T150404Z_dense_sky_layer_s0_dcd14f4833"
AIBENCH = ROOT.parent / "CloudScope" / "build" / "windows-msvc" / "bin" / "Release" / "cloudscope-aibench.exe"


def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def timed(fn, warm: int = 5, runs: int = 20, sync=None) -> dict:
    for _ in range(warm):
        fn()
    if sync:
        sync()
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        fn()
        if sync:
            sync()
        t.append((time.perf_counter() - s) * 1000)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)), "runs": runs}


def main() -> int:
    t0 = time.perf_counter()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="P094")
    ap.add_argument("--onnx", default=str(ROOT / "models" / "stratia-1.0.1-demo" / "stratia.onnx"))
    ap.add_argument("--frame", default=None, help="a B0268 frame for the CloudScope bench")
    a = ap.parse_args()
    size = X.C.DEFAULT_INPUT_SIZE
    backbone, facts = X.hf_backbone("dinov3_vits16")
    parts = X.TrainedParts(GLOBAL_RUN, CBH_RUN, DENSE_RUN, ROOT / "configs" / "calibration.json")
    model, _ = X.load_trained(parts, backbone, d_model=facts["dim"])
    model.eval()
    params = {
        "backbone (DINOv3 ViT-S/16, frozen)": count(model.backbone),
        "SkyRay encoder, global": count(model.global_geo),
        "global heads": count(model.global_heads),
        "SkyRay encoder, CBH": count(model.cbh_geo),
        "CBH head": count(model.cbh_head),
        "dense heads": count(model.dense),
        "OOD score (buffers)": sum(b.numel() for b in model.ood.buffers()),
    }
    params["total trainable above the backbone"] = sum(v for k, v in params.items() if "backbone" not in k and "OOD" not in k)
    params["total"] = count(model)
    ex = {k: torch.from_numpy(v[:1]) for k, v in X.example_inputs(1, size, seed=0).items()}
    from torch.utils.flop_counter import FlopCounterMode

    for p in model.parameters():  # the counter's module tracker hooks grad-requiring inputs; none here
        p.requires_grad_(False)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(ex["image"], ex["ray_map"], ex["meta"])
    flops = int(fc.get_total_flops())
    with torch.no_grad(), FlopCounterMode(display=False) as fcb:
        model.backbone(ex["image"])
    flops_backbone = int(fcb.get_total_flops())
    print(
        f"params total {params['total'] / 1e6:.2f} M; FLOPs {flops / 1e9:.1f} G (backbone {flops_backbone / 1e9:.1f} G)",
        flush=True,
    )

    latency = {}
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        m = model.to(dev)
        exd = {k: v.to(dev) for k, v in ex.items()}
        with torch.no_grad():
            latency["pytorch fp32, RTX 4050, batch 1"] = timed(
                lambda: m(exd["image"], exd["ray_map"], exd["meta"]), sync=torch.cuda.synchronize
            )
            with torch.autocast("cuda", dtype=torch.float16):
                latency["pytorch autocast fp16, RTX 4050, batch 1"] = timed(
                    lambda: m(exd["image"], exd["ray_map"], exd["meta"]), sync=torch.cuda.synchronize
                )
            ex8 = {k: v.repeat(8, *([1] * (v.dim() - 1))) for k, v in exd.items()}
            r = timed(lambda: m(ex8["image"], ex8["ray_map"], ex8["meta"]), runs=10, sync=torch.cuda.synchronize)
            latency["pytorch fp32, RTX 4050, batch 8 (per frame)"] = {
                "median_ms": r["median_ms"] / 8,
                "p95_ms": r["p95_ms"] / 8,
                "runs": r["runs"],
            }
        latency["gpu"] = torch.cuda.get_device_name(0)
        model.to("cpu")
    session = X.onnx_session(Path(a.onnx))
    one = {k: v[:1] for k, v in X.example_inputs(1, size, seed=0).items()}
    latency["onnx runtime CPU, batch 1"] = timed(lambda: X.run_onnx(session, one), warm=3, runs=10)
    latency["cpu"] = platform.processor()
    bench = None
    if AIBENCH.exists():
        cmd = [str(AIBENCH), "--model", str(Path(a.onnx).parent), "--provider", "all", "--runs", "20", "--warmup", "3", "--json"]
        if a.frame:
            cmd += ["--picture", a.frame]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False).stdout
            bench = json.loads(out[out.index("{") :])
            for r in bench.get("rows", []):
                if r.get("ok"):
                    latency[f"CloudScope ONNX Runtime {r['provider']}, batch 1"] = {
                        "median_ms": r["p50_ms"],
                        "p95_ms": r["p95_ms"],
                        "mean_ms": r["mean_ms"],
                        "runs": r["runs"],
                        "preprocess_ms": r.get("preprocess_ms"),
                    }
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            print(f"aibench not used: {e}", flush=True)
    result = {
        "params": params,
        "flops_forward_512": flops,
        "flops_backbone_512": flops_backbone,
        "input": f"{size} x {size}",
        "onnx": str(Path(a.onnx).relative_to(ROOT)),
        "latency": latency,
        "pi5": "not measured: no Raspberry Pi 5 on hand",
        "machine": platform.node(),
    }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    with Run(
        __import__("omegaconf").OmegaConf.create({"onnx": result["onnx"], "input": result["input"]}),
        name="efficiency",
        phase=a.phase,
    ) as run:
        run.finish(
            params_m=params["total"] / 1e6,
            gflops=flops / 1e9,
            **{k.replace(" ", "_").replace(",", "")[:40]: v["median_ms"] for k, v in latency.items() if isinstance(v, dict)},
        )
    write_doc(result, run.run_id)
    print(f"done in {time.perf_counter() - t0:.0f} s -> {OUT}, {DOC}", flush=True)
    return 0


def write_doc(r: dict, run_id: str) -> None:
    lines = [
        "# Efficiency (P094)",
        "",
        f"The deployed graph (`{r['onnx']}`, {r['input']} input), run `{run_id}` on {r['machine']}.",
        "",
        "## Parameters",
        "",
        "| Component | Parameters |",
        "|---|---|",
    ]
    for k, v in r["params"].items():
        lines.append(f"| {k} | {v / 1e6:.3f} M |")
    lines += [
        "",
        f"FLOPs per frame (one forward pass at {r['input']}): **{r['flops_forward_512'] / 1e9:.1f} GFLOPs**, of which the backbone "  # noqa: E501
        f"{r['flops_backbone_512'] / 1e9:.1f} GFLOPs; everything above the backbone costs {(r['flops_forward_512'] - r['flops_backbone_512']) / 1e9:.2f} GFLOPs.",  # noqa: E501
        "",
        "## Latency",
        "",
        "| Setting | Median ms | p95 ms | Runs | Note |",
        "|---|---|---|---|---|",
    ]
    for k, v in r["latency"].items():
        if isinstance(v, dict):
            note = f"preprocess {v['preprocess_ms']:.0f} ms (host resize of a 16 MP frame)" if v.get("preprocess_ms") else ""
            lines.append(f"| {k} | {v['median_ms']:.1f} | {v['p95_ms']:.1f} | {v['runs']} | {note} |")
    lines += [
        f"| Raspberry Pi 5 | — | — | — | {r['pi5']} |",
        "",
        f"GPU: {r['latency'].get('gpu', '—')}; CPU: {r['latency'].get('cpu', '—')}. The CloudScope rows come from "
        "`cloudscope-aibench` (ONNX Runtime 1.24 inside the desktop application) and are the numbers the demo runs at; "
        "the PyTorch rows are the research model without export. NFR-PERF-05 (CloudScope) asks for the latency "
        "targets here.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
