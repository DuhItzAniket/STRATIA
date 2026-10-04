"""Hardware budget profiling (P010): memory and throughput of DINOv3 backbones on this GPU.

    python scripts/profile_backbones.py --out docs/compute_profile.json

For each backbone and resolution, measures:
  * inference (bf16 autocast, no grad): peak memory and images/s at batch 32
  * training (fp16 autocast + AdamW, all parameters), with and without gradient checkpointing,
    and LoRA (rank 8 on attention projections, backbone frozen): largest batch that fits
    (powers of two up to 64) and images/s at that batch
Peak memory is torch.cuda.max_memory_allocated (allocator view; the driver also reserves memory).

The allocator is capped at --mem-fraction of physical VRAM (default 0.90). Without the cap, Windows
drivers with "CUDA sysmem fallback" spill into system RAM instead of raising out-of-memory, which makes
an oversized batch look like it fits while running orders of magnitude slower (observed on this laptop).
A batch is also rejected if its throughput collapses below 30% of the previous batch size's.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

MODELS = {"vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
          "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
          "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m"}


def load(name: str):
    from transformers import AutoModel

    return AutoModel.from_pretrained(MODELS[name]).cuda()


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def timed(fn, iters: int) -> float:
    fn()  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def inference(model, res: int, batch: int = 32) -> dict:
    model.eval()
    free()
    x = torch.randn(batch, 3, res, res, device="cuda")

    def step():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(pixel_values=x).last_hidden_state.float().mean().item()

    try:
        dt = timed(step, 5)
        return {"batch": batch, "img_per_s": round(batch / dt, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    except torch.cuda.OutOfMemoryError:
        return {"batch": batch, "oom": True}


def lora_wrap(model):
    from peft import LoraConfig, get_peft_model

    targets = sorted({n.split(".")[-1] for n, m in model.named_modules()
                      if isinstance(m, torch.nn.Linear) and any(k in n for k in ("q_proj", "k_proj", "v_proj", "qkv"))})
    if not targets:
        raise RuntimeError("no attention projection layers found for LoRA")
    return get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0.0)), targets


def train_try(model, res: int, batch: int) -> float | None:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(batch, 3, res, res, device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(pixel_values=x).last_hidden_state.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    try:
        return timed(step, 3)
    except torch.cuda.OutOfMemoryError:
        return None


def training(name: str, res: int, mode: str) -> dict:
    free()
    model = load(name)
    targets = None
    if mode == "lora":
        model, targets = lora_wrap(model)
    elif mode == "full_ckpt":
        model.gradient_checkpointing_enable()
    model.train()
    best = None
    for batch in (1, 2, 4, 8, 16, 32, 64):
        free()
        dt = train_try(model, res, batch)
        if dt is None:
            break
        ips = batch / dt
        if best is not None and ips < 0.3 * best["img_per_s"]:
            best["stopped"] = f"throughput collapsed at batch {batch} ({ips:.1f} img/s): memory spill suspected"
            break
        best = {"max_batch": batch, "img_per_s": round(ips, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    del model
    free()
    out = best or {"max_batch": 0, "note": "OOM at batch 1"}
    if targets:
        out["lora_targets"] = targets
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/compute_profile.json")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--res", nargs="+", type=int, default=[224, 384, 512])
    ap.add_argument("--mem-fraction", type=float, default=0.90)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction, 0)
    free_b, total_b = torch.cuda.mem_get_info()
    report = {"gpu": torch.cuda.get_device_name(0), "total_gb": round(total_b / 1e9, 2),
              "free_at_start_gb": round(free_b / 1e9, 2), "mem_fraction_cap": a.mem_fraction,
              "torch": torch.__version__, "results": {}}
    for name in a.models:
        for res in a.res:
            key = f"{name}@{res}"
            r = {}
            model = load(name)
            r["inference_bf16"] = inference(model, res)
            del model
            modes = ["full", "full_ckpt", "lora"] if name != "vitl16" else ["full_ckpt", "lora"]
            for mode in modes:
                r[f"train_{mode}"] = training(name, res, mode)
            report["results"][key] = r
            print(key, json.dumps(r), flush=True)
            Path(a.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")  # partial results
    return 0


if __name__ == "__main__":
    sys.exit(main())
