"""Environment check: `python -m stratia.env_check [--json PATH] [--skip-hf]`.

Reports Python, package versions, GPU, driver and CUDA; runs a CUDA matmul and
mixed-precision smoke test; checks that the gated DINOv3 checkpoints are reachable
(config files only, never weights). Exits non-zero if any required check fails.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
import time

REQUIRED = ["torch", "torchvision", "transformers", "numpy", "pandas", "scipy", "sklearn",
            "xarray", "netCDF4", "h5py", "zarr", "PIL", "cv2", "imagehash", "pvlib", "omegaconf"]
DINOV3 = ["facebook/dinov3-vits16-pretrain-lvd1689m",
          "facebook/dinov3-vitb16-pretrain-lvd1689m",
          "facebook/dinov3-vitl16-pretrain-lvd1689m"]


def versions() -> dict:
    out = {}
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001 - report any import failure
            out[name] = f"MISSING ({type(e).__name__})"
    return out


def nvidia_smi() -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    q = "name,driver_version,memory.total,memory.used,temperature.gpu,power.limit"
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=20)
    if r.returncode != 0:
        return None
    vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")]
    return dict(zip(q.split(","), vals))


def cuda_smoke() -> dict:
    import torch

    res = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}
    if not res["available"]:
        return res
    dev = torch.device("cuda")
    res["device"] = torch.cuda.get_device_name(0)
    res["total_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    res["bf16_supported"] = torch.cuda.is_bf16_supported()
    a = torch.randn(2048, 2048, device=dev)
    b = torch.randn(2048, 2048, device=dev)
    ref = (a.double() @ b.double()).float()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    res["fp32_matmul_tflops"] = round(10 * 2 * 2048**3 / (time.perf_counter() - t0) / 1e12, 2)
    res["fp32_max_rel_err"] = float(((c - ref).abs().max() / ref.abs().max()).item())
    for dtype in (torch.float16, torch.bfloat16):
        with torch.autocast("cuda", dtype=dtype):
            d = a @ b
        res[f"autocast_{str(dtype).split('.')[-1]}_ok"] = bool(torch.isfinite(d).all().item())
    return res


def hf_access() -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    out = {}
    try:
        out["user"] = HfApi().whoami()["name"]
    except Exception as e:  # noqa: BLE001
        out["user"] = f"NOT LOGGED IN ({type(e).__name__})"
    for repo in DINOV3:
        try:
            hf_hub_download(repo, "config.json")
            out[repo] = "ok"
        except Exception as e:  # noqa: BLE001
            out[repo] = f"FAIL ({type(e).__name__})"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the report to this path")
    ap.add_argument("--skip-hf", action="store_true", help="skip the Hugging Face access check (offline/CI)")
    args = ap.parse_args()

    report = {"python": sys.version.split()[0], "platform": platform.platform(),
              "executable": sys.executable, "packages": versions(), "nvidia_smi": nvidia_smi()}
    failures = [k for k, v in report["packages"].items() if str(v).startswith("MISSING")]
    try:
        import numpy

        if int(numpy.__version__.split(".")[0]) >= 2:
            failures.append("numpy>=2 (pinned <2)")
        report["cuda"] = cuda_smoke()
        if not report["cuda"]["available"]:
            failures.append("CUDA unavailable")
        elif report["cuda"]["fp32_max_rel_err"] > 1e-3:
            failures.append("CUDA matmul inaccurate")
    except ImportError as e:
        failures.append(f"torch import failed: {e}")
    if not args.skip_hf:
        report["huggingface"] = hf_access()
        failures += [f"HF {k}" for k, v in report["huggingface"].items() if k != "user" and v != "ok"]
    report["failures"] = failures
    report["ok"] = not failures

    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
