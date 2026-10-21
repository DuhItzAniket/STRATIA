"""Reference outputs of the dummy export for host contract tests (P096; CloudScope P066).

    python scripts/make_contract_reference.py [--model tests/fixtures/stratia_dummy/stratia_dummy.onnx]
                                              [--out tests/fixtures/stratia_dummy/reference.json]

The inputs are defined by closed formulas that a host can reproduce without this file (see `make_inputs`), so
only the outputs are stored: every small head in full and the two dense heads by their per-channel means and a
4 x 4 lattice of sampled positions. A host that runs the same ONNX file through its own ONNX Runtime must match
these numbers to 1e-4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from stratia import contract as C  # noqa: E402

SAMPLE_ROWS = (0, 37, 74, 111)
SAMPLE_COLS = (0, 43, 86, 127)


def make_inputs(size: int = C.DEFAULT_INPUT_SIZE, patch: int = C.PATCH_SIZE) -> dict[str, np.ndarray]:
    """image[c, y, x] = ((7 x + 13 y + 29 c) mod 251) / 250 * 2 - 1; ray map toward (dx, dy, 0.8) per cell, with every
    cell whose row + column is a multiple of 7 invalid (zeros); meta = [0.6, 0.8, 1]."""
    c, y, x = np.meshgrid(np.arange(3), np.arange(size), np.arange(size), indexing="ij")
    image = (((7 * x + 13 * y + 29 * c) % 251) / 250.0 * 2.0 - 1.0).astype(np.float32)[None]
    g = size // patch
    r, col = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    dx = (col + 0.5) / g - 0.5
    dy = (r + 0.5) / g - 0.5
    dz = np.full_like(dx, 0.8)
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    valid = ((r + col) % 7 != 0).astype(np.float32)
    ray_map = np.stack([dx / norm * valid, dy / norm * valid, dz / norm * valid, valid]).astype(np.float32)[None]
    meta = np.array([[0.6, 0.8, 1.0]], dtype=np.float32)
    return {"image": image, "ray_map": ray_map, "meta": meta}


def summarise(name: str, value: np.ndarray) -> dict | list:
    value = value[0]
    if value.ndim == 1:
        return [float(v) for v in value]
    means = [float(value[ch].mean()) for ch in range(value.shape[0])]
    samples = [[float(value[ch, r, c]) for ch in range(value.shape[0])] for r in SAMPLE_ROWS for c in SAMPLE_COLS]
    return {"shape": list(value.shape), "channel_mean": means, "sample_rows": list(SAMPLE_ROWS),
            "sample_cols": list(SAMPLE_COLS), "samples": samples}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=ROOT / "tests" / "fixtures" / "stratia_dummy" / "stratia_dummy.onnx")
    ap.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures" / "stratia_dummy" / "reference.json")
    a = ap.parse_args()
    import onnxruntime as ort

    session = ort.InferenceSession(str(a.model), providers=["CPUExecutionProvider"])
    inputs = make_inputs()
    outputs = session.run(None, inputs)
    names = [o.name for o in session.get_outputs()]
    doc = {
        "description": "Outputs of stratia_dummy.onnx on ONNX Runtime CPU for the closed-form inputs of "
                       "scripts/make_contract_reference.py (STRATIA P096; CloudScope P066 contract test, tolerance 1e-4)",
        "model": a.model.name,
        "onnxruntime": ort.__version__,
        "inputs": {"image": "image[c, y, x] = ((7 x + 13 y + 29 c) mod 251) / 250 * 2 - 1 (512 x 512)",
                   "ray_map": "cell (row, col) of the 32 x 32 grid: unit vector of (dx, dy, 0.8) with "
                              "dx = (col + 0.5) / 32 - 0.5, dy = (row + 0.5) / 32 - 0.5; cells with "
                              "(row + col) mod 7 == 0 are invalid (all zeros)",
                   "meta": [0.6, 0.8, 1.0]},
        "outputs": {name: summarise(name, value) for name, value in zip(names, outputs, strict=True)},
    }
    a.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"{a.out} ({a.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
