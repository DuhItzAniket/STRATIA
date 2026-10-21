"""The exported STRATIA graph (P096): one ONNX model with the contract's three inputs and seven outputs.

``ExportModel`` composes, in one forward pass:

* the frozen backbone (DINOv3 ViT-S/16 through Hugging Face, or ``FakeBackbone`` for the dummy fixture) on the
  normalised ``image``: ``cls`` [N, d] and the patch tokens [N, 32 x 32, d] at 512 px;
* the dense heads (P074) on the full token grid: ``sky_parse_logits`` and ``layer_logits`` at H/4;
* the global heads (P073) and the cloud-base head (P075) on the token grid average-pooled 4 x 4 to 8 x 8, exactly as
  their training caches were built (``FeatureCache.build(pool=4)``), each with its own trained ``GeometryEncoder``;
* the contract's 32 x 32 ``ray_map`` pooled to the 8 x 8 grid those heads were trained on (``pool_ray_map``: mean of
  the valid unit directions, re-normalised; a pooled patch is valid when at least half of its sub-patches are);
* the calibration of P089 folded into the graph: the genus and cloud-base temperatures divide the logits before the
  outputs, the oktas CORAL probabilities are recomputed from the tempered cumulative logits, and the OOD statistics
  (mean, precision, z-scores) live in the ``OodScore`` buffers. Hosts apply no temperature (model card:
  ``calibration.folded_into_graph = true``); the threshold stays in the card.

Everything is a plain function of the inputs (``torch.where``, no Python branching on data), so the legacy ONNX
exporter traces one graph for the known and the unknown geometry states (P050).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import contract as C
from ..model import cbh_bins as B
from ..model.cbh import CbhHead
from ..model.dense import DenseHeads
from ..model.heads import CoralHead, GlobalHeads
from ..model.ood import OodScore
from ..model.skyray import GeometryEncoder
from ..model.stratia import STRATIA, FakeBackbone

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILE = REPO_ROOT / "schemas" / "model_card.schema.json"
GRID_POOL = 4            # 32 x 32 tokens -> 8 x 8 for the global and CBH heads (P067 / P073 caches)
OPSET = 17
DEFAULT_TEMPERATURES = {"genus": 1.0, "etage": 1.0, "oktas": 1.0, "cbh": 1.0}
METADATA_KEY = "stratia_model_card"   # the model card is also embedded in the ONNX file's metadata


# ------------------------------------------------------------------------------------------------ backbones
class HfBackbone(nn.Module):
    """A Hugging Face DINO model: normalised image [N, 3, H, W] -> (cls [N, d], patches [N, L, d])."""

    def __init__(self, model: nn.Module, n_skip: int):
        super().__init__()
        self.model, self.n_skip = model, n_skip

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(pixel_values=image)
        cls = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return cls, out.last_hidden_state[:, self.n_skip :, :]


class FakeBackboneWrapper(nn.Module):
    """``FakeBackbone`` behind the same interface (the dummy fixture and the tests)."""

    def __init__(self, backbone: FakeBackbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(image, patches=True)
        return out["cls"], out["patches"]


def hf_backbone(key: str = "dinov3_vits16") -> tuple[HfBackbone, dict]:
    """The frozen backbone of the training caches, in float32 on the CPU, and the facts the model card records."""
    from ..features.backbones import BACKBONES, FrozenBackbone

    fb = FrozenBackbone(key, device="cpu", dtype=torch.float32)
    spec = BACKBONES[key]
    n_skip = 1 + (int(getattr(fb.model.config, "num_register_tokens", 0)) if spec.family == "dinov3" else 0)
    facts = {"backbone": spec.hf_id, "patch_size": spec.patch, "mean": list(spec.mean), "std": list(spec.std), "dim": fb.dim}
    return HfBackbone(fb.model.eval(), n_skip), facts


# ------------------------------------------------------------------------------------------------ the graph
def pool_ray_map(ray_map: torch.Tensor, pool: int = GRID_POOL) -> torch.Tensor:
    """[N, 4, h, w] -> [N, 4, h/pool, w/pool]: the mean of the valid unit directions (invalid ones are zero by the
    contract), re-normalised; valid where at least half of the sub-patches are valid (the training ray maps were
    computed directly at the coarse patch; the difference is measured in the export report)."""
    valid = ray_map[:, 3:4]
    frac = F.avg_pool2d(valid, pool)
    dirs = F.avg_pool2d(ray_map[:, :3], pool) / frac.clamp_min(1e-6)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-6)
    ok = (frac >= 0.5).to(ray_map.dtype)
    return torch.cat([dirs * ok, ok], dim=1)


def pool_tokens(tokens: torch.Tensor, pool: int = GRID_POOL) -> torch.Tensor:
    """[N, L, d] -> [N, L / pool^2, d] by average pooling the square grid (as ``FeatureCache.build(pool=...)``)."""
    n, L, d = tokens.shape
    s = int(math.isqrt(L))
    grid = tokens.transpose(1, 2).reshape(n, d, s, s)
    return F.avg_pool2d(grid, pool).flatten(2).transpose(1, 2)


class ExportModel(nn.Module):
    def __init__(self, backbone: nn.Module, global_geo: GeometryEncoder, global_heads: GlobalHeads, cbh_geo: GeometryEncoder,
                 cbh_head: CbhHead, dense: DenseHeads, ood: OodScore, temperatures: dict[str, float] | None = None,
                 pool: int = GRID_POOL):
        super().__init__()
        self.backbone, self.global_geo, self.global_heads = backbone, global_geo, global_heads
        self.cbh_geo, self.cbh_head, self.dense, self.ood = cbh_geo, cbh_head, dense, ood
        self.pool = pool
        t = {**DEFAULT_TEMPERATURES, **(temperatures or {})}
        self.register_buffer("t_genus", torch.tensor(float(t["genus"])))
        self.register_buffer("t_etage", torch.tensor(float(t["etage"])))
        self.register_buffer("t_oktas", torch.tensor(float(t["oktas"])))
        self.register_buffer("t_cbh", torch.tensor(float(t["cbh"])))

    def forward(self, image: torch.Tensor, ray_map: torch.Tensor, meta: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cls, tokens = self.backbone(image)
        cls, tokens = cls.float(), tokens.float()
        dense = self.dense(tokens)
        pooled = pool_tokens(tokens, self.pool)
        rm = pool_ray_map(ray_map, self.pool)
        t, c = self.global_geo(pooled, cls, rm, meta)
        g = self.global_heads(t, c)
        t2, c2 = self.cbh_geo(pooled, cls, rm, meta)
        cb = self.cbh_head(t2, c2, rm)
        oktas = CoralHead.probs_from_cumulative(g["oktas_cum_logits"] / self.t_oktas)
        cbh = torch.softmax(cb["cbh_logits"] / self.t_cbh, dim=1)
        ood = self.ood(g["pooled"], g["genus_logits"])            # the statistics were fitted on the raw logits
        return (g["genus_logits"] / self.t_genus, g["etage_logits"] / self.t_etage, oktas, dense["sky_parse_logits"],
                dense["layer_logits"], cbh, ood)


# ------------------------------------------------------------------------------------------------ assembly
@dataclass
class TrainedParts:
    """Where the deployed weights come from (recorded in the model card)."""

    global_run: str
    cbh_run: str
    dense_run: str
    calibration_file: Path
    runs_root: Path = REPO_ROOT / "runs"
    extra: dict = field(default_factory=dict)


def _state(run_dir: Path, device: str = "cpu") -> dict:
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model = dict(state["model"])
    if state.get("ema"):                       # the global trainer evaluates the EMA weights (calibrate.py does the same)
        model.update(state["ema"]["shadow"])
    return model


def load_trained(parts: TrainedParts, backbone: nn.Module, d_model: int = 384) -> tuple[ExportModel, dict]:
    """The deployed graph from the three head checkpoints and the calibration file."""
    from omegaconf import OmegaConf

    g_dir, c_dir, d_dir = (parts.runs_root / r for r in (parts.global_run, parts.cbh_run, parts.dense_run))
    g_cfg = OmegaConf.load(g_dir / "config.yaml")
    g_net = STRATIA(None, d_model=d_model, dense=False, cbh=False, ood=False, global_hidden=int(g_cfg["hidden"]))
    g_state = {k[len("net.") :]: v for k, v in _state(g_dir).items() if k.startswith("net.")}
    g_net.load_state_dict(g_state)
    cbh_geo, cbh_head = GeometryEncoder(d_model), CbhHead(d_model)
    c_state = _state(c_dir)
    cbh_geo.load_state_dict({k[len("geo.") :]: v for k, v in c_state.items() if k.startswith("geo.")})
    cbh_head.load_state_dict({k[len("head.") :]: v for k, v in c_state.items() if k.startswith("head.")})
    dense = DenseHeads(d_model)
    dense.load_state_dict(_state(d_dir))
    calib = json.loads(parts.calibration_file.read_text(encoding="utf-8"))
    ood = OodScore(int(g_cfg["hidden"]))
    st = calib["ood_statistics"]
    with torch.no_grad():
        ood.mean.copy_(torch.tensor(st["mean"], dtype=torch.float32))
        ood.precision.copy_(torch.tensor(st["precision"], dtype=torch.float32))
        ood.stats.copy_(torch.tensor(st["stats"], dtype=torch.float32))
        ood.fitted.fill_(True)
    model = ExportModel(backbone, g_net.geo, g_net.global_heads, cbh_geo, cbh_head, dense, ood, calib["temperatures"]).eval()
    facts = {"global_run": parts.global_run, "cbh_run": parts.cbh_run, "dense_run": parts.dense_run,
             "calibration": {k: calib[k] for k in ("temperatures", "cbh_conformal_radius", "ood_threshold", "fitted_on")},
             "global_summary": json.loads((g_dir / "summary.json").read_text(encoding="utf-8")),
             "cbh_summary": json.loads((c_dir / "summary.json").read_text(encoding="utf-8")),
             "dense_summary": json.loads((d_dir / "summary.json").read_text(encoding="utf-8"))}
    return model, facts


def dummy_model(d_model: int = 32, hidden: int = 64, seed: int = 0) -> ExportModel:
    """A small, deterministic, untrained graph with the contract's interface (tests; CloudScope's contract test)."""
    torch.manual_seed(seed)
    net = STRATIA(None, d_model=d_model, dense=False, cbh=False, ood=False, global_hidden=hidden)
    ood = OodScore(hidden)
    with torch.no_grad():
        ood.fit(torch.randn(256, hidden), torch.randn(256, len(C.GENUS_CLASSES)))
    model = ExportModel(FakeBackboneWrapper(FakeBackbone(d_model, C.PATCH_SIZE)), net.geo, net.global_heads,
                        GeometryEncoder(d_model), CbhHead(d_model), DenseHeads(d_model, width=32, mid=32, out=32), ood,
                        {"genus": 2.0, "etage": 1.0, "oktas": 1.0, "cbh": 1.5})
    return model.eval()


# ------------------------------------------------------------------------------------------------ export
def example_inputs(n: int = 2, size: int = C.DEFAULT_INPUT_SIZE, seed: int = 0) -> dict[str, np.ndarray]:
    """Random contract inputs: one half of the batch with a valid ray map / meta, the other half all zeros."""
    rng = np.random.default_rng(seed)
    g = size // C.PATCH_SIZE
    image = rng.standard_normal((n, 3, size, size), dtype=np.float32)
    dirs = rng.standard_normal((n, 3, g, g)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    valid = (rng.random((n, 1, g, g)) > 0.3).astype(np.float32)
    ray_map = np.concatenate([dirs * valid, valid], axis=1)
    meta = np.tile(np.array([[0.6, 0.8, 1.0]], dtype=np.float32), (n, 1))
    ray_map[n // 2 :] = 0.0
    meta[n // 2 :] = 0.0
    return {"image": image, "ray_map": ray_map, "meta": meta}


def export_onnx(model: ExportModel, path: str | Path, size: int = C.DEFAULT_INPUT_SIZE, opset: int = OPSET,
                metadata: dict[str, str] | None = None) -> Path:
    """Write the ONNX file (dynamic batch axis on every input and output) and check it."""
    import onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ex = example_inputs(2, size)
    args = tuple(torch.from_numpy(ex[k]) for k in C.INPUT_NAMES)
    dynamic = {k: {0: "batch"} for k in C.INPUT_NAMES + C.OUTPUT_NAMES}
    with torch.no_grad():
        torch.onnx.export(model, args, str(path), opset_version=opset, input_names=list(C.INPUT_NAMES),
                          output_names=list(C.OUTPUT_NAMES), dynamic_axes=dynamic, dynamo=False, do_constant_folding=True)
    m = onnx.load(str(path))
    for k, v in (metadata or {}).items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return path


def onnx_session(path: str | Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_onnx(session, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    outs = session.run(None, {k: np.ascontiguousarray(inputs[k], dtype=np.float32) for k in C.INPUT_NAMES})
    return dict(zip(C.OUTPUT_NAMES, outs, strict=True))


def run_torch(model: ExportModel, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with torch.no_grad():
        outs = model(*(torch.from_numpy(inputs[k]) for k in C.INPUT_NAMES))
    return {k: v.numpy() for k, v in zip(C.OUTPUT_NAMES, outs, strict=True)}


def parity(model: ExportModel, session, inputs: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Per output: max |torch - ort| and that maximum relative to the output's largest magnitude."""
    a, b = run_torch(model, inputs), run_onnx(session, inputs)
    report = {}
    for k in C.OUTPUT_NAMES:
        diff = np.abs(a[k] - b[k])
        scale = max(float(np.abs(a[k]).max()), 1e-6)
        report[k] = {"max_abs": float(diff.max()), "max_rel": float(diff.max() / scale), "shape": list(b[k].shape)}
    return report


def check_contract(outputs: dict[str, np.ndarray], n: int, size: int = C.DEFAULT_INPUT_SIZE) -> list[str]:
    """Names, shapes, finiteness and the probability constraints of the contract; [] when everything holds."""
    problems = []
    want = C.output_shapes(n, size, size, k_cbh=B.K_BINS)
    for k, shape in want.items():
        if k not in outputs:
            problems.append(f"missing output {k}")
            continue
        if tuple(outputs[k].shape) != shape:
            problems.append(f"{k}: shape {tuple(outputs[k].shape)} != {shape}")
        if not np.isfinite(outputs[k]).all():
            problems.append(f"{k}: non-finite values")
    for k in ("oktas_probs", "cbh_probs"):
        if k in outputs and (outputs[k].min() < -1e-6 or np.abs(outputs[k].sum(1) - 1).max() > 1e-4):
            problems.append(f"{k}: not a probability distribution")
    return problems


# ------------------------------------------------------------------------------------------------ model card
def manifest_hash(manifest: str | Path = REPO_ROOT / "data" / "manifest.parquet") -> str | None:
    """SHA-256 over the ordered sample ids of the manifest (the data card's content hash is its first 16 digits)."""
    manifest = Path(manifest)
    if not manifest.exists():
        return None
    import pandas as pd

    ids = pd.read_parquet(manifest, columns=["sample_id"]).sample_id
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def git_sha() -> str:
    from ..utils.run import _git

    sha = _git("rev-parse", "HEAD")
    return sha if sha else "0" * 40


def model_card(model_version: str, backbone_facts: dict, trained: dict | None, size: int = C.DEFAULT_INPUT_SIZE,
               ood_threshold: float | None = None, training_hash: str | None = None, dummy: bool = False,
               created: dt.datetime | None = None) -> dict:
    """The model card of an export (stratia-contract v1, validated by ``validate_model_card``)."""
    calib = (trained or {}).get("calibration", {})
    temps = {**DEFAULT_TEMPERATURES, **calib.get("temperatures", {})}
    card = {
        "contract_version": C.CONTRACT_VERSION,
        "model_name": "stratia-dummy" if dummy else "stratia",
        "model_version": model_version,
        "created_utc": (created or dt.datetime.now(dt.UTC)).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha(),
        "input": {"height": size, "width": size, "patch_size": int(backbone_facts["patch_size"]),
                  "mean": [float(x) for x in backbone_facts["mean"]], "std": [float(x) for x in backbone_facts["std"]],
                  "resize": f"full-frame resize of the whole frame to {size} x {size} (OpenCV INTER_AREA), no crop, "
                            "no camera-mask zeroing (the heads were trained on transform ffr1 without it); "
                            "then scale to [0, 1] and normalise with mean / std"},
        "genus_classes": list(C.GENUS_CLASSES),
        "etage_classes": list(C.ETAGE_CLASSES),
        "sky_parse_classes": list(C.SKY_PARSE_CLASSES),
        "layer_classes": list(C.LAYER_CLASSES),
        "cbh_bin_edges_m": [float(x) for x in B.EDGES_M],
        "calibration": {"temperatures": {k: float(v) for k, v in temps.items()}, "folded_into_graph": True,
                        "cbh_conformal_radius_cdf": {k: float(v) for k, v in calib.get("cbh_conformal_radius", {}).items()},
                        "fitted_on": calib.get("fitted_on", "untrained dummy" if dummy else "")},
        "ood_threshold": float(ood_threshold if ood_threshold is not None else calib.get("ood_threshold", 0.0)),
        "training_data": ([{"dataset": "none (untrained dummy model)", "manifest_sha256": "0" * 64}] if dummy else
                          [{"dataset": "STRATIA manifest (CCSN, MGCD, SWIMCAT, SWIMSEG/SWINSEG, Montenegro, Almeria, "
                                       "Eye2Sky; docs/data/data_card.md)", "manifest_sha256": training_hash or "unknown"}]),
        "licences": ["Apache-2.0 (STRATIA code and head weights)",
                     "DINOv3 License (backbone weights facebook/dinov3-vits16-pretrain-lvd1689m): Built with DINOv3",
                     "training datasets keep their own licences (docs/data/data_card.md)"],
        "evaluation_report": "docs/data/onnx_export.md; docs/data/gate_g4.md; docs/data/calibration.md",
        "backbone": backbone_facts.get("backbone", "fake"),
        "outputs_note": {"cbh_probs": f"{B.K_BINS} height bins (edges cbh_bin_edges_m) + index {B.K_BINS} = no cloud overhead",
                         "ood_score": "z-scored energy + Mahalanobis average; above ood_threshold = show as uncertain",
                         "ray_map": "the 32 x 32 contract ray map is average-pooled to 8 x 8 inside the graph for the "
                                    "global and cloud-base heads; the dense heads use the full 32 x 32 token grid"},
        "trained_from": {k: v for k, v in (trained or {}).items() if k.endswith("_run")},
        "head_summaries": {k: v for k, v in (trained or {}).items() if k.endswith("_summary")},
        "onnx": {"opset": OPSET, "dynamic_axes": "batch", "inputs": list(C.INPUT_NAMES), "outputs": list(C.OUTPUT_NAMES)},
    }
    return card


def validate_model_card(card: dict, schema_file: Path = SCHEMA_FILE) -> list[str]:
    """Schema problems ([] when valid)."""
    import jsonschema

    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in v.iter_errors(card)]


def write_model_card(card: dict, path: str | Path) -> Path:
    problems = validate_model_card(card)
    if problems:
        raise ValueError("model card does not validate: " + "; ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    return path


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
