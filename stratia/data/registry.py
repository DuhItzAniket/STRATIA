"""Data registry (configs/datasets.yaml) and machine paths (configs/paths.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
REQUIRED_FIELDS = {"name", "tasks", "source_url", "licence", "licence_status", "status"}


def load_paths(path: Path | None = None) -> dict:
    p = path or REPO_ROOT / "configs" / "paths.yaml"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: copy configs/paths.example.yaml to configs/paths.yaml and edit it")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def load_registry(path: Path | None = None) -> dict:
    p = path or REPO_ROOT / "configs" / "datasets.yaml"
    reg = yaml.safe_load(p.read_text(encoding="utf-8"))["datasets"]
    for key, d in reg.items():
        missing = REQUIRED_FIELDS - d.keys()
        if missing:
            raise ValueError(f"dataset {key!r} lacks fields {sorted(missing)}")
        if d["licence_status"] not in {"verified", "unverified"}:
            raise ValueError(f"dataset {key!r}: bad licence_status {d['licence_status']!r}")
        if d["status"] not in {"present", "partial", "missing", "not_downloaded"}:
            raise ValueError(f"dataset {key!r}: bad status {d['status']!r}")
    return reg


def dataset_dirs(entry: dict, data_root: Path) -> dict[str, Path]:
    if "local_paths" in entry:
        return {k: data_root / v for k, v in entry["local_paths"].items()}
    if "local_path" in entry:
        return {"main": data_root / entry["local_path"]}
    return {}


def count_images(folder: Path, globs: list[str] | str | None = None, exclude_macos: bool = True) -> int:
    """Count image files under `folder`, optionally only those matching `globs` (masks live beside images)."""
    if isinstance(globs, str):
        globs = [globs]
    candidates = [p for g in globs for p in folder.glob(g)] if globs else folder.rglob("*")
    n = 0
    for p in candidates:
        if p.suffix.lower() not in IMAGE_EXT or not p.is_file():
            continue
        if exclude_macos and ("__MACOSX" in p.parts or p.name.startswith("._")):
            continue
        n += 1
    return n
