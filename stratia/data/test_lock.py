"""Test-set lock (P039): the test files of every protocol are hashed into `test_lock.json` (committed), a final
evaluation on them needs a logged reason, and CI refuses training configs that reference a test manifest."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from stratia.data.splits import sha256_of_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "test_lock.json"
FINAL_LOG = REPO_ROOT / "runs" / "final_eval_log.md"
TEST_FILES = {"in_domain": ("source", "split", "test"), "lodo": ("fold", "role", "test"),
              "held_out_station": ("variant", "role", "test")}


def test_entries(splits_dir: str | Path) -> dict[str, dict]:
    """name -> {n, sha256} for every test part of every protocol file under `splits_dir`."""
    out = {}
    for protocol, (key, col, value) in TEST_FILES.items():
        path = Path(splits_dir) / f"{protocol}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["sample_id", key, col])
        for k, part in df[df[col] == value].groupby(key):
            out[f"{protocol}/{k}/test"] = {"n": int(len(part)), "sha256": sha256_of_ids(part.sample_id)}
    return out


def build_lock(splits_dir: str | Path, config_version: int, seed: int, manifest_hash: str) -> dict:
    return {"locked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "config_version": config_version, "seed": seed,
            "manifest_sample_id_sha256_prefix": manifest_hash, "tests": test_entries(splits_dir)}


def verify_lock(lock: dict, splits_dir: str | Path) -> list[str]:
    """Problems between the lock and the split files on disk (missing files are reported, not ignored)."""
    problems = []
    current = test_entries(splits_dir)
    for name, entry in lock.get("tests", {}).items():
        if name not in current:
            problems.append(f"{name}: locked but not present in {splits_dir}")
        elif current[name]["sha256"] != entry["sha256"] or current[name]["n"] != entry["n"]:
            problems.append(f"{name}: differs from the lock ({current[name]['n']} vs {entry['n']} ids)")
    for name in current:
        if name not in lock.get("tests", {}):
            problems.append(f"{name}: present on disk but not locked")
    return problems


def well_formed(lock: dict) -> list[str]:
    problems = []
    for key in ("locked_at", "config_version", "seed", "tests"):
        if key not in lock:
            problems.append(f"missing key {key!r}")
    for name, entry in lock.get("tests", {}).items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))) or int(entry.get("n", 0)) <= 0:
            problems.append(f"{name}: malformed entry")
    return problems


def require_final_reason(reason: str, lock: dict, log_path: str | Path = FINAL_LOG, run: str = "") -> Path:
    """A final evaluation on locked test sets must state why; the reason is appended to the log and returned."""
    if not reason or len(reason.strip()) < 10:
        raise ValueError("a final evaluation on the locked test sets needs a reason of at least ten characters")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        header = "# Final evaluations on the locked test sets\n\n| When (UTC) | Lock | Run | Reason |\n|---|---|---|---|\n"
        log_path.write_text(header, encoding="utf-8", newline="\n")
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    with log_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"| {stamp} | {lock.get('locked_at', '?')} | {run} | {reason.strip()} |\n")
    return log_path


FORBIDDEN_PATTERNS = (re.compile(r"splits[/\\][^\s\"']*test"), re.compile(r"\btest\.parquet\b"))


def scan_training_configs(config_dir: str | Path) -> list[str]:
    """Training configs must not reference a test manifest. A config is a training config when it has a top-level
    `train` or `training` section; any string under it matching a test-file pattern, or `split: test`, is reported."""
    problems = []
    for path in sorted(Path(config_dir).glob("**/*.y*ml")):
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            problems.append(f"{path.name}: not valid YAML ({e})")
            continue
        if not isinstance(cfg, dict):
            continue
        for section in ("train", "training"):
            if section in cfg:
                for where, value in _walk(cfg[section], section):
                    if isinstance(value, str) and (any(p.search(value) for p in FORBIDDEN_PATTERNS)
                                                   or (where.endswith(".split") and value.strip().lower() == "test")):
                        problems.append(f"{path.name}: {where} references a test set ({value!r})")
    return problems


def _walk(node, prefix):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{prefix}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{prefix}[{i}]")
    else:
        yield prefix, node


def load_lock(path: str | Path = LOCK_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
