"""CI check for the test-set lock (P039).

    python scripts/check_test_lock.py

Fails when test_lock.json is missing or malformed, when a training config under configs/ references a test set,
or, if the split files exist locally, when they differ from the lock. In CI the data is absent, so only the first
two checks run there.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data import test_lock as tl  # noqa: E402


def main() -> int:
    if not tl.LOCK_PATH.exists():
        print("test_lock.json is missing")
        return 1
    lock = tl.load_lock()
    problems = tl.well_formed(lock) + tl.scan_training_configs(tl.REPO_ROOT / "configs")
    splits = tl.REPO_ROOT / "data" / "splits"
    if any(splits.glob("*.parquet")):
        problems += tl.verify_lock(lock, splits)
        checked = "lock well-formed, configs clean, split files match the lock"
    else:
        checked = "lock well-formed, configs clean (no split files locally)"
    for p in problems:
        print("FAILED:", p)
    if not problems:
        print(f"test lock OK: {len(lock['tests'])} test sets; {checked}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
