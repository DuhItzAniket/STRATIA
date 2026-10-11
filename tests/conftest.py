import os
import sys
from pathlib import Path

import pytest

# Make the `stratia` package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """On GitHub Actions, turn every failing test into a workflow annotation (`::error file=...`), which the public
    check-run annotations API exposes without a sign-in; the job log itself is not public."""
    outcome = yield
    report = outcome.get_result()
    if report.when in ("call", "setup") and report.failed and os.environ.get("GITHUB_ACTIONS"):
        path, lineno, _ = report.location
        msg = report.longreprtext.strip().splitlines()
        tail = " | ".join(line.strip() for line in msg[-6:])[:900].replace("\n", " ")
        print(f"\n::error file={path},line={int(lineno) + 1},title={item.nodeid}::{tail}", flush=True)
