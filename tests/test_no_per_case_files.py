# SPDX-License-Identifier: MIT
"""Guard: per-case result files stay on the analysis machine.

Case ids may appear where they help: a doc, figure or example that discusses a specific case names it (VitalDB is
open data and its case ids are public). What must not be published is bulk per-case output (one row per case or
per beat): those files are written next to the analysis and git ignores them. This test fails if one is tracked.
Aggregate CSV files are allowed under results/.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PER_CASE_SUFFIXES = {".jsonl", ".parquet", ".vital", ".db", ".sqlite", ".duckdb"}


def _tracked() -> list[str]:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True,
                              check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):        # not a git checkout: scan the tree
        return [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


def test_no_per_case_files_tracked():
    bad = [n for n in _tracked() if Path(n).suffix.lower() in PER_CASE_SUFFIXES
           or (Path(n).suffix.lower() == ".csv" and not n.startswith("results/") and not n.startswith("tests/"))]
    assert not bad, "per-case files must stay on the analysis machine:\n" + "\n".join(bad)
