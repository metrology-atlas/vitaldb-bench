# SPDX-License-Identifier: MIT
"""Guard: no VitalDB case ids in anything this repository publishes.

Case ids are per-case information. They belong in per-case files on the analysis machine (which git
ignores), never in code, docs, results, examples or commit messages. This test scans every tracked file
and every commit message for a case id in a case context and fails if it finds one.

What counts as a case id here: a number from 4 up, used as a case (``case 1234``, ``cases=[12, 34]``,
``--case 1234``, ``c != 1234``, ``1234.vital``, ``vitaldb-1234``). The numbers 0-3 are allowed, as
placeholders in usage examples (``--cases 1,2``).

``GRANDFATHERED`` lists files that held case ids before this check existed, reviewed on 14 October 2026,
with how many matches each may contain. Do not add new entries: replace the id with a placeholder, or
read the case list from a per-case file instead.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDERS = {0, 1, 2, 3}
_NUMS = r"(\d{1,4}(?:\s*,\s*\d{1,4})*)(?![\w.:%-])"
PATTERNS = {
    "case word": re.compile(r"(?<![\w{.-])(?:case|cases|caseid|caseids|case_id|case_ids)\b\s*(?:[:=]|==|!=|\bin\b)?\s*"
                            r"[\[({]?\s*" + _NUMS, re.I),
    "comparison": re.compile(r"\bc\s*(?:==|!=)\s*(\d{1,4})\b"),
    "file name": re.compile(r"\b(\d{4})\.vital\b"),
    "device tag": re.compile(r"\bvitaldb-(\d{4})\b"),
    "command line": re.compile(r"--(?:case|cases|exclude|exclude-cases)[ =](\d{1,4}(?:,\d{1,4})*)", re.I),
}
GRANDFATHERED: dict[str, int] = {
    "scripts/t30_jobs.txt": 1,   # a test build that leaves one case out
    "scripts/t30_measure.sh": 1,   # the same case, left out of that build's fidelity check
    "tools/adhoc/sa_plot_model.py": 1,   # the six example cases of one figure
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".sh", ".toml", ".json", ".csv", ".cfg", ".ini", ".yml", ".yaml", ".tex",
                 ".bib", ".env", ".example", ""}


def _hits(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        for name, pat in PATTERNS.items():
            for m in pat.finditer(line):
                nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
                if any(n not in PLACEHOLDERS for n in nums):
                    out.append(f"{name}: {line.strip()[:120]}")
    return out


def _tracked() -> list[Path]:
    try:
        names = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True,
                               check=True).stdout.splitlines()
        return [ROOT / n for n in names]
    except (OSError, subprocess.CalledProcessError):        # not a git checkout: scan the tree
        return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


def test_no_case_ids_in_files():
    problems = []
    for path in _tracked():
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix not in TEXT_SUFFIXES or rel == "tests/test_no_case_ids.py" or not path.exists():
            continue
        try:
            hits = _hits(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        allowed = GRANDFATHERED.get(rel, 0)
        if len(hits) != allowed:
            problems.append(f"{rel} ({len(hits)} found, {allowed} allowed): " + " | ".join(hits[:3]))
    assert not problems, "possible VitalDB case ids:\n" + "\n".join(problems)


def test_no_case_ids_in_commit_messages():
    try:
        log = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%h %B"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return                                               # not a git checkout
    hits = _hits(log)
    assert not hits, "possible VitalDB case ids in commit messages:\n" + "\n".join(hits)
