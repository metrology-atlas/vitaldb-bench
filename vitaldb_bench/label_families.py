# SPDX-License-Identifier: MIT
"""The derived label families: where each family's rows come from and what its labels mean.

Shared by both exports: :mod:`vitaldb_bench.timenet_labels` writes the families into a TimeNet build as
record annotations, and the GPL package vitaldb-bench-atriumdb (``vitaldb_bench_atriumdb.labels``) writes
them into an AtriumDB dataset. This module holds data and CSV reading only, so it stays MIT.

Times in the CSVs are the .vital files' own absolute time stamps in seconds. Every family has an
``available_at``, the earliest point its information could have been known (the availability rule). The
vitaldb_bench_atriumdb.labels module docstring describes each family.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

NS = 10**9


def seconds_to_ns(x: str | float) -> int:
    """Exact conversion of a time in seconds (text or a number) to the nearest nanosecond."""
    return round(Fraction(str(x)) * NS)


@dataclass(frozen=True)
class Family:
    """One label family: where its rows come from and how they become labels.

    Attributes:
        name: Family name; also the companion-table folder.
        csv: CSV file in the results folder.
        measure: Track the labels sit on (an AtriumDB measure, a TimeNet track), or None for the whole case.
        source: Label source (the library that produced the rows).
        prefix: Every label name of the family starts with this.
        start: CSV column with the start time, seconds (the file's own time stamps).
        end: CSV column with the end time.
        label: Function from a CSV row to the label name.
        measure_col: When set, each row names its own measure in this column.
        available_at: When the label could first be known, unless a row sets its own.
    """
    name: str
    csv: str
    measure: str | None
    source: str
    prefix: str                              # label names of the family start with this
    start: str
    end: str
    label: Callable[[dict], str]
    measure_col: str | None = None           # when set, each row names its own measure in this column
    available_at: str = "case_end"           # when the label could first be known, unless a row says otherwise


def _saw_name(r: dict) -> str:
    return f"ppg_sawtooth:{r['tier']}" + ("" if str(r.get("usable_for_pat")) == "True" else ":x")


FAMILIES = {f.name: f for f in (
    Family("ppg_timing", "ppg_timing_labels.csv", "SNUADC/PLETH", "physiosync", "ppg_timing:", "start_s", "end_s",
           lambda r: r["label"]),
    Family("ppg_sawtooth", "ppg_sawtooth_labels.csv", "SNUADC/PLETH", "physiosync", "ppg_sawtooth:", "start", "end",
           _saw_name),
    Family("ppg_sawtooth_chain", "ppg_sawtooth_chain_labels.csv", "SNUADC/PLETH", "physiosync", "ppg_sawtooth_chain",
           "start", "end", lambda r: "ppg_sawtooth_chain"),
    Family("ppg_offset", "ppg_offset_labels.csv", "SNUADC/PLETH", "physiosync", "ppg_offset", "start", "end",
           lambda r: "ppg_offset"),
    Family("art_line", "art_line_labels.csv", "SNUADC/ART", "lineaccess", "art_line:", "start_s", "end_s",
           lambda r: r["label"]),
    Family("art_zero", "art_zero_labels.csv", "SNUADC/ART", "lineaccess", "art_zero:", "start_s", "end_s",
           lambda r: r["label"]),
    Family("art_pressure_suspect", "art_pressure_suspect_labels.csv", "SNUADC/ART", "lineaccess",
           "art_pressure_suspect", "start_s", "end_s", lambda r: r["label"]),
    Family("art_flush_response", "art_flush_response_labels.csv", "SNUADC/ART", "lineaccess", "art_flush_response:",
           "start_s", "end_s", lambda r: r["label"]),
    Family("lab_draw", "lab_draw_labels.csv", "SNUADC/ART", "vitaldb-bench", "lab_draw:", "start_s", "end_s",
           lambda r: r["label"]),
    Family("lab_time", "lab_time_labels.csv", None, "vitaldb-bench", "lab_time:", "start_s", "end_s",
           lambda r: r["label"], available_at="lab_time_plus_turnaround"),
    Family("numeric_delay", "numeric_delay_labels.csv", None, "vitaldb-bench", "numeric_delay", "start_s", "end_s",
           lambda r: r["label"], measure_col="track"),
    Family("hr_delay_group", "hr_delay_group_labels.csv", "Solar8000/HR", "vitaldb-bench", "hr_delay_group:",
           "start_s", "end_s", lambda r: r["label"]),
    Family("bis_smoothing", "bis_smoothing_labels.csv", "BIS/BIS", "vitaldb-bench", "bis_smoothing:",
           "start_s", "end_s", lambda r: r["label"]),
)}


# Families renamed or removed; both label passes drop their labels from older builds.
# lab_result (until 13 October 2026) treated VitalDB's lab timestamp as the result time; it is now lab_time.
RETIRED = {"lab_result": ("vitaldb-bench", "lab_result")}       # family -> (label source, name prefix)


def read_family(results: Path, fam: Family) -> dict[int, list[dict]]:
    """The family's CSV rows, by case.

    Args:
        results: Folder that holds the label CSVs.
        fam: The family.

    Returns:
        ``{caseid: [row, ...]}``; empty when the CSV is missing.
    """
    by_case: dict[int, list[dict]] = defaultdict(list)
    p = results / fam.csv
    if not p.exists():
        return by_case
    with p.open(newline="") as fh:
        for r in csv.DictReader(fh):
            by_case[int(r["case"])].append(r)
    return by_case


def versions_from(results: Path) -> dict[str, str]:
    """Version string per label source from ``label_versions.json`` in the results folder.

    Args:
        results: Folder that holds the label CSVs.

    Returns:
        ``{source: version}``; empty when the file is missing.
    """
    p = results / "label_versions.json"
    return json.loads(p.read_text()) if p.exists() else {}
