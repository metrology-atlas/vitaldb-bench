# SPDX-License-Identifier: MIT
"""Recover each VitalDB case's clock time of day (the date stays hidden).

VitalDB stores every time relative to ``casestart`` (= 0, the start of the .vital recording; the files are
anchored at 2100-01-01 00:00:00 UTC). ``clinical_data.csv`` still carries the time of day:

1. Minute: ``adm`` and ``dis`` are dates (midnight), relative to casestart truncated to the minute
   (``dis - adm`` is a whole number of days in every case; ``adm mod 60 = 0`` in 6,386 of 6,388). So the
   case start's clock minute is ``(-adm) mod 86400``.
2. Seconds: ``anestart``, ``aneend``, ``opstart`` and ``opend`` are EMR clock times on whole minutes (all
   four share one seconds phase in 90% of cases; opend is off it in 612). The seconds past the minute at
   casestart are ``(-t) mod 60`` for the phase that at least three of the four share.

``tod_s = minute + seconds`` is the case start's time of day on the EMR clock (local time, KST), in seconds
after midnight. Truncation (not rounding) of casestart to the minute is what puts 92-98% of opstart, aneend
and opend on absolute 5-minute marks (rounding gives ~48%). The EMR clock's offset from the recorder clock
is not known. Cases where adm is not on a minute or fewer than three events agree get no clock.

``shift_s`` moves the file's anchor (2100-01-01 00:00 UTC at casestart) to 2100-01-01 00:00 local time plus
tod_s, as UTC: ``tod_s - 3600 * tz_hours``. Relative times are unchanged. The date stays fictional.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

EMR = ("anestart", "aneend", "opstart", "opend")
METHOD = ("casestart time of day = (-adm mod 86400) [minute; adm is the admission date relative to casestart "
          "truncated to the minute] + (-t mod 60) [seconds; the phase shared by >= 3 of anestart, aneend, opstart, "
          "opend]; EMR clock, local time")
FILE = "case_clock.json"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def time_of_day(row: dict) -> dict:
    """Recover the case start's clock time of day from one row of ``clinical_data.csv``.

    Args:
        row: The case's row, as ``csv.DictReader`` gives it (values are strings); ``adm`` and the four EMR
            times ``anestart``, ``aneend``, ``opstart`` and ``opend`` are used.

    Returns:
        ``{"tod_s", "minute_s", "second", "events_agree", "status": "ok"}`` when a clock is recovered, else
        ``{"status": <reason>}``. ``tod_s`` is seconds after local midnight on the EMR clock.

    Examples:
        >>> adm = -(2 * 86400 + 7 * 3600 + 43 * 60)     # admission date, two days before a 07:43 start
        >>> row = {"adm": str(adm), "anestart": "-552", "aneend": "10848", "opstart": "1668", "opend": "10368"}
        >>> r = time_of_day(row)
        >>> r["status"], r["tod_s"] == 7 * 3600 + 43 * 60 + 12
        ('ok', True)
    """
    adm = _f(row.get("adm"))
    if adm is None:
        return {"status": "no adm"}
    if round(adm) % 60:
        return {"status": "adm not on a minute"}
    ev = [_f(row.get(k)) for k in EMR]
    ph = Counter(int(round(x)) % 60 for x in ev if x is not None)
    if not ph:
        return {"status": "no EMR events"}
    phase, n = ph.most_common(1)[0]
    if n < 3:
        return {"status": "EMR events disagree on the minute phase"}
    minute_s = int(round(-adm)) % 86400
    second = (-phase) % 60
    return {"tod_s": (minute_s + second) % 86400, "minute_s": minute_s, "second": second, "events_agree": n,
            "status": "ok"}


def clock_table(clinical_csv: str | Path) -> dict[int, dict]:
    """Recover the clock time of day of every case in ``clinical_data.csv``.

    Args:
        clinical_csv: Path to the release's ``clinical_data.csv``.

    Returns:
        ``{caseid: time_of_day(row)}`` for every row.
    """
    with open(clinical_csv, encoding="utf-8-sig", newline="") as fh:
        return {int(r["caseid"]): time_of_day(r) for r in csv.DictReader(fh)}


def shift_seconds(tod_s: int, tz_hours: float) -> int:
    """Anchor shift (s) that puts a case's start at its clock time of day, expressed in UTC.

    Args:
        tod_s: Case start's time of day on the local clock, seconds after midnight.
        tz_hours: Local time zone offset from UTC in hours (Korea Standard Time is 9).

    Returns:
        Seconds to add to every stored time stamp of the case.

    Examples:
        >>> shift_seconds(7 * 3600, 9)      # 07:00 KST is 22:00 UTC the day before
        -7200
    """
    return int(round(tod_s - tz_hours * 3600))


def write_table(path: str | Path, table: dict[int, dict], tz_hours: float) -> dict:
    """Write the per-case anchor shifts (only cases with a clock) beside an AtriumDB dataset.

    Args:
        path: The ``case_clock.json`` file to write.
        table: Output of :func:`clock_table`.
        tz_hours: Local time zone offset from UTC in hours.

    Returns:
        The document written: method, anchor, the cases left unshifted, and ``shift_s`` per shifted case.
    """
    cases = {str(c): {**v, "shift_s": shift_seconds(v["tod_s"], tz_hours)} for c, v in table.items() if v.get("status") == "ok"}
    doc = {"enabled": True, "tz_hours": tz_hours, "method": METHOD,
           "anchor": "2100-01-01 00:00 local time at the case's clock time of day; date fictional",
           "unshifted": sorted(int(c) for c, v in table.items() if v.get("status") != "ok"), "cases": cases}
    Path(path).write_text(json.dumps(doc, indent=1))
    return doc


def load_shifts_ns(dataset_dir: str | Path) -> dict[int, int]:
    """Per-case anchor shift in ns from ``<dataset>/case_clock.json``.

    Args:
        dataset_dir: The AtriumDB dataset folder.

    Returns:
        ``{caseid: shift_ns}``; empty when the dataset has no ``case_clock.json`` or it is disabled (the
        default build, on the 2100-01-01 anchor).
    """
    p = Path(dataset_dir) / FILE
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    if not doc.get("enabled"):
        return {}
    return {int(c): int(v["shift_s"]) * 10**9 for c, v in doc["cases"].items()}
