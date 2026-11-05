# SPDX-License-Identifier: MIT
"""Statistics of the PPG timing and sawtooth labels, as quoted on the atlas variable pages ppg-timing-label and
ppg-sawtooth-label (timing: as stored, and at the availability time).

    python tools/label_stats.py --timing ppg_timing_labels.csv --sawtooth ppg_sawtooth_labels.csv \
        --ends $W/results/pat/numeric_delay_labels.csv --out label_stats.json

Per label file: the number of labels and recordings, the count per tier; the label length (support: min and max, s);
the as-stored bias (a label used at its start carries its whole interval: bias = -length; median, p5, p95); and the
bias held to the end of the recording, the label's availability time (end of recording minus label end, where the end
of a recording is the last end of its numeric delay labels; median, p5, p95). For sawtooth labels also the median
sync error SD per tier and the number usable for pulse arrival time. Percentiles are numpy's method "lower" (an
observed value) of the positive quantity, the length or the time to the end, the bias being minus the length; in
seconds, rounded to 0.1 s. This reproduces the statistics the pages quoted from the 31 Oct release run.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TIERS = ("A", "B", "C", "U", "I", "N")


def q(v, sign: int = 1) -> dict | None:
    """median, p5 and p95 of sign * v, v positive (percentiles by method "lower" on v)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    lo, hi = (float(np.percentile(v, p, method="lower")) for p in (5, 95))
    if sign < 0:
        lo, hi = -hi, -lo
    return {"n": int(v.size), "median": round(sign * float(np.median(v)), 1), "p5": round(lo, 1), "p95": round(hi, 1)}


def rows(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def ends(path: str) -> dict[int, float]:
    out: dict[int, float] = defaultdict(lambda: -np.inf)
    for r in rows(path):
        try:
            c, e = int(r["case"]), float(r["end_s"])
        except (KeyError, ValueError):
            continue
        out[c] = max(out[c], e)
    return dict(out)


def stats(R: list[dict], start: str, end: str, rec_end: dict[int, float]) -> dict:
    s = np.array([float(r[start]) for r in R])
    e = np.array([float(r[end]) for r in R])
    length = e - s
    avail = np.array([rec_end.get(int(r["case"]), np.nan) - x for r, x in zip(R, e)])
    tiers = Counter(r["tier"] for r in R)
    return {"labels": len(R), "recordings": len({r["case"] for r in R}),
            "tiers": {t: tiers[t] for t in TIERS if tiers[t]} | {t: n for t, n in tiers.items() if t not in TIERS},
            "support_s": [round(float(length.min()), 1), round(float(length.max()), 1)] if R else None,
            "bias_as_stored_s": q(length, -1), "bias_to_recording_end_s": q(avail),
            "no_recording_end": int(np.sum(~np.isfinite(avail))),
            "negative_to_end": int(np.sum(avail < 0))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timing", required=True)
    ap.add_argument("--sawtooth", required=True)
    ap.add_argument("--ends", required=True, help="numeric_delay_labels.csv: the end of each recording")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec_end = ends(a.ends)
    T, S = rows(a.timing), rows(a.sawtooth)
    out = {"inputs": {"timing": a.timing, "sawtooth": a.sawtooth, "ends": a.ends},
           "timing": stats(T, "start_s", "end_s", rec_end), "sawtooth": stats(S, "start", "end", rec_end)}
    by = defaultdict(list)
    for r in S:
        try:
            by[r["tier"]].append(float(r["sync_sd_ms"]))
        except (KeyError, ValueError):
            pass
    out["sawtooth"]["sync_sd_ms_median"] = {t: {"median": round(float(np.median(v)), 1), "n": len(v)}
                                            for t, v in sorted(by.items(), key=lambda x: TIERS.index(x[0])
                                                               if x[0] in TIERS else 99)}
    out["sawtooth"]["usable_for_pat"] = sum(r.get("usable_for_pat") == "True" for r in S)
    if T and "applied" in T[0]:
        out["timing"]["applied"] = sum(r["applied"] == "True" for r in T)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
