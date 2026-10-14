# SPDX-License-Identifier: MIT
"""Time between values of the waveform-derived numerics in VitalDB. Aggregates only.

    python tools/numeric_intervals.py --root R --cases 1,2,... --out part.json
    python tools/numeric_intervals.py --merge part1.json part2.json --out all.json

For each numeric track, accumulated over the cases given:
  rec   intervals between successive records (the recorder's stamps)
  chg   intervals between successive *changes* of value (a repeat of the same value is not a new value)
  fine  record intervals 0..15 s in 5 ms bins (jitter around the nominal period)
  step  the most common non-zero absolute value steps (value resolution)
  case_median_rec / case_median_chg  one median per case (spread between cases)
  sizes counts of samples per record
Stretches with a record gap > 60 s are not counted as intervals (recording pauses, device off).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

TRACKS = [
    "Solar8000/HR", "Solar8000/PLETH_HR", "Solar8000/PLETH_SPO2",
    "Solar8000/ART_SBP", "Solar8000/ART_MBP", "Solar8000/ART_DBP",
    "Solar8000/FEM_SBP", "Solar8000/FEM_MBP", "Solar8000/FEM_DBP",
    "Solar8000/CVP", "Solar8000/ST_II", "Solar8000/RR", "Solar8000/RR_CO2", "Solar8000/ETCO2",
    "Solar8000/VENT_PIP", "Solar8000/VENT_RR",
    "Solar8000/NIBP_SBP",
    "Primus/ETCO2", "Primus/RR_CO2", "Primus/PIP_MBAR", "Primus/PEEP_MBAR", "Primus/MAWP_MBAR",
    "Primus/TV", "Primus/MV",
    "BIS/BIS", "BIS/SEF", "BIS/EMG", "BIS/SR", "BIS/SQI",
]
LOG_EDGES = np.logspace(-2, np.log10(3600), 241)
FINE_EDGES = np.arange(0, 15.0001, 0.005)
MAX_GAP = 60.0


def empty():
    return {"cases": 0, "records": 0, "rec": np.zeros(LOG_EDGES.size - 1, np.int64),
            "chg": np.zeros(LOG_EDGES.size - 1, np.int64), "fine": np.zeros(FINE_EDGES.size - 1, np.int64),
            "step": Counter(), "sizes": Counter(), "case_median_rec": [], "case_median_chg": []}


def add(acc, t, x, sizes):
    o = np.argsort(t, kind="stable")
    t, x = t[o], x[o]
    ok = np.isfinite(x)
    t, x = t[ok], x[ok]
    if t.size < 3:
        return
    acc["cases"] += 1
    acc["records"] += int(t.size)
    acc["sizes"].update(Counter(int(s) for s in np.asarray(sizes)))
    d = np.diff(t)
    keep = d <= MAX_GAP
    acc["rec"] += np.histogram(d[keep], LOG_EDGES)[0]
    acc["fine"] += np.histogram(d[keep], FINE_EDGES)[0]
    if keep.any():
        acc["case_median_rec"].append(float(np.median(d[keep])))
    # changes: break the series at long gaps so a pause is not counted as one interval
    seg = np.r_[0, np.cumsum(~keep)]
    chg = np.r_[False, (np.diff(x) != 0) & keep]
    tc, sc = t[chg], seg[chg]
    dc = np.diff(tc)[np.diff(sc) == 0]
    acc["chg"] += np.histogram(dc, LOG_EDGES)[0]
    if dc.size:
        acc["case_median_chg"].append(float(np.median(dc)))
    st = np.abs(np.diff(x))
    st = st[st > 0]
    acc["step"].update(Counter(np.round(st, 3).tolist()))


def scan(root, cases):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    A = {n: empty() for n in TRACKS}
    for c in cases:
        p = Path(root) / "vital_files" / f"{c:04d}.vital"
        try:
            v = read_vital(p, track_names=set(TRACKS))
        except Exception:
            continue
        for n, tr in v.tracks.items():
            if n not in A or not tr.record_times.size:
                continue
            t = np.asarray(tr.record_times, dtype=float)
            x = np.asarray(tr.physical(), dtype=float)
            sizes = np.asarray(tr.record_sizes) if tr.record_sizes is not None else np.ones(t.size)
            if x.size != t.size:          # multi-sample records: keep the first of each
                continue
            add(A[n], t, x, sizes)
    return A


def to_json(A):
    out = {}
    for n, a in A.items():
        out[n] = {"cases": a["cases"], "records": a["records"], "rec": a["rec"].tolist(), "chg": a["chg"].tolist(),
                  "fine": a["fine"].tolist(), "step": {str(k): v for k, v in a["step"].most_common(40)},
                  "sizes": {str(k): v for k, v in a["sizes"].items()},
                  "case_median_rec": a["case_median_rec"], "case_median_chg": a["case_median_chg"]}
    return {"log_edges": LOG_EDGES.tolist(), "fine_edges": [0, 15, 0.005], "tracks": out}


def merge(paths):
    M = None
    for p in paths:
        J = json.load(open(p))
        if M is None:
            M = J
            continue
        for n, a in J["tracks"].items():
            m = M["tracks"][n]
            m["cases"] += a["cases"]
            m["records"] += a["records"]
            for k in ("rec", "chg", "fine"):
                m[k] = (np.asarray(m[k]) + np.asarray(a[k])).tolist()
            for k in ("step", "sizes"):
                c = Counter(m[k])
                c.update(a[k])
                m[k] = dict(c)
            for k in ("case_median_rec", "case_median_chg"):
                m[k] += a[k]
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--merge", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.merge:
        J = merge(a.merge)
    else:
        J = to_json(scan(a.root, [int(x) for x in a.cases.split(",") if x]))
    json.dump(J, open(a.out, "w"))


if __name__ == "__main__":
    main()
