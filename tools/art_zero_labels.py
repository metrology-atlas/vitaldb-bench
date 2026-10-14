# SPDX-License-Identifier: MIT
"""Zeroing labels from tools/art_zero_jumps.py output. This machine only.

    python tools/art_zero_labels.py --jumps out.jsonl [...] --out-zero art_zero_labels.csv \
        --out-suspect art_pressure_suspect_labels.csv [--dbp-thr 7.0]

A zero access is an arterial-line access with an open-to-air plateau. Each gets one label over the access
(start to recovery):

    art_zero:ok        |diastolic change| across it <= --dbp-thr
    art_zero:suspect   larger: the calibration probably moved
    art_zero:unknown   no clean before/after windows

with the changes (MAP, SBP, DBP, pulse pressure = SBP - DBP, pulse duration = width at 50% height), the pressure before, and the open-to-air readings. The default
threshold is the 90th percentile of |diastolic change| between random clean 20 s windows the same gap apart
(6.9 mmHg in 206 cases). For each suspect zero, art_pressure_suspect spans the pressures it may have
recalibrated: from the previous zero access's recovery (or the start of ART) to this zero's start.
"""

from __future__ import annotations

import argparse
import csv
import json

ZERO_FIELDS = ["case", "label", "start_s", "end_s", "d_map", "d_sbp", "d_dbp", "d_pp", "d_pulse_duration_ms", "pre_map", "pre_sbp",
               "pre_dbp", "air_first", "air_last", "air_step", "air_seconds", "kinds", "window_s", "guard_s", "dbp_thr"]
SUSPECT_FIELDS = ["case", "label", "start_s", "end_s", "zero_start_s", "zero_d_dbp", "since"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jumps", nargs="+", required=True)
    ap.add_argument("--out-zero", required=True)
    ap.add_argument("--out-suspect", required=True)
    ap.add_argument("--dbp-thr", type=float, default=7.0)
    a = ap.parse_args()
    n = {"ok": 0, "suspect": 0, "unknown": 0}
    seen = set()
    with open(a.out_zero, "w", newline="") as fz, open(a.out_suspect, "w", newline="") as fs:
        wz = csv.DictWriter(fz, fieldnames=ZERO_FIELDS)
        ws = csv.DictWriter(fs, fieldnames=SUSPECT_FIELDS)
        wz.writeheader()
        ws.writeheader()
        for path in a.jumps:
            for line in open(path):
                r = json.loads(line)
                c = r["case"]
                if c in seen or "error" in r or "art_start_abs" not in r:
                    continue
                seen.add(c)
                prev_end, since = r["art_start_abs"], "recording start"
                for q in sorted(r["accesses"], key=lambda q: q["start_abs"]):
                    if not q.get("air"):
                        continue
                    d = q.get("delta") or {}
                    ddbp = d.get("dbp")
                    flag = "unknown" if ddbp is None else ("suspect" if abs(ddbp) > a.dbp_thr else "ok")
                    n[flag] += 1
                    pre = q.get("pre") or {}
                    air = q["air"]
                    wz.writerow({"case": c, "label": f"art_zero:{flag}", "start_s": q["start_abs"], "end_s": q["recovered_abs"],
                                 "d_map": d.get("map"), "d_sbp": d.get("sbp"), "d_dbp": ddbp, "d_pp": None if d.get("sbp") is None or ddbp is None else round(d["sbp"] - ddbp, 2),
                                 "d_pulse_duration_ms": d.get("pw50_ms"),
                                 "pre_map": pre.get("map"), "pre_sbp": pre.get("sbp"), "pre_dbp": pre.get("dbp"),
                                 "air_first": air[0]["first"], "air_last": air[-1]["last"],
                                 "air_step": round(air[-1]["last"] - air[0]["first"], 2),
                                 "air_seconds": sum(x["n"] for x in air), "kinds": "+".join(q["kinds"]),
                                 "window_s": r.get("win"), "guard_s": r.get("guard"), "dbp_thr": a.dbp_thr})
                    if flag == "suspect" and q["start_abs"] > prev_end:
                        ws.writerow({"case": c, "label": "art_pressure_suspect", "start_s": prev_end, "end_s": q["start_abs"],
                                     "zero_start_s": q["start_abs"], "zero_d_dbp": ddbp, "since": since})
                    prev_end, since = q["recovered_abs"], "previous zero"
    print(n)


if __name__ == "__main__":
    main()
