# SPDX-License-Identifier: MIT
"""Flush-response labels from tools/flush_ringing.py output. This machine only.

    python tools/flush_response_labels.py --ringing full_fr.jsonl --out art_flush_response_labels.csv

One label per flush release, from the release to 300 ms after it (the ringing), named by the damping
coefficient zeta of its square-wave response:

    art_flush_response:underdamped       zeta < 0.2
    art_flush_response:low_damping       0.2 <= zeta < 0.45
    art_flush_response:moderate_damping  0.45 <= zeta < 0.7
    art_flush_response:high_damping      zeta >= 0.7
    art_flush_response:no_ringing        no undershoot (overdamped, a slow release, or not arterial afterwards)

Parameters: zeta, natural and damped frequency, fall time, plateau and settled pressure, undershoot and
overshoot, the change in natural frequency since the previous ringing release of the case (d_fn_hz; a sudden
fall marks a changed line), and the flush's access (start, recovery) with MAP/SBP/DBP 20 s before and after.
The ART samples are the monitor's analogue output, so the monitor's filtering is part of the response.
"""

from __future__ import annotations

import argparse
import csv
import json

FIELDS = ["case", "label", "start_s", "end_s", "zeta", "fn_hz", "fd_hz", "fall_ms", "plateau", "settled", "undershoot",
          "overshoot", "d_fn_hz", "minutes_since_prev", "access_start_s", "access_recovered_s", "access_kinds",
          "pre_map", "pre_sbp", "pre_dbp", "post_map", "post_sbp", "post_dbp"]


def damping_class(r: dict) -> str:
    if not r.get("ringing"):
        return "no_ringing"
    z = r.get("zeta")
    if z is None:
        return "no_ringing"
    if z < 0.2:
        return "underdamped"
    if z < 0.45:
        return "low_damping"
    if z < 0.7:
        return "moderate_damping"
    return "high_damping"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ringing", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seen, n = set(), {}
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for path in a.ringing:
            with open(path) as src:
                for line in src:
                    r = json.loads(line)
                    c = r["case"]
                    if c in seen or "error" in r:
                        continue
                    seen.add(c)
                    rel = sorted(((x, f) for f in r.get("flushes", []) for x in f["releases"]), key=lambda p: p[0]["t"])
                    prev = None
                    for x, f in rel:
                        cls = damping_class(x)
                        n[cls] = n.get(cls, 0) + 1
                        pre, post = f.get("pre") or [None] * 3, f.get("post") or [None] * 3
                        d_fn = since = None
                        if x.get("fn_hz") and prev is not None:
                            d_fn = round(x["fn_hz"] - prev["fn_hz"], 1)
                            since = round((x["t"] - prev["t"]) / 60, 1)
                        w.writerow({"case": c, "label": f"art_flush_response:{cls}", "start_s": x["t"],
                                    "end_s": round(x["t"] + 0.3, 3), "zeta": x.get("zeta"), "fn_hz": x.get("fn_hz"),
                                    "fd_hz": x.get("fd_hz"), "fall_ms": x.get("fall_ms"), "plateau": x.get("plateau"),
                                    "settled": x.get("settled"), "undershoot": x.get("undershoot"),
                                    "overshoot": x.get("overshoot"), "d_fn_hz": d_fn, "minutes_since_prev": since,
                                    "access_start_s": f["start"], "access_recovered_s": f["recovered"],
                                    "access_kinds": f["kinds"], "pre_map": pre[0], "pre_sbp": pre[1], "pre_dbp": pre[2],
                                    "post_map": post[0], "post_sbp": post[1], "post_dbp": post[2]})
                        if x.get("fn_hz"):
                            prev = x
    print(n)


if __name__ == "__main__":
    main()
