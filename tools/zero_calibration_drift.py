# SPDX-License-Identifier: MIT
"""Slow calibration drift corrected at zeroing: the line-minus-cuff MAP difference between zeroings. This
machine only; aggregates leave.

    python tools/zero_calibration_drift.py --minutes artmin.jsonl --nibp nibp_art.jsonl --jumps zero_w20.jsonl \
        --out out.jsonl [--max-span-h 3]

d = ART MAP (35 s before each cuff result) - cuff MAP, corrected for the NIBP pressure-level bias. For each line
access (zero = open-to-air plateau; flush = "high" without it) and for random times, the cuff readings since the
previous access of the same kind (at most --max-span-h before) and until the next one (at most as long after):
- pre: number of readings, slope of d (mmHg/h), d fitted at the event (or the last reading);
- post: median of d;
- step = post - pre-at-event;
- the ART MAP jump at the event (20 s windows from art_zero_jumps; 1-min windows for all kinds).
A recalibration moves the line but not the cuff, so d steps by the line's own jump; a physiological change moves
both and leaves d unchanged. A slow calibration drift shows as a pre-event slope that the step undoes.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", required=True)
    ap.add_argument("--nibp", required=True)
    ap.add_argument("--jumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-span-h", type=float, default=3.0)
    a = ap.parse_args()
    M = {json.loads(l)["case"]: json.loads(l) for l in open(a.minutes)}
    N = {json.loads(l)["case"]: json.loads(l) for l in open(a.nibp)}
    J = {json.loads(l)["case"]: json.loads(l) for l in open(a.jumps)}
    cuff, allp = {}, []
    for c, r in N.items():
        if not r.get("art"):
            continue
        T = np.array(r["t"]) + r["first"]
        V = np.array(r["v"], float)
        keep = []
        for i, x in enumerate(r["art"]):
            if x is None:
                continue
            if keep and T[i] - T[keep[-1]] <= 10:
                keep[-1] = i
            else:
                keep.append(i)
        rows = [(T[i], r["art"][i][0] - V[i, 2], V[i, 2]) for i in keep]
        if rows:
            cuff[c] = np.array(rows, float)
            allp += rows
    allp = np.array(allp)
    b = np.polyfit(allp[:, 2], allp[:, 1], 1)
    for c in cuff:
        cuff[c][:, 1] -= np.polyval(b, cuff[c][:, 2])
    rng = np.random.default_rng(0)
    span = a.max_span_h * 3600
    out = []
    for c, r in M.items():
        if "error" in r or c not in cuff or len(r.get("minutes", [])) < 30:
            continue
        cu = cuff[c]
        s = np.array(r["minutes"], float)
        tm, mp = s[:, 0] + 30, s[:, 1]
        acc = sorted(r["accesses"])
        jz = sorted(J.get(c, {}).get("accesses", []), key=lambda q: q["start_rel"])
        off, jmap = None, {}
        if jz and acc:
            for o in [x[0] - jz[0]["start_rel"] for x in acc]:
                if sum(any(abs(x[0] - (q["start_rel"] + o)) < 1.5 for x in acc) for q in jz) >= 0.9 * len(jz):
                    off = o
                    break
        if off is not None:
            jmap = {round(q["start_rel"] + off): q for q in jz}
        kinds = {"zero": [x for x in acc if "low" in x[3]], "flush": [x for x in acc if "high" in x[3] and "low" not in x[3]]}
        rnd = []
        for _ in range(8):
            t = rng.uniform(tm[0] + 600, tm[-1] - 600)
            if not any(x[0] - 300 <= t <= x[2] + 300 for x in acc):
                rnd.append([t, t, t + 60, "random"])
        kinds["random"] = sorted(rnd)
        for kind, evs in kinds.items():
            for i, x in enumerate(evs):
                st, rec = x[0], x[2]
                lo = max(evs[i - 1][2] if i else -1e18, st - span)
                hi = min(evs[i + 1][0] if i + 1 < len(evs) else 1e18, rec + span)
                pre = cu[(cu[:, 0] > lo) & (cu[:, 0] < st)]
                post = cu[(cu[:, 0] > rec) & (cu[:, 0] < hi)]
                if len(pre) < 1 or len(post) < 1:
                    continue
                row = {"case": c, "kind": kind, "t": st, "n_pre": len(pre), "n_post": len(post),
                       "pre_span_min": round((st - pre[0, 0]) / 60, 1), "post_med": round(float(np.median(post[:, 1])), 2),
                       "pre_last": round(float(pre[-1, 1]), 2), "pre_med": round(float(np.median(pre[:, 1])), 2)}
                if len(pre) >= 3 and pre[-1, 0] - pre[0, 0] >= 1200:
                    k = np.polyfit((pre[:, 0] - st) / 3600, pre[:, 1], 1)
                    row["pre_slope_h"] = round(float(k[0]), 2)
                    row["pre_at_event"] = round(float(k[1]), 2)
                    row["pre_resid_sd"] = round(float(np.std(pre[:, 1] - np.polyval(k, (pre[:, 0] - st) / 3600))), 2)
                ref = row.get("pre_at_event", row["pre_med"])
                row["step"] = round(row["post_med"] - ref, 2)
                ib = np.flatnonzero(tm <= st)
                ia = np.flatnonzero(tm >= rec)
                if ib.size and ia.size and st - tm[ib[-1]] <= 180 and tm[ia[0]] - rec <= 180:
                    row["art_dmap_1min"] = round(float(mp[ia[0]] - mp[ib[-1]]), 2)
                q = jmap.get(round(st))
                if q and q.get("delta"):
                    row["art_dmap_20s"] = q["delta"]["map"]
                    row["art_ddbp_20s"] = q["delta"]["dbp"]
                out.append(row)
    with open(a.out, "w") as fh:
        for w in out:
            fh.write(json.dumps(w) + "\n")
    print(json.dumps({"events": len(out)}))


if __name__ == "__main__":
    main()
