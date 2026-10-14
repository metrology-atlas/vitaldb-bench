# SPDX-License-Identifier: MIT
"""Trends leading up to arterial-line flushes and zeroings, event by event. This machine only; aggregates and
figures leave.

    python tools/art_pre_event_trends.py --minutes artmin.jsonl --nibp nibp_art.jsonl --jumps zero_w20.jsonl \
        --out events.jsonl --fig fig.png

For every access with a flush ("high") or an open-to-air plateau (zero), the pre-event window runs from the
previous access's recovery (at most 60 min back) to the event; >= 20 clean ART minutes are needed. Per event:
- ART pulse pressure (SBP - DBP) and MAP slopes over the window (per hour), from per-minute values;
- the cuff readings in the window: line minus cuff MAP (corrected for the NIBP pressure-level bias) and the
  line/cuff pulse-pressure ratio, with their slopes when there are >= 3 readings; the first readings in the
  30 min after the event;
- the jump at the event: 20 s before/after (art_zero_jumps) and last/first clean minute.
The same is computed for pseudo-events at random times (same window rules) as a control: a falling pulse
pressure followed by a rise happens by mean reversion alone, so only the excess over the control counts.
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
    ap.add_argument("--max-pre-min", type=float, default=60)
    ap.add_argument("--min-minutes", type=int, default=20)
    a = ap.parse_args()
    M = {json.loads(l)["case"]: json.loads(l) for l in open(a.minutes)}
    N = {json.loads(l)["case"]: json.loads(l) for l in open(a.nibp)}
    J = {json.loads(l)["case"]: json.loads(l) for l in open(a.jumps)}
    # NIBP level bias over all readings
    allp = []
    cuff = {}
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
        rows = [(T[i], r["art"][i][0] - V[i, 2], (r["art"][i][1] - r["art"][i][2]) / max(V[i, 0] - V[i, 1], 5), V[i, 2]) for i in keep]
        cuff[c] = np.array(rows, float) if rows else np.empty((0, 4))
        allp += rows
    allp = np.array(allp)
    b = np.polyfit(allp[:, 3], allp[:, 1], 1)
    for c in cuff:
        if cuff[c].size:
            cuff[c][:, 1] = cuff[c][:, 1] - np.polyval(b, cuff[c][:, 3])
    rng = np.random.default_rng(0)
    out = []
    for c, r in M.items():
        if "error" in r or len(r.get("minutes", [])) < 60:
            continue
        s = np.array(r["minutes"], float)
        tm, mp, pp = s[:, 0] + 30, s[:, 1], s[:, 2] - s[:, 3]
        acc = sorted(r["accesses"])
        # 20 s jumps: art_zero_jumps accesses in the same order (same detector), matched on start_rel
        jz = sorted(J.get(c, {}).get("accesses", []), key=lambda q: q["start_rel"])
        off = None
        if jz and acc:
            cands = [x[0] - jz[0]["start_rel"] for x in acc]
            for o in cands:
                if sum(any(abs(x[0] - (q["start_rel"] + o)) < 1.5 for x in acc) for q in jz) >= 0.9 * len(jz):
                    off = o
                    break
        jmap = {}
        if off is not None:
            for q in jz:
                jmap[round(q["start_rel"] + off)] = q
        def window(t_ev, prev_end):
            lo = max(prev_end, t_ev - a.max_pre_min * 60)
            m = (tm >= lo) & (tm <= t_ev)
            if m.sum() < a.min_minutes:
                return None
            h = (tm[m] - t_ev) / 3600
            row = {"pre_min": round((t_ev - lo) / 60, 1), "n_min": int(m.sum()),
                   "pp_slope_h": round(float(np.polyfit(h, pp[m], 1)[0]), 2),
                   "pp_slope_pct_h": round(float(np.polyfit(h, pp[m], 1)[0] / np.mean(pp[m]) * 100), 1),
                   "map_slope_h": round(float(np.polyfit(h, mp[m], 1)[0]), 2), "pp_pre": round(float(np.mean(pp[m][-5:])), 1)}
            cu = cuff.get(c, np.empty((0, 4)))
            k = (cu[:, 0] >= lo) & (cu[:, 0] <= t_ev) if cu.size else np.zeros(0, bool)
            row["cuff_n"] = int(k.sum())
            if k.sum() >= 1:
                row["cuff_dmap_last"] = round(float(cu[k][-1, 1]), 1)
                row["cuff_ppr_last"] = round(float(cu[k][-1, 2]), 2)
            if k.sum() >= 3:
                hh = (cu[k, 0] - t_ev) / 3600
                row["cuff_absdmap_slope_h"] = round(float(np.polyfit(hh, np.abs(cu[k, 1]), 1)[0]), 2)
                row["cuff_ppr_slope_h"] = round(float(np.polyfit(hh, cu[k, 2], 1)[0]), 3)
            return row
        def after(t_end, t_ev):
            m_b = (tm <= t_ev)
            m_a = (tm >= t_end)
            if not m_b.any() or not m_a.any():
                return {}
            ib, ia = np.flatnonzero(m_b)[-1], np.flatnonzero(m_a)[0]
            if t_ev - tm[ib] > 180 or tm[ia] - t_end > 180:
                return {}
            res = {"d_pp_1min": round(float(pp[ia] - pp[ib]), 1), "d_map_1min": round(float(mp[ia] - mp[ib]), 1)}
            cu = cuff.get(c, np.empty((0, 4)))
            k = (cu[:, 0] > t_end) & (cu[:, 0] <= t_end + 1800) if cu.size else np.zeros(0, bool)
            if k.any():
                res["cuff_dmap_after"] = round(float(cu[k][0, 1]), 1)
                res["cuff_ppr_after"] = round(float(cu[k][0, 2]), 2)
            return res
        prev_end = tm[0] - 1
        for x in acc:
            st, en, rec, kinds = x
            kind = "zero" if "low" in kinds else ("flush" if "high" in kinds else None)
            if kind:
                w = window(st, prev_end)
                if w:
                    w.update({"case": c, "kind": kind, "t": st, **after(rec, st)})
                    q = jmap.get(round(st))
                    if q and q.get("delta"):
                        d = q["delta"]
                        w.update({"d_pp_20s": round(d["sbp"] - d["dbp"], 1), "d_dbp_20s": d["dbp"], "d_map_20s": d["map"]})
                    out.append(w)
            prev_end = rec
        # pseudo-events
        ends = np.array([x[2] for x in acc]) if acc else np.empty(0)
        starts = np.array([x[0] for x in acc]) if acc else np.empty(0)
        for _ in range(max(len(acc), 5)):
            t_ev = rng.uniform(tm[0] + 1200, tm[-1] - 600)
            if starts.size and np.any((starts <= t_ev + 240) & (ends >= t_ev - 60)):
                continue
            pe = ends[ends < t_ev].max() if (ends < t_ev).any() else tm[0] - 1
            w = window(t_ev, pe)
            if w:
                w.update({"case": c, "kind": "random", "t": t_ev, **after(t_ev + 60, t_ev)})
                out.append(w)
    with open(a.out, "w") as fh:
        for w in out:
            fh.write(json.dumps(w) + "\n")
    print(json.dumps({"events": len(out), "bias": [round(b[1], 2), round(b[0], 3)]}))


if __name__ == "__main__":
    main()
