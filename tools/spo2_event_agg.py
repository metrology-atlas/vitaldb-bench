# SPDX-License-Identifier: MIT
"""Aggregate tools/spo2_event_pilot.py output (aggregates only).

    python tools/spo2_event_agg.py out1.jsonl [...] --summary summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

GRID = np.arange(-20, 61, 1.0)
DUR_BINS = [(3, 5), (5, 10), (10, 20), (20, 60), (60, 1e9)]
BURST_BINS = [(0, 1), (1, 3), (3, 10), (10, 1e9)]
THR = 2.0   # % SpO2 / bpm: a response is a deviation of at least this from the pre-event median


def pct(x, p=(10, 25, 50, 75, 90)):
    x = np.asarray([v for v in x if v is not None], float)
    return [round(float(v), 2) for v in np.percentile(x, p)] if x.size else None


def ev_summary(evs, key):
    S = np.array([[np.nan if z is None else z for z in e[key]] for e in evs if e.get(key)], float)
    if not S.size:
        return None
    post = S[:, GRID >= 0]
    mabs = np.nanmax(np.abs(np.nan_to_num(post, nan=0)), axis=1)
    resp = mabs >= THR
    drop = np.isnan(post).mean(axis=1) > 0.1
    onset, last_dev = [], []
    for e, row in zip([e for e in evs if e.get(key)], S):
        g = np.flatnonzero((GRID >= 0) & (np.abs(np.nan_to_num(row, nan=0)) >= THR))
        if g.size:
            onset.append(float(GRID[g[0]]))
            last_dev.append(float(GRID[g[-1]] - e["dur"]))
    out = {"n": int(S.shape[0]), "responded": int(resp.sum()), "responded_ge1": int((mabs >= 1).sum()),
           "responded_ge3": int((mabs >= 3).sum()), "dropout": int(drop.sum()),
           "onset_s": pct(onset), "end_after_burst_end_s": pct(last_dev),
           "mean_curve": [None if not np.isfinite(z) else round(float(z), 3) for z in np.nanmean(S, axis=0)],
           "mean_abs_curve": [None if not np.isfinite(z) else round(float(z), 3) for z in np.nanmean(np.abs(S), axis=0)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()
    rows, seen = [], set()
    for f in a.files:
        for l in open(f):
            r = json.loads(l)
            if r["case"] not in seen:
                seen.add(r["case"])
                rows.append(r)
    S = {"cases": len(rows), "errors": [r["error"][:120] for r in rows if "error" in r],
         "skipped": Counter(r["skip"] for r in rows if "skip" in r).most_common()}
    ok = [r for r in rows if "error" not in r and "skip" not in r]
    S["cases_used"] = len(ok)
    # episodes
    E = [e for r in ok for e in r.get("episodes", [])]
    S["episodes"] = {"total": len(E), "recorder_gap": sum(e["recorder_gap"] for e in E),
                     "types": Counter(e["type"] for e in E if not e["recorder_gap"]).most_common()}
    for key in ("spo2", "phr"):
        for typ in ("flat", "low"):
            ee = [e for e in E if not e["recorder_gap"] and e["type"] == typ and key in e]
            if not ee:
                continue
            byd = []
            for lo, hi in DUR_BINS:
                b = [e for e in ee if lo <= e["dur"] < hi]
                byd.append({"dur": [lo, hi if hi < 1e8 else None], "n": len(b),
                            "dropout": sum(e[key]["dropout"] for e in b),
                            "held_const_no_dropout": sum(bool(e[key].get("held_const")) for e in b if not e[key]["dropout"])})
            dd = [e for e in ee if e[key]["dropout"]]
            S["episodes"][f"{key}_{typ}"] = {
                "n": len(ee), "by_duration": byd,
                "hold_s": pct([e[key]["hold"] for e in dd]), "resume_s": pct([e[key]["resume"] for e in dd]),
                "gap_end_s": pct([e[key].get("gap_end") for e in dd]),
                "longest_without_dropout_s": max([e["dur"] for e in ee if not e[key]["dropout"]], default=None),
                "shortest_with_dropout_s": min([e["dur"] for e in dd], default=None)}
    # bursts
    for typ in ("cautery", "pleth_only", "control"):
        evs = [e for r in ok for e in r.get(typ, [])]
        d = {"n": len(evs), "cases": sum(1 for r in ok if r.get(typ))}
        if typ == "cautery":
            d["pleth_hf_rise"] = sum(bool(e.get("pleth_hf_rise")) for e in evs)
        if typ == "pleth_only":
            d["kinds"] = Counter(e.get("kind") for e in evs).most_common()
            for kd in ("hf", "big_amp"):
                sub = [e for e in evs if e.get("kind") == kd]
                d["spo2_" + kd] = ev_summary(sub, "spo2") if sub else None
        for key in ("spo2", "phr"):
            d[key] = ev_summary(evs, key)
            if typ != "control":
                d[key + "_by_dur"] = []
                for lo, hi in BURST_BINS:
                    sub = [e for e in evs if lo <= e["dur"] < hi]
                    s_ = ev_summary(sub, key) if sub else None
                    if s_:
                        s_ = {k: v for k, v in s_.items() if k not in ("mean_curve",)}
                    d[key + "_by_dur"].append({"dur": [lo, hi if hi < 1e8 else None], **(s_ or {"n": 0})})
        S[typ] = d
    S["counts_per_case_median"] = {k: float(np.median([r["counts"][k] for r in ok])) for k in ok[0]["counts"]} if ok else None
    F = [r["pleth_hr_fit"] for r in ok if "pleth_hr_fit" in r]
    if F:
        fam = Counter("|".join(f["best"]["cand"].split("|")[:2]) for f in F)
        S["pleth_hr_fit"] = {"n": len(F), "best_family": fam.most_common(),
                             "windows": sorted(float(f["best"]["cand"].split("|")[2]) for f in F),
                             "delay_min_s": pct([f["best"]["d"] for f in F]),
                             "delay_mid_s": pct([f["best"]["d"] + f["gap_median"] / 2 for f in F]),
                             "sigma": pct([f["best"]["s"] for f in F]),
                             "floor_better": sum(f.get("round_minus_floor_ll", 0) < 0 for f in F),
                             "hold_best_U": Counter(
                                 (max(f["hold"]["U"], key=lambda u: u["delta_bic"])["U"]
                                  if max(u["delta_bic"] for u in f["hold"]["U"]) > 10 else "none") for f in F if "hold" in f
                             ).most_common()}
    json.dump(S, open(a.summary, "w"), indent=1)


if __name__ == "__main__":
    main()
