# SPDX-License-Identifier: MIT
"""Is the Primus AWP lead real against true time, or only against the CVP reference? A bound from causality.

    bash tools/run_tool_scan.sh primus_anchor_check          (writes $W/primus_anchor_check/part_*.jsonl)
    python tools/primus_anchor_check.py agg $W/primus_anchor_check/part_*.jsonl --work $W --out $W/primus_anchor_check

The Primus numerics are stamped by the recorder PC when they arrive (primus-awp-conversion, primus_numeric_clock),
so a PIP value cannot carry a stamp before the breath it describes has happened. If the AWP stamps sit early by L
against true time, the delay from a breath on the AWP stamps to the PIP stamp that reports it is at least L. The
shortest such delay, D_min, is therefore an upper bound on L.

Per PIP record at PC stamp t: the breaths whose peak (on the AWP stamps) lies in [t - 15 s, t]; the record is used
only if exactly one of them has a peak within 0.6 mbar of its value (PIP is whole mbar), so the breath it reports is
known; its delay D = t - that peak. Per stretch between AWP pauses (records after the first 20 s): the number of
such records and their delays' 2nd, 10th and 50th percentiles; D_min is the 2nd percentile (robust to a few wrong
matches).

A first version chose, per 30-minute block, the delay at which the share of matched PIP records rises; peak
pressure varies too little between neighbouring breaths for that share to step (it stays within about 15% of its
best down to half a second shorter), so the delays it found were not informative (null result, test on six
recordings, work/primus_anchor_check/test/). Records with one matching breath avoid that.

agg joins each stretch with its AWP offset against CVP, each stretch on its right breath
($W/primus_awp_anchor/breaths.json line, $W/primus_breath_rule/stretch_k.json): lag_cvp = CVP breath time minus AWP
stamp at the mid-time of the stretch's PIP records. It reports (1) D_min minus lag_cvp: if the AWP lead against CVP were all real, D_min
could not fall below lag_cvp; (2) within recordings with several stretches, the slope of D_min on lag_cvp: about 1
if the stretch-to-stretch offset changes are in the AWP stamps (the PC-stamped PIP does not move with them).

numerics (the bound from the release-wide PIP fits, no recordings read):
    python tools/primus_anchor_check.py numerics --work $W --out $W/primus_anchor_check
The numeric derivation (tools/numeric_derivation_pilot.py, release-wide outputs $W/overnight/numeric/part_*.jsonl;
atlas study numerics-derivation-release) fits per recording the delay d from the end of a breath's inspiration
(the pressure back below half its peak) on the AWP stamps to the PIP stamp, for the rule "peak of the last breath".
As the PIP cannot be computed before its breath happened, d = d_true - b with d_true >= 0, where b is the AWP bias
against true time, so the AWP cannot be early by more than d (give or take the few tenths of a second from the peak
to the end of inspiration). Joined with each recording's AWP lag against CVP, each stretch on its right breath
($W/primus_numbers/per_recording.json): d - lag_cvp = d_true - (the CVP reference's own delay + the airway-to-CVP
physiology); and across recordings the slope of d on lag_cvp is about 1 if the offset differences are in the AWP
stamps. The PIP fit maps the AWP stamps to recorder time with a drift from each stretch's start (about +0.37 s
an hour); the CVP lag is a median over windows on the AWP stamps; the two differ by a few tenths of a second.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pat import sample_times  # noqa: E402
from vent_sync_pilot import breath_onsets  # noqa: E402

D = np.arange(0.0, 12.0001, 0.05)
TOL, MIN_REC, MIN_UNIQUE = 0.6, 60, 30


def unique_delays(pt, pv, peak_t, peak_v, window=15.0):
    out = []
    for t, v in zip(pt, pv):
        a, b = np.searchsorted(peak_t, t - window), np.searchsorted(peak_t, t, side="right")
        if b - a < 2:
            continue
        m = np.flatnonzero(np.abs(peak_v[a:b] - v) <= TOL)
        if m.size == 1:
            out.append(float(t - peak_t[a + m[0]]))
    return np.array(out)


def scan_case(root, c):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names={"Primus/AWP", "Primus/PIP_MBAR"})
    aw, pip = v.tracks.get("Primus/AWP"), v.tracks.get("Primus/PIP_MBAR")
    if aw is None or pip is None or not aw.record_times.size or not pip.record_times.size:
        return {"case": c, "skip": "no AWP or PIP"}
    ta = sample_times(aw)
    xa = np.nan_to_num(aw.physical().astype(np.float64)[: ta.size])
    on = breath_onsets(ta, xa, float(aw.srate))
    if on.size < 200:
        return {"case": c, "skip": "too few breaths"}
    idx = np.searchsorted(ta, on)
    pk_i = [a + int(np.argmax(xa[a:b])) if b > a else -1 for a, b in zip(idx[:-1], idx[1:])]
    ok = np.array([i >= 0 for i in pk_i]) & (np.diff(on) < 12)
    peak_t = np.array([ta[i] for i, o in zip(pk_i, ok) if o]); peak_v = np.array([xa[i] for i, o in zip(pk_i, ok) if o])
    pt = np.asarray(pip.record_times, float)
    pv = np.asarray(pip.physical(), float)[: pt.size]
    m = np.isfinite(pv) & (pv >= 3) & (pv <= 80) & (pt > peak_t[0] + 15) & (pt < peak_t[-1])
    pt, pv = pt[m], pv[m]
    z = float(ta[0])
    pauses = ta[1:][np.diff(ta) > 5]
    edges = np.concatenate([[-np.inf], pauses, [np.inf]])
    rows = []
    for si in range(edges.size - 1):
        s_ = (pt > edges[si] + 20) & (pt < edges[si + 1])
        if s_.sum() < MIN_REC:
            continue
        P, V = pt[s_], pv[s_]
        Dl = unique_delays(P, V, peak_t, peak_v)
        if Dl.size >= MIN_UNIQUE:
            p2, p10, p50 = np.percentile(Dl, [2, 10, 50])
            rows.append({"stretch": si, "t_mid": round(float(np.median(P) - z), 1), "n_records": int(P.size),
                         "n_unique": int(Dl.size), "d_p2": round(float(p2), 3), "d_p10": round(float(p10), 3),
                         "d_p50": round(float(p50), 3)})
    return {"case": c, "z": z, "pauses": (pauses - z).round(1).tolist(), "blocks": rows}


def scan(a):
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            if a.time_budget and time.time() - t0 > a.time_budget:
                break
            t1 = time.time()
            try:
                r = scan_case(a.root, c)
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 1)
            fh.write(json.dumps(r) + "\n"); fh.flush()


def agg(a):
    W = Path(a.work)
    B = {r["case"]: r for r in json.loads((W / "primus_awp_anchor" / "breaths.json").read_text())["stretches"] if r["arm"] == "cvp"}
    K = {(x["case"], x["stretch"]): x for x in json.loads((W / "primus_breath_rule" / "stretch_k.json").read_text()) if x["arm"] == "cvp"}
    R = [json.loads(l) for p in a.parts for l in open(p, encoding="utf-8") if l.strip()]
    rows = []
    for r in R:
        rec = B.get(r["case"])
        if not rec or not r.get("blocks"):
            continue
        st = {s_["i"]: s_ for s_ in rec["stretches"]}
        for b in r["blocks"]:
            s_ = st.get(b["stretch"]); k = K.get((r["case"], b["stretch"]))
            if not s_ or not k or not s_.get("T"):
                continue
            lag = s_["at_start"] + s_["slope_ppm"] * 1e-6 * (b["t_mid"] - s_["start"]) - k["k"] * s_["T"]
            rows.append({"case": r["case"], "stretch": b["stretch"], "step": k["step"], "d_min": b["d_p2"],
                         "d_p10": b["d_p10"], "n_unique": b["n_unique"], "lag_cvp": round(lag, 3),
                         "diff": round(b["d_p2"] - lag, 3)})
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [5, 10, 25, 50, 75, 90, 95])] if len(v) else None
    d = np.array([x["diff"] for x in rows])
    own = np.array([x["diff"] for x in rows if x["step"] == 1])
    # within-recording slope of d_min on lag_cvp, over recordings with stretches that differ in lag by > 0.5 s
    sl = []
    from collections import defaultdict
    by = defaultdict(list)
    for x in rows:
        by[x["case"]].append(x)
    for c, xs in by.items():
        L = np.array([x["lag_cvp"] for x in xs]); Dm = np.array([x["d_min"] for x in xs])
        if len(xs) >= 3 and np.ptp(L) > 0.5:
            sl.append(float(np.polyfit(L, Dm, 1)[0]))
    summ = {"recordings_read": len(R), "stretches": len(rows), "recordings": len(by),
            "d_min_s_p5_10_25_50_75_90_95": q([x["d_min"] for x in rows]),
            "lag_cvp_s_p5_10_25_50_75_90_95": q([x["lag_cvp"] for x in rows]),
            "d_min_minus_lag_cvp_s_p5_10_25_50_75_90_95": q(d),
            "share_d_min_below_lag_cvp": round(float(np.mean(d < 0)), 3) if d.size else None,
            "own_pause_stretches": {"n": int(own.size), "d_min_minus_lag_cvp_p5_25_50_75_95": [round(float(x), 3) for x in np.percentile(own, [5, 25, 50, 75, 95])] if own.size else None},
            "within_recording_slope_d_min_on_lag_cvp": {"recordings": len(sl), "p10_25_50_75_90": [round(float(x), 3) for x in np.percentile(sl, [10, 25, 50, 75, 90])] if sl else None}}
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "summary.json").write_text(json.dumps(summ, indent=1)); (o / "blocks.json").write_text(json.dumps(rows))
    print(json.dumps(summ, indent=1))


def numerics(a):
    W = Path(a.work)
    lag = {}
    pr = json.loads((W / "primus_numbers" / "per_recording.json").read_text())
    for c, l in zip(pr["case"], pr["lag_median_s"]):
        lag[c] = l
    rows = []
    import glob as _g
    for p in sorted(_g.glob(str(W / "overnight" / "numeric" / "part_*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            t = (r.get("tracks") or {}).get("PIP")
            if not t or not t.get("best") or r["case"] not in lag:
                continue
            b = t["best"]
            rows.append({"case": r["case"], "cand": b["cand"], "d": b["d"], "lag_cvp": lag[r["case"]]})
    last = [x for x in rows if x["cand"] == "mean|n|1"]
    d = np.array([x["d"] for x in last]); L = np.array([x["lag_cvp"] for x in last])
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [5, 10, 25, 50, 75, 90, 95])] if len(v) else None
    slope = np.polyfit(L, d, 1) if d.size > 10 else [np.nan, np.nan]
    rs = float(np.corrcoef(L, d)[0, 1]) if d.size > 10 else np.nan
    from scipy.stats import theilslopes, spearmanr
    ts = theilslopes(d, L) if d.size > 10 else (np.nan,) * 4
    summ = {"recordings_with_both": len(rows), "with_rule_last_breath": len(last),
            "d_s_p5_10_25_50_75_90_95": q(d), "lag_cvp_s_p5_10_25_50_75_90_95": q(L),
            "d_minus_lag_cvp_s_p5_10_25_50_75_90_95": q(d - L),
            "share_d_below_lag_cvp": round(float(np.mean(d < L)), 3) if d.size else None,
            "slope_d_on_lag_cvp_theil": [round(float(x), 3) for x in ts[:2]] + [round(float(ts[2]), 3), round(float(ts[3]), 3)],
            "slope_ols_and_intercept": [round(float(x), 3) for x in slope], "pearson_r": round(rs, 3),
            "spearman_rho": round(float(spearmanr(L, d)[0]), 3) if d.size > 10 else None}
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "numerics_summary.json").write_text(json.dumps(summ, indent=1)); (o / "numerics.json").write_text(json.dumps(rows))
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("--root", required=True); s.add_argument("--cases", required=True)
    s.add_argument("--out", required=True); s.add_argument("--time-budget", type=float, default=0)
    g = sp.add_parser("agg"); g.add_argument("parts", nargs="+"); g.add_argument("--work", required=True); g.add_argument("--out", required=True)
    n = sp.add_parser("numerics"); n.add_argument("--work", required=True); n.add_argument("--out", required=True)
    a = ap.parse_args()
    {"scan": scan, "agg": agg, "numerics": numerics}[a.cmd](a)


if __name__ == "__main__":
    main()
