# SPDX-License-Identifier: MIT
"""Where the Primus AWP offset comes from: per-stretch offsets, slopes and pause steps, and the breath period.

    python tools/primus_awp_anchor.py stretches --parts $W/primus_sync_accuracy/part_*.jsonl --out $W/primus_awp_anchor
    python tools/primus_awp_anchor.py rr --root R --parts $W/primus_sync_accuracy/part_*.jsonl --out $W/primus_awp_anchor [--time-budget 140]

Background. Within each stretch between AWP pauses (stamp gaps over 5 s) the AWP records are exactly 0.48 s apart
(30 samples at 62.5 Hz): the recorder sets an anchor at the start of the stretch and counts samples from it, so the
stretch's offset is the anchor's error. The lag here is tools/primus_sync_accuracy.py's: reference (SNUADC CVP, else
the low-passed ART) minus the AWP stamp of the same breath, in 10-minute windows; bias = -lag.

stretches: per recording (CVP arm, else ART), a line of lag against time in each stretch with at least 3 windows: its
offset at the stretch start, slope (ppm) and residual SD; the step at each pause (the next stretch's start offset
minus the previous line extrapolated). Writes stretches.json and stretches_summary.json.

rr: per recording with a measured arm, the ventilator's set and measured breath rates (Primus/SET_RR_IPPV,
Primus/RR_CO2), every 4th record, so that offsets and steps can be compared with the breath period. The lag is found
within half a breath period of a coarse envelope estimate and each window starts from the previous one, so a whole
stretch could sit a whole breath period off. Resumable; stops after --time-budget seconds (rerun to continue).
Writes rr.jsonl.

breaths: joins stretches.json with rr.jsonl. Per stretch, the breath period T = 60 / the median set rate
(SET_RR_IPPV; else the measured RR_CO2) over the stretch; per pause, T over the stretch after it. Reports the
offsets and steps in breath periods: if whole stretches sit k breaths off, offsets cluster at a base plus whole T
and large steps sit near whole multiples of T; a control shifts each step's T by an unrelated stretch's T.
Writes breaths.json and breaths_summary.json.

    python tools/primus_awp_anchor.py breaths --out $W/primus_awp_anchor

arms: validation with two independent references. In recordings with both a CVP and an ART arm, per stretch the
median of (ART lag - CVP lag) over matched windows (same time, within 1 s), with the stretch's breath period T. A
slip in one arm shows as a difference of about a whole T on top of the physiological difference between the
references; the distribution of the difference in breath periods, by set rate, says how often. Writes
arms_summary.json and arms.json.

    python tools/primus_awp_anchor.py arms --parts $W/primus_sync_accuracy/part_*.jsonl --out $W/primus_awp_anchor

gapsteps: joins each pause step (breaths.json) with its AWP gap (records/part_*.jsonl, matched by time within
2 s): is the gap a whole number of records (the 0.48 s grid carries on) or not (a new anchor)? Steps by gap kind,
in seconds and in breath periods. Writes gapsteps_summary.json and gapsteps.json (one row per step).

    python tools/primus_awp_anchor.py gapsteps --out $W/primus_awp_anchor [--records $W/primus_awp_anchor/records]

records: per recording with Primus/AWP (and Primus/CO2 for comparison), how its record times are made: the share of
consecutive records exactly one record length apart (30 samples at the track's rate, within 1 ms), the record
length, and each gap (stamp step over 5 s) with its length, its length in record lengths, and the time of the last
record before it and the first after it (relative to the first AWP record). Also the first record of every Primus
track and of SNUADC/ECG_II, relative to the first AWP record. Writes records.jsonl and records_summary.json.

    OUT=$W/primus_awp_anchor/records bash tools/run_tool_scan.sh primus_awp_anchor     ("scan" = records, 8 parts)
    python tools/primus_awp_anchor.py records-agg $W/primus_awp_anchor/records/part_*.jsonl --out $W/primus_awp_anchor
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _arm(r):
    for k in ("cvp", "art"):
        a = r.get("arms", {}).get(k)
        if a and len(a["t"]) >= 3:
            return k, a
    return None, None


def stretches(a):
    rows = []
    for p in a.parts:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            k, arm = _arm(r)
            if arm is None:
                continue
            t, y, P = np.array(arm["t"]), np.array(arm["lag"]), np.array(r["pauses"])
            edges = np.concatenate([[-1e9], P, [1e9]])
            st = []
            for i in range(len(edges) - 1):
                m = (t > edges[i]) & (t < edges[i + 1])
                if m.sum() >= 3:
                    b, c = np.polyfit(t[m], y[m], 1)
                    t0 = max(edges[i], 0.0)
                    st.append({"i": i, "start": float(t0), "n": int(m.sum()), "slope_ppm": float(b * 1e6),
                               "at_start": float(c + b * t0), "resid_sd": float(np.std(y[m] - (c + b * t[m]))),
                               "t": t[m].tolist(), "lag": y[m].tolist()})
            if st:
                rows.append({"case": r["case"], "arm": k, "z": r["z"], "pauses": P.tolist(), "stretches": st})
    q = lambda v: [round(float(x), 4) for x in np.percentile(v, [5, 10, 25, 50, 75, 90, 95])] if len(v) else None
    s0 = [r["stretches"][0]["at_start"] for r in rows]
    sl = [s["slope_ppm"] for r in rows for s in r["stretches"] if s["n"] >= 6]
    steps = [b["at_start"] - (x["at_start"] + x["slope_ppm"] * 1e-6 * (b["start"] - x["start"]))
             for r in rows for x, b in zip(r["stretches"], r["stretches"][1:])]
    summ = {"recordings": len(rows), "arms": {k: sum(r["arm"] == k for r in rows) for k in ("cvp", "art")},
            "first_stretch_lag_at_start_s_p5_10_25_50_75_90_95": q(s0),
            "slope_ppm_p5_10_25_50_75_90_95": q(sl), "slope_n": len(sl),
            "pause_step_s_p5_10_25_50_75_90_95": q(steps), "pause_steps": len(steps),
            "share_abs_step_over_0.5_s": round(float(np.mean(np.abs(steps) > 0.5)), 3) if steps else None,
            "share_abs_step_over_1_s": round(float(np.mean(np.abs(steps) > 1)), 3) if steps else None}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "stretches.json").write_text(json.dumps(rows))
    (out / "stretches_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def rr(a):
    from vitaldb_reader.vital_file import read_vital
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = out / "rr.jsonl"
    done = {json.loads(l)["case"] for l in f.read_text().splitlines() if l.strip()} if f.exists() else set()
    t0 = time.time()
    with open(f, "a", encoding="utf-8") as fh:
        for p in a.parts:
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                if _arm(r)[1] is None or r["case"] in done:
                    continue
                if time.time() - t0 > a.time_budget:
                    print("time budget reached; rerun to continue"); return
                v = read_vital(Path(a.root) / "vital_files" / f"{r['case']:04d}.vital",
                               track_names={"Primus/SET_RR_IPPV", "Primus/RR_CO2"})
                res = {"case": r["case"]}
                for k in ("Primus/SET_RR_IPPV", "Primus/RR_CO2"):
                    tr = v.tracks.get(k)
                    if tr is None or not np.asarray(tr.record_times).size:
                        continue
                    t = np.asarray(tr.record_times) - r["z"]
                    x = np.asarray(tr.physical(), float)[: t.size]
                    res[k.split("/")[1]] = {"t": t.round(1).tolist()[::4], "x": x.tolist()[::4]}
                fh.write(json.dumps(res) + "\n"); fh.flush()
    print("done")


PRIMUS_TRACKS = {f"Primus/{n}" for n in ['AWP', 'CO2', 'COMPLIANCE', 'ETCO2', 'EXP_SEVO', 'FEN2O', 'FEO2', 'FIN2O', 'FIO2', 'FLOW_AIR', 'FLOW_N2O', 'FLOW_O2', 'INCO2', 'INSP_SEVO', 'MAC', 'MAWP_MBAR', 'MV', 'PAMB_MBAR', 'PEEP_MBAR', 'PIP_MBAR', 'PPLAT_MBAR', 'RR_CO2', 'SET_AGE', 'SET_FIO2', 'SET_FRESH_FLOW', 'SET_INSP_PAUSE', 'SET_INSP_TM', 'SET_INTER_PEEP', 'SET_PIP', 'SET_RR_IPPV', 'SET_TV_L', 'TV', 'VENT_LEAK']} | {"SNUADC/ECG_II"}


def _period(rr, t0, t1):
    for k in ("SET_RR_IPPV", "RR_CO2"):
        d = rr.get(k)
        if not d:
            continue
        t, x = np.asarray(d["t"], float), np.asarray(d["x"], float)
        m = (t >= t0) & (t < t1) & np.isfinite(x) & (x >= 4) & (x <= 40)
        if m.sum() >= 3:
            return 60.0 / float(np.median(x[m])), k
    return None, None


def _by_rate(off, T):
    """Offsets (s) per set breath rate. If some stretches sit a whole breath off, their mode moves with the breath
    period (one breath above the main mode); a stamp offset that does not depend on the rate stays put. Strata are
    by the rate itself, so the comparison is not biased by selecting on offset / T."""
    out = {}
    rate = np.round(60 / T)
    for r_ in (10, 12, 14, 16, 18):
        m = rate == r_
        if m.sum() < 80:
            continue
        h = np.histogram(off[m], bins=np.arange(-2, 10.01, 0.5))[0]
        out[f"{r_}/min (T {60 / r_:.2f} s)"] = {"n": int(m.sum()),
                                                "offset_s_p10_25_50_75_90": [round(float(x), 2) for x in np.percentile(off[m], [10, 25, 50, 75, 90])],
                                                "hist_0.5s_bins_from_-2": h.tolist()}
    return out


def breaths(a):
    out = Path(a.out)
    S = json.loads((out / "stretches.json").read_text())
    RR = {json.loads(l)["case"]: json.loads(l) for l in (out / "rr.jsonl").read_text().splitlines() if l.strip()}
    rows, steps = [], []
    for r in S:
        rr = RR.get(r["case"])
        if rr is None:
            continue
        st = r["stretches"]
        for i, x in enumerate(st):
            end = st[i + 1]["start"] if i + 1 < len(st) else x["t"][-1] + 300
            T, src = _period(rr, x["start"], end)
            x["T"], x["T_src"] = T, src
        rows.append({"case": r["case"], "arm": r["arm"], "stretches": [
            {k: x[k] for k in ("i", "start", "n", "slope_ppm", "at_start", "resid_sd", "T", "T_src")} for x in st]})
        for x, b in zip(st, st[1:]):
            if b.get("T") and x.get("T"):
                step = b["at_start"] - (x["at_start"] + x["slope_ppm"] * 1e-6 * (b["start"] - x["start"]))
                steps.append({"case": r["case"], "step": step, "T": b["T"], "T_before": x["T"], "arm": r["arm"]})
    rng = np.random.default_rng(1)
    st_all = [x for r in rows for x in r["stretches"] if x["T"]]
    off = np.array([x["at_start"] for x in st_all]); T = np.array([x["T"] for x in st_all])
    sp = np.array([s_["step"] for s_ in steps]); sT = np.array([s_["T"] for s_ in steps])
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [5, 25, 50, 75, 95])] if len(v) else None

    def near_int(k):  # distance of k (in breath periods) from the nearest whole number
        return np.abs(k - np.round(k))
    big = np.abs(sp) > 0.5 * sT * 0.5   # steps over a quarter breath
    ctrl_T = rng.permutation(sT)
    summ = {"stretches_with_T": len(st_all), "T_s_p5_25_50_75_95": q(T),
            "offset_s_p5_25_50_75_95": q(off),
            "offset_over_T_p5_25_50_75_95": q(off / T),
            "pause_steps_with_T": len(sp),
            "steps_over_quarter_breath": int(big.sum()),
            "step_over_T_of_those_p5_25_50_75_95": q(sp[big] / sT[big]),
            "near_whole_T_(within_0.1)_share": round(float(np.mean(near_int(sp[big] / sT[big]) < 0.1)), 3) if big.any() else None,
            "control_near_whole_T_share_(T_permuted)": round(float(np.mean(near_int(sp[big] / ctrl_T[big]) < 0.1)), 3) if big.any() else None,
            "uniform_expectation": 0.2,
            "offset_over_T_histogram_0.25_bins_from_-1": np.histogram(off / T, bins=np.arange(-1, 3.01, 0.25))[0].tolist(),
            "offset_s_histogram_0.5s_bins_from_-2": np.histogram(off, bins=np.arange(-2, 10.01, 0.5))[0].tolist(),
            "offset_by_set_rate": _by_rate(off, T),
            "abs_step_within_0.1T_of_one_breath_share": round(float(np.mean(np.abs(np.abs(sp[big]) / sT[big] - 1) < 0.1)), 3) if big.any() else None}
    (out / "breaths.json").write_text(json.dumps({"stretches": rows, "steps": steps}))
    (out / "breaths_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def arms(a):
    out = Path(a.out)
    RR = {json.loads(l)["case"]: json.loads(l) for l in (out / "rr.jsonl").read_text().splitlines() if l.strip()}
    rows = []
    for p in a.parts:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            A = r.get("arms", {})
            if not (A.get("cvp") and A.get("art") and len(A["cvp"]["t"]) >= 3 and len(A["art"]["t"]) >= 3):
                continue
            tc, lc = np.array(A["cvp"]["t"]), np.array(A["cvp"]["lag"])
            ta, la = np.array(A["art"]["t"]), np.array(A["art"]["lag"])
            edges = np.concatenate([[-1e9], np.array(r["pauses"]), [1e9]])
            for i in range(len(edges) - 1):
                mc = (tc > edges[i]) & (tc < edges[i + 1])
                d = []
                for t_, l_ in zip(tc[mc], lc[mc]):
                    j = np.where(np.abs(ta - t_) <= 1.0)[0]
                    if j.size:
                        d.append(la[j[0]] - l_)
                if len(d) < 3:
                    continue
                T, _ = _period(RR.get(r["case"], {}), max(edges[i], 0.0), min(edges[i + 1], tc[-1] + 300))
                if T:
                    rows.append({"case": r["case"], "stretch": i, "n": len(d), "diff_s": float(np.median(d)),
                                 "diff_sd_s": float(np.std(d)), "T": T})
    d = np.array([x["diff_s"] for x in rows]); T = np.array([x["T"] for x in rows])
    k = d / T
    summ = {"stretches": len(rows), "recordings": len({x["case"] for x in rows}),
            "diff_s_p5_25_50_75_95": [round(float(x), 3) for x in np.percentile(d, [5, 25, 50, 75, 95])],
            "diff_over_T_histogram_0.25_bins_from_-2.5": np.histogram(k, bins=np.arange(-2.5, 2.51, 0.25))[0].tolist(),
            "share_abs_diff_under_quarter_T": round(float(np.mean(np.abs(k) < 0.25)), 3),
            "share_abs_diff_0.75_to_1.25_T": round(float(np.mean((np.abs(k) >= 0.75) & (np.abs(k) < 1.25))), 3),
            "within_stretch_sd_of_diff_median_s": round(float(np.median([x["diff_sd_s"] for x in rows])), 3),
            "by_set_rate": {}}
    rate = np.round(60 / T)
    for r_ in (10, 12, 14, 16, 18):
        m = rate == r_
        if m.sum() >= 40:
            summ["by_set_rate"][f"{r_}/min"] = {"n": int(m.sum()),
                                                "hist_0.5s_bins_from_-8": np.histogram(d[m], bins=np.arange(-8, 8.01, 0.5))[0].tolist()}
    (out / "arms.json").write_text(json.dumps(rows))
    (out / "arms_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def gapsteps(a):
    out = Path(a.out)
    B = json.loads((out / "breaths.json").read_text())
    G = {}
    for p in sorted(Path(a.records or out / "records").glob("part_*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r.get("awp"):
                G[r["case"]] = r["awp"]["gaps"]
    rows = []
    for r in B["stretches"]:
        gaps = G.get(r["case"])
        if not gaps:
            continue
        st = r["stretches"]
        for x, b in zip(st, st[1:]):
            if not (b.get("T") and x.get("T")):
                continue
            g = [q for q in gaps if abs(q["after"] - b["start"]) <= 2.0]
            if len(g) != 1:
                continue
            g = g[0]
            step = b["at_start"] - (x["at_start"] + x["slope_ppm"] * 1e-6 * (b["start"] - x["start"]))
            whole = abs(g["len_records"] - round(g["len_records"])) < 0.02
            rows.append({"case": r["case"], "step": step, "T": b["T"], "whole": whole, "gap_s": g["len_s"]})
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [5, 10, 25, 50, 75, 90, 95])] if len(v) else None
    summ = {"steps_matched_to_a_gap": len(rows)}
    for name, flag in (("whole_records_(grid_kept)", True), ("not_whole_(new_anchor)", False)):
        m = [x for x in rows if x["whole"] == flag]
        st_ = np.array([x["step"] for x in m]); T = np.array([x["T"] for x in m])
        k = st_ / T if len(m) else np.array([])
        summ[name] = {"n": len(m), "step_s_p5_10_25_50_75_90_95": q(st_),
                      "share_abs_step_over_0.25_s": round(float(np.mean(np.abs(st_) > 0.25)), 3) if len(m) else None,
                      "share_abs_step_0.75_to_1.25_T": round(float(np.mean((np.abs(k) > 0.75) & (np.abs(k) < 1.25))), 3) if len(m) else None,
                      "gap_s_p5_50_95": q(np.array([x["gap_s"] for x in m]))[::3] if len(m) else None}
    (out / "gapsteps_summary.json").write_text(json.dumps(summ, indent=1))
    (out / "gapsteps.json").write_text(json.dumps(rows))
    print(json.dumps(summ, indent=1))


def _rec_track(tr):
    rt = np.asarray(tr.record_times, float)
    if rt.size < 2:
        return None
    n = np.asarray(tr.record_sizes) if hasattr(tr, "record_sizes") else None
    L = 30 / float(tr.srate)
    d = np.diff(rt)
    inner = d[d <= 5]
    g = np.where(d > 5)[0]
    return {"records": int(rt.size), "record_s": L,
            "samples_per_record_mode": int(np.bincount(np.asarray(n, int)).argmax()) if n is not None and len(n) else None,
            "share_exact": float(np.mean(np.abs(inner - L) < 1e-3)) if inner.size else None,
            "spacing_p1_50_99": [float(x) for x in np.percentile(inner, [1, 50, 99])] if inner.size else None,
            "gaps": [{"before": float(rt[i] - rt[0]), "after": float(rt[i + 1] - rt[0]), "len_s": float(d[i]),
                      "len_records": float(d[i] / L)} for i in g]}


def records(a):
    from vitaldb_reader.vital_file import read_vital
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    with open(out, "a", encoding="utf-8") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            try:
                v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names=PRIMUS_TRACKS)
                aw = v.tracks.get("Primus/AWP")
                if aw is None or np.asarray(aw.record_times).size < 2:
                    res = {"case": c, "skip": "no AWP"}
                else:
                    z = float(np.asarray(aw.record_times)[0])
                    res = {"case": c, "z": z, "awp": _rec_track(aw)}
                    co2 = v.tracks.get("Primus/CO2")
                    if co2 is not None and np.asarray(co2.record_times).size > 1:
                        res["co2"] = _rec_track(co2)
                    res["first_record"] = {k: float(np.asarray(t.record_times)[0] - z) for k, t in v.tracks.items()
                                           if (k.startswith("Primus/") or k == "SNUADC/ECG_II") and np.asarray(t.record_times).size}
            except Exception as e:  # noqa: BLE001
                res = {"case": c, "error": repr(e)}
            fh.write(json.dumps(res) + "\n"); fh.flush()


def records_agg(a):
    R = [json.loads(l) for p in a.parts for l in open(p, encoding="utf-8") if l.strip()]
    ok = [r for r in R if "awp" in r and r["awp"]]
    q = lambda v: [round(float(x), 4) for x in np.percentile(v, [5, 25, 50, 75, 95])] if len(v) else None
    ex = [r["awp"]["share_exact"] for r in ok if r["awp"]["share_exact"] is not None]
    gl = [g["len_records"] for r in ok for g in r["awp"]["gaps"]]
    frac = [x - round(x) for x in gl]
    co2same = [r for r in ok if "co2" in r and r["co2"]]
    def gaps_match(r):
        a_ = [round(g["after"], 2) for g in r["awp"]["gaps"]]; b_ = [round(g["after"], 2) for g in r["co2"]["gaps"]]
        return a_ == b_
    summ = {"recordings_read": len(R), "with_awp": len(ok), "errors": sum("error" in r for r in R),
            "awp_share_exact_spacing_p5_25_50_75_95": q(ex),
            "recordings_all_exact_(>=0.999)": int(sum(x >= 0.999 for x in ex)),
            "awp_gaps": len(gl),
            "gap_len_records_minus_nearest_integer_abs_p25_50_75": [round(float(x), 3) for x in np.percentile(np.abs(frac), [25, 50, 75])] if gl else None,
            "share_gaps_whole_records_(within_0.02)": round(float(np.mean(np.abs(frac) < 0.02)), 3) if gl else None,
            "with_co2": len(co2same), "co2_gaps_same_as_awp": int(sum(gaps_match(r) for r in co2same))}
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "records_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("stretches"); s.add_argument("--parts", nargs="+", required=True); s.add_argument("--out", required=True)
    r = sp.add_parser("rr"); r.add_argument("--root", required=True); r.add_argument("--parts", nargs="+", required=True)
    r.add_argument("--out", required=True); r.add_argument("--time-budget", type=float, default=140)
    sc = sp.add_parser("scan"); sc.add_argument("--root", required=True); sc.add_argument("--cases", required=True)
    sc.add_argument("--out", required=True)
    m = sp.add_parser("arms"); m.add_argument("--parts", nargs="+", required=True); m.add_argument("--out", required=True)
    gs = sp.add_parser("gapsteps"); gs.add_argument("--out", required=True)
    gs.add_argument("--records", help="folder of records/part_*.jsonl (default OUT/records)")
    b = sp.add_parser("breaths"); b.add_argument("--out", required=True)
    c = sp.add_parser("records"); c.add_argument("--root", required=True); c.add_argument("--cases", required=True)
    c.add_argument("--out", required=True)
    g = sp.add_parser("records-agg"); g.add_argument("parts", nargs="+"); g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"stretches": stretches, "rr": rr, "breaths": breaths, "arms": arms, "gapsteps": gapsteps, "records": records, "scan": records, "records-agg": records_agg}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
