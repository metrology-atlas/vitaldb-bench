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
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
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
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
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
    c = sp.add_parser("records"); c.add_argument("--root", required=True); c.add_argument("--cases", required=True)
    c.add_argument("--out", required=True)
    g = sp.add_parser("records-agg"); g.add_argument("parts", nargs="+"); g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"stretches": stretches, "rr": rr, "records": records, "scan": records, "records-agg": records_agg}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
