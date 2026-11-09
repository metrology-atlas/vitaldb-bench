# SPDX-License-Identifier: MIT
"""Can the beat-to-beat ECG-to-ART delay serve as a breath reference for the Primus? A feasibility test
(metrology-atlas: ventilator-breath-matching, primus-awp-anchoring; plan item B4).

    python tools/primus_ptt_arm.py scan --root R --cases 1,2,3 --modes M --ports P --step-rates S --out part.jsonl \
        [--time-budget S]
    python tools/primus_ptt_arm.py agg part*.jsonl --out DIR

Background. The Primus AWP is placed on the SNUADC timeline by breath matching against the CVP, or, without one,
against the arterial mean pressure (tools/primus_sync_accuracy.py). The ART arm sits a whole breath away from the CVP
arm in about a third of stretches (tools/primus_awp_anchor.py arms): the ventilator's breaths are nearly identical,
so a reference whose breath response has a different shape can lock onto the neighbouring breath. The pulse transit
time also swings with each breath (intrathoracic pressure changes the pre-ejection period and the arterial
pressure), from the ECG and the ART alone, so it would be a third reference, and one available in every recording
with an arterial line.

scan: the recordings given (those with AWP, CVP, ART and ECG II); SNUADC on its stamp model. Per beat (R peaks of
ECG II): the ECG-to-ART delay to the steepest ART upstroke within 0.05-0.45 s of the R peak (ptt), and the pulse
pressure (pp, systolic minus diastolic); each series interpolated onto the lag's 0.02 s grid, left empty where beats
are more than 3 s apart. The CVP arm is measured as in tools/primus_sync_accuracy.py (examples/ppv_by_breath.cvp_lag:
10-minute windows, breath-triggered averages). In each window it keeps, the other references (art, the 2 s mean;
ptt; pp; ptt and pp also with the opposite sign, as the shape match needs the sign of the breath response) are
matched to the same AWP average over a search of one and a half breath periods either side of the CVP lag, so that
each can pick a neighbouring breath; the best lag and its shape correlation r are kept, and the lag's agreement
between the window's odd and even breaths. (The release tool's own coarse step, an envelope correlation, does not
find the ptt and pp breath envelopes: r about 0.1-0.2 in a test recording.)

agg: per recording, ptt and pp take the sign with the higher median r. A window counts for an arm when r >= 0.8 and
its odd and even breaths agree within 0.1 s (the CVP arm's own rule); per arm against CVP, the difference minus its recording's
median difference (the physiology between the references), in breath periods: the share within a quarter period
and the share 0.75-1.25 periods off (a whole breath); the arms' coverage (windows with a lag against windows with a
CVP lag); and where ART and CVP disagree by a whole breath, which of them ptt sides with. Writes DIR/summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fallback_rate import load_step_rates  # noqa: E402
import ppv_by_breath as P  # noqa: E402
from pat import r_peaks, sample_times  # noqa: E402
from primus_sync_accuracy import lowpass_mean  # noqa: E402
from vent_sync_pilot import shape_lag, trig_avg  # noqa: E402

GRID = P.G          # the lag's own fine grid (0.02 s)


def beat_series(te, xe, xa, fs):
    """Per beat: time of the R peak, ECG-to-ART delay to the steepest upstroke, pulse pressure."""
    ok = np.isfinite(xe)
    rp = r_peaks(np.where(ok, xe, np.nanmedian(xe)), fs)
    d = np.r_[np.diff(np.nan_to_num(xa, nan=np.nanmedian(xa))), 0.0]
    a0, a1 = int(0.05 * fs), int(0.45 * fs)
    rows = []
    for i in rp:
        if i + a1 + 1 >= xa.size or i >= te.size:
            break
        seg = xa[i + a0:i + a1]
        if not np.isfinite(seg).all():
            continue
        j = i + a0 + int(np.argmax(d[i + a0:i + a1]))
        sbp, dbp = float(np.max(seg)), float(np.min(xa[max(i - int(0.15 * fs), 0):i + int(0.15 * fs)]))
        if not (10 < sbp - dbp < 150 and dbp > 10 and sbp < 300):
            continue
        rows.append((te[i], (j - i) / fs, sbp - dbp))
    return np.array(rows, float).reshape(-1, 3)


def on_grid(tb, v):
    """A beat series on the 0.02 s grid; empty where beats are more than 3 s apart."""
    tg = np.arange(tb[0], tb[-1], GRID)
    y = np.interp(tg, tb, v)
    k = np.searchsorted(tb, tg, side="right")
    gap = (tb[np.clip(k, 0, tb.size - 1)] - tb[np.clip(k - 1, 0, tb.size - 1)]) > 3
    y[gap] = np.nan
    return tg, y


def scan_case(c, a, modes, ports, step_rates):
    from vitaldb_reader.vital_file import read_vital
    v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital",
                   track_names={"Primus/AWP", "SNUADC/CVP", "SNUADC/ART", "SNUADC/ECG_II"})
    aw, cv, ar, ec = (v.tracks.get(k) for k in ("Primus/AWP", "SNUADC/CVP", "SNUADC/ART", "SNUADC/ECG_II"))
    if any(t is None or not t.record_times.size for t in (aw, cv, ar, ec)):
        return {"case": c, "skip": "needs AWP, CVP, ART and ECG II"}
    if ec.srate != ar.srate or ec.record_times.size != ar.record_times.size:
        return {"case": c, "skip": "ECG and ART records differ"}
    ta = sample_times(aw)
    xa = np.nan_to_num(aw.physical().astype(np.float64))[: ta.size]
    on = P.breath_onsets(ta, xa, float(aw.srate))
    if on.size < 200:
        return {"case": c, "skip": "fewer than 200 mechanical breaths"}
    port, mode = ports.get(c), modes.get(c)
    fs = float(ar.srate)
    te = P.model_times(ec, port, mode, step_rates)
    xe = ec.physical().astype(np.float64)[: te.size]
    xr = ar.physical().astype(np.float64)[: te.size]
    tc = P.model_times(cv, port, mode, step_rates)
    xc = cv.physical().astype(np.float64)[: tc.size]
    b = beat_series(te, xe, xr, fs)
    z = float(ta[0])
    res = {"case": c, "z": z, "beats": int(b.shape[0]), "windows": []}
    if b.shape[0]:
        res["ptt_s_p5_50_95"] = np.percentile(b[:, 1], [5, 50, 95]).round(4).tolist()
    tl, ll = P.cvp_lag(ta, xa, tc, xc, on)
    if not tl.size or b.shape[0] < 500:
        return res
    t0 = float(np.floor(min(ta[0], tc[0], te[0])))
    n = int((max(ta[-1], tc[-1], te[-1]) - t0) / P.G) + 1
    awp = P.binned(ta, xa, t0, P.G, n)
    refs = {"art": (te, lowpass_mean(np.nan_to_num(xr, nan=float(np.nanmedian(xr))), fs))}
    for k, col in (("ptt", 1), ("pp", 2)):
        refs[k] = on_grid(b[:, 0], b[:, col])
    grids = {}
    for k, (t, x) in refs.items():
        ok = np.isfinite(x)
        grids[k] = P.binned(t[ok], x[ok], t0, P.G, n)
        if k != "art":
            grids[k + "_neg"] = -grids[k]
    taus = np.arange(-4.0, 12.0, P.G)
    for m, lag in zip(tl, ll):
        o = on[(on >= m - 300) & (on < m + 300)]
        if o.size < 40:
            continue
        T = float(np.median(np.diff(o)))
        lo, hi = max(lag - 1.5 * T, -3.5), min(lag + 1.5 * T, 11.5)
        aa = trig_avg(awp, t0, o, taus)
        a1, a2 = trig_avg(awp, t0, o[0::2], taus), trig_avg(awp, t0, o[1::2], taus)
        w = {"t": round(float(m - z), 1), "T": round(T, 3), "cvp": round(float(lag), 4)}
        for k, gr in grids.items():
            L, r = shape_lag(aa, trig_avg(gr, t0, o, taus), taus, lo, hi)
            h1 = shape_lag(a1, trig_avg(gr, t0, o[0::2], taus), taus, lo, hi)[0]
            h2 = shape_lag(a2, trig_avg(gr, t0, o[1::2], taus), taus, lo, hi)[0]
            if L is not None:
                w[k] = [round(float(L), 4), round(float(r), 3),
                        None if h1 is None or h2 is None else round(float(abs(h1 - h2)), 3)]
        res["windows"].append(w)
    return res


def scan(a):
    modes = {int(k): v for k, v in json.loads(Path(a.modes).read_text()).items()}
    ports = {int(k): (v or {}).get("port") for k, v in json.loads(Path(a.ports).read_text()).items()}
    step_rates = load_step_rates(a.step_rates)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t_start = time.time()
    with open(out, "a", encoding="utf-8") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            if a.time_budget and time.time() - t_start > a.time_budget:
                break
            t1 = time.time()
            try:
                r = scan_case(c, a, modes, ports, step_rates.for_case(c))
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 1)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def agg(a):
    rows = [json.loads(l) for p in a.parts for l in open(p) if l.strip()]
    ok = [r for r in rows if r.get("windows")]
    out = {"recordings": len(rows), "with_cvp_windows": len(ok),
           "skipped": sum(1 for r in rows if r.get("skip")), "errors": sum(1 for r in rows if r.get("error")),
           "cvp_windows": sum(len(r["windows"]) for r in ok)}
    side = {"ptt_with_cvp": 0, "ptt_with_art": 0, "neither_or_no_ptt": 0}
    res = {k: {"d": [], "r": [], "rec": [], "cover": 0} for k in ("art", "ptt", "pp")}
    for r in ok:
        W = r["windows"]
        per = {}
        for k in res:
            cands = [k] if k == "art" else [k, k + "_neg"]
            best = max(cands, key=lambda c: np.median([w[c][1] for w in W if c in w] or [-1]))
            res[k]["r"] += [w[best][1] for w in W if best in w]
            good = [w for w in W if best in w and w[best][1] >= 0.8 and w[best][2] is not None and w[best][2] < 0.1]
            res[k]["cover"] += len(good)
            if len(good) < 3:
                continue
            d = np.array([w[best][0] - w["cvp"] for w in good])
            med = float(np.median(d))
            x = (d - med) / np.array([w["T"] for w in good])
            per[k] = dict(zip([w["t"] for w in good], x))
            res[k]["d"] += x.tolist()
            res[k]["rec"].append(float(np.mean(np.abs(x) < 0.25)))
        if "art" in per:
            for t, x in per["art"].items():
                if 0.75 <= abs(x) <= 1.25:
                    y = per.get("ptt", {}).get(t)
                    side["neither_or_no_ptt" if y is None else "ptt_with_cvp" if abs(y) < 0.25
                         else "ptt_with_art" if abs(y - x) < 0.25 else "neither_or_no_ptt"] += 1
    for k, v in res.items():
        d = np.abs(np.array(v["d"]))
        out[k] = {"windows_usable": v["cover"], "coverage_of_cvp_windows": round(v["cover"] / max(1, out["cvp_windows"]), 3),
                  "r_p25_50_75": np.percentile(v["r"], [25, 50, 75]).round(3).tolist() if v["r"] else None,
                  "recordings": len(v["rec"]),
                  "within_quarter_T": round(float(np.mean(d < 0.25)), 3) if d.size else None,
                  "whole_breath_off": round(float(np.mean((d >= 0.75) & (d <= 1.25))), 3) if d.size else None,
                  "recording_share_within_quarter_p25_50_75":
                      np.percentile(v["rec"], [25, 50, 75]).round(3).tolist() if v["rec"] else None}
    out["where_art_is_a_breath_off"] = side
    ptt = [r["ptt_s_p5_50_95"] for r in rows if r.get("ptt_s_p5_50_95")]
    out["ptt_median_s_p25_50_75"] = np.percentile([x[1] for x in ptt], [25, 50, 75]).round(3).tolist() if ptt else None
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    for k in ("root", "cases", "modes", "ports", "step_rates", "out"):
        s.add_argument("--" + k.replace("_", "-"), required=True)
    s.add_argument("--time-budget", type=float, default=0)
    g = sp.add_parser("agg")
    g.add_argument("parts", nargs="+")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
