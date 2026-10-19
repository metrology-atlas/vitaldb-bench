# SPDX-License-Identifier: MIT
"""The Primus stamps' step at each airway-pressure pause, measured directly from short windows either side.

    python tools/primus_pause_steps.py scan --root R --cases 1,2 --modes M --ports P --step-rates S --out part.jsonl
    python tools/primus_pause_steps.py agg part*.jsonl [--out DIR]

At each AWP pause (a gap over 5 s in the AWP stamps; about 21 s every 7,230 s) the Primus stamps step against the
SNUADC (and so against the recorder PC) by up to about +-0.5 s. The 10-minute windows of the breath-matching fit see
these steps only through fitted lines, and windows that straddle a pause mix two lags. Here each step is measured
on its own: the AWP-to-CVP lag in 2-minute windows (stepped by 1 minute) for 20 minutes before the gap and 20
minutes after it, a line fitted on each side, and the step taken as the difference of the two lines at the middle
of the gap. The SNUADC is on its stamp model, so the step is the Primus's against the PC timeline.

Recorded with each step: the gap's length; the drift built up since the previous pause (the stretch's slope from the
10-minute fit times its length), which a step that reset the drift would undo; the Primus's small whole-sample
corrections within a minute of the gap (AWP stamp steps between 1 ms and 1 s); whether CO2 also has a gap there;
and whether a SNUADC reset falls within 20 minutes (its 17.7 ms loss is modelled, but flagged). Controls: the same
measurement at up to two moments per recording at least 25 minutes from any pause, where no step is expected.

agg: steps against controls, their precision, and what predicts them (accumulated drift, gap length, correction
size, CO2 gap).
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
import ppv_by_breath as P  # noqa: E402
from pat import sample_times  # noqa: E402
from vent_sync_pilot import G, binned, shape_lag, trig_avg  # noqa: E402

SIDE_S, WIN_S, STEP_S = 1200.0, 120.0, 60.0
MIN_BREATHS, R_MIN, HALF_AGREE_S = 10, 0.7, 0.15
TAUS = np.arange(-4.0, 12.0, G)


def short_lags(o_all, awp50, cvp50, t0, centre_fn, a, b):
    """Lag in WIN_S windows stepped by STEP_S over [a, b]; returns arrays (mid times, lags)."""
    tt, ll = [], []
    for ws in np.arange(a, b - WIN_S + 1e-9, STEP_S):
        o = o_all[(o_all >= ws) & (o_all < ws + WIN_S)]
        if o.size < MIN_BREATHS:
            continue
        T, mid = float(np.median(np.diff(o))), ws + WIN_S / 2
        c = centre_fn(mid)
        lo, hi = c - T / 2, c + T / 2
        lag, r = shape_lag(trig_avg(awp50, t0, o, TAUS), trig_avg(cvp50, t0, o, TAUS), TAUS, lo, hi)
        h1 = shape_lag(trig_avg(awp50, t0, o[0::2], TAUS), trig_avg(cvp50, t0, o[0::2], TAUS), TAUS, lo, hi)[0]
        h2 = shape_lag(trig_avg(awp50, t0, o[1::2], TAUS), trig_avg(cvp50, t0, o[1::2], TAUS), TAUS, lo, hi)[0]
        if lag is not None and r >= R_MIN and h1 is not None and h2 is not None and abs(h1 - h2) < HALF_AGREE_S:
            tt.append(mid)
            ll.append(lag)
    return np.array(tt), np.array(ll)


def side_value(t, y, at):
    """Line through one side's windows, evaluated at `at`; with its standard error."""
    if t.size < 5:
        return None, None
    x = t - at
    X = np.column_stack([np.ones(x.size), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s2 = float(res @ res) / max(t.size - 2, 1)
    cov = s2 * np.linalg.pinv(X.T @ X)
    return float(b[0]), float(np.sqrt(cov[0, 0]))


def measure(at_lo, at_hi, on, awp50, cvp50, t0, centre_fn):
    """Step across [at_lo, at_hi] (a gap, or a single moment for a control)."""
    tb, yb = short_lags(on, awp50, cvp50, t0, centre_fn, at_lo - SIDE_S, at_lo)
    ta, ya = short_lags(on, awp50, cvp50, t0, centre_fn, at_hi, at_hi + SIDE_S)
    mid = 0.5 * (at_lo + at_hi)
    vb, sb = side_value(tb, yb, mid)
    va, sa = side_value(ta, ya, mid)
    if vb is None or va is None:
        return None
    return {"step_s": va - vb, "se_s": float(np.hypot(sb, sa)), "n_before": int(tb.size), "n_after": int(ta.size)}


def stamp_steps(tr):
    """Record-to-record stamp steps (s) that are neither gaps nor zero: the small corrections, with their times."""
    rt = np.asarray(tr.record_times, float)
    exp = rt[:-1] + np.asarray(tr.record_sizes[: rt.size - 1], float) / float(tr.srate)
    d = rt[1:] - exp
    k = (np.abs(d) >= 0.001) & (np.abs(d) < 1.0)
    return rt[1:][k], d[k]


def scan_case(c, a, modes, ports, step_rates):
    from physiosync import stamp_clock
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={"Primus/AWP", "Primus/CO2", "SNUADC/CVP"})
    aw, co2, cv = v.tracks.get("Primus/AWP"), v.tracks.get("Primus/CO2"), v.tracks.get("SNUADC/CVP")
    if aw is None or cv is None or not aw.record_times.size or not cv.record_times.size:
        return {"case": c, "skip": "no AWP or CVP"}
    ta = sample_times(aw)
    xa = np.nan_to_num(aw.physical().astype(np.float64))[: ta.size]
    on = P.breath_onsets(ta, xa, float(aw.srate))
    if on.size < 200:
        return {"case": c, "skip": "fewer than 200 mechanical breaths"}
    g = np.diff(ta)
    gaps = [(float(ta[i]), float(ta[i + 1])) for i in np.flatnonzero(g > 5)]
    if not gaps:
        return {"case": c, "skip": "no AWP pause"}
    tc = P.model_times(cv, ports.get(c), modes.get(c), step_rates)
    xc = cv.physical().astype(np.float64)[: tc.size]
    okc = np.isfinite(xc)
    t0 = float(np.floor(min(ta[0], tc[0])))
    n50 = int((max(ta[-1], tc[-1]) - t0) / G) + 1
    awp50, cvp50 = binned(ta, xa, t0, G, n50), binned(tc[okc], xc[okc], t0, G, n50)
    tl, ll = P.cvp_lag(ta, xa, tc, xc, on)
    if tl.size < 4:
        return {"case": c, "skip": "too few 10-min windows"}
    edges = np.array([gs for gs, _ in gaps])

    def centre_fn(t):
        # the 10-min lag nearest in time on the same side of any pause
        same = np.searchsorted(edges, tl) == np.searchsorted(edges, t)
        cand = tl[same] if same.any() else tl
        vals = ll[same] if same.any() else ll
        return float(vals[np.argmin(np.abs(cand - t))])

    # stretch slopes from the 10-min windows (s per s), per stretch
    st = np.searchsorted(edges, tl)
    slopes = {}
    for s in np.unique(st):
        m = st == s
        if m.sum() >= 3 and np.ptp(tl[m]) > 1800:
            slopes[int(s)] = float(np.polyfit(tl[m], ll[m], 1)[0])
    corr_t, corr_d = stamp_steps(aw)
    co2_gaps = []
    if co2 is not None and co2.record_times.size:
        tco = sample_times(co2)
        gc = np.diff(tco)
        co2_gaps = tco[1:][gc > 1.0]
    rt_ev = stamp_clock(cv.record_times, cv.record_offsets[: cv.record_times.size], float(cv.srate),
                        correction_sign=-1)["corrections"]["event_times_s"]
    rt_ev = np.array([x for x in rt_ev], float)
    z = float(ta[0])
    out = {"case": c, "z": z, "pauses": [], "controls": []}
    for k, (gs, ge) in enumerate(gaps):
        m = measure(gs, ge, on, awp50, cvp50, t0, centre_fn)
        prev = gaps[k - 1][1] if k > 0 else float(ta[0])
        sl = slopes.get(k)
        cw = (corr_t >= gs - 60) & (corr_t <= ge + 60)
        rec = {"t": round(gs - z, 1), "gap_s": round(ge - gs, 2), "since_prev_s": round(gs - prev, 1),
               "stretch_slope_ppm": round(sl * 1e6, 1) if sl is not None else None,
               "drift_built_s": round(sl * (gs - prev), 4) if sl is not None else None,
               "corrections_s": [round(float(x), 4) for x in corr_d[cw]],
               "correction_after_gap_s": [round(float(x - ge), 1) for x in corr_t[cw]],
               "co2_gap": bool(np.any(np.abs(np.asarray(co2_gaps) - ge) < 30)) if len(co2_gaps) else False,
               "snuadc_reset_near": bool(np.any(np.abs(rt_ev - gs) < SIDE_S)) if rt_ev.size else False}
        if m:
            rec.update({k2: (round(v2, 4) if isinstance(v2, float) else v2) for k2, v2 in m.items()})
        out["pauses"].append(rec)
    # controls: moments >= 25 min from any pause, inside the AWP record, at most two
    rng = np.random.default_rng(c)
    cand = np.arange(ta[0] + SIDE_S, ta[-1] - SIDE_S, 300.0)
    cand = cand[[np.min(np.abs(x - edges)) > 1500 for x in cand]] if cand.size else cand
    for x in rng.permutation(cand)[:2]:
        m = measure(x, x, on, awp50, cvp50, t0, centre_fn)
        if m:
            out["controls"].append({"t": round(float(x - z), 1), **{k2: (round(v2, 4) if isinstance(v2, float) else v2)
                                                                   for k2, v2 in m.items()}})
    return out


def scan(a):
    modes = {int(k): v for k, v in json.loads(Path(a.modes).read_text()).items()}
    ports = {int(k): (v or {}).get("port") for k, v in json.loads(Path(a.ports).read_text()).items()}
    step_rates = json.loads(Path(a.step_rates).read_text())
    out = Path(a.out)
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
                r = scan_case(c, a, modes, ports, step_rates)
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 1)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def agg(a):
    R = [json.loads(l) for p in a.parts for l in open(p, encoding="utf-8") if l.strip()]
    S = [dict(case=r["case"], **p) for r in R for p in r.get("pauses", []) if "step_s" in p]
    C = [dict(case=r["case"], **p) for r in R for p in r.get("controls", []) if "step_s" in p]
    pct = lambda x: [round(float(v), 4) for v in np.percentile(x, [5, 10, 25, 50, 75, 90, 95])] if len(x) else None
    st = np.array([s["step_s"] for s in S]); se = np.array([s["se_s"] for s in S])
    ct = np.array([s["step_s"] for s in C])
    out = {"recordings_read": len(R), "errors": sum(1 for r in R if "error" in r),
           "skipped": {k: sum(1 for r in R if r.get("skip") == k) for k in {r.get("skip") for r in R if r.get("skip")}},
           "pauses_measured": len(S), "pauses_total": sum(len(r.get("pauses", [])) for r in R),
           "step_s_p5_10_25_50_75_90_95": pct(st), "control_s_p5_10_25_50_75_90_95": pct(ct),
           "step_se_s_median": round(float(np.median(se)), 4) if len(se) else None,
           "share_abs_step_over_3se": round(float(np.mean(np.abs(st) > 3 * se)), 3) if len(se) else None,
           "controls_abs_over_3se": round(float(np.mean(np.abs(ct) > 3 * np.array([s["se_s"] for s in C]))), 3) if C else None}
    # what predicts the step
    def corr(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        return (round(float(np.corrcoef(x[m], y[m])[0, 1]), 3), int(m.sum())) if m.sum() > 10 else None
    drift = [s.get("drift_built_s") if s.get("drift_built_s") is not None else np.nan for s in S]
    out["vs_drift_built"] = {"corr": corr(drift, st)}
    m = np.isfinite(np.array(drift, float))
    if m.sum() > 10:
        from scipy.stats import theilslopes
        ts = theilslopes(st[m], -np.array(drift, float)[m])
        out["vs_drift_built"]["share_of_drift_taken_back_theil"] = round(float(ts[0]), 3)
    out["vs_gap_length"] = {"corr": corr([s["gap_s"] for s in S], st)}
    csum = [sum(s["corrections_s"]) if s["corrections_s"] else 0.0 for s in S]
    out["vs_small_corrections"] = {"corr": corr(csum, st), "pauses_with_correction": int(sum(1 for s in S if s["corrections_s"]))}
    co = np.array([s["co2_gap"] for s in S])
    out["co2_gap_too"] = {"pauses": int(co.sum()), "median_step_s_with": round(float(np.median(st[co])), 4) if co.any() else None,
                          "median_step_s_without": round(float(np.median(st[~co])), 4) if (~co).any() else None}
    near = np.array([s["snuadc_reset_near"] for s in S])
    out["snuadc_reset_within_20min"] = {"pauses": int(near.sum()),
                                        "median_abs_step_s_with": round(float(np.median(np.abs(st[near]))), 4) if near.any() else None,
                                        "median_abs_step_s_without": round(float(np.median(np.abs(st[~near]))), 4) if (~near).any() else None}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1))
        with open(Path(a.out) / "steps.jsonl", "w", encoding="utf-8") as f:
            for s in S:
                f.write(json.dumps(s) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    for k in ("root", "cases", "modes", "ports", "step_rates", "out"):
        s.add_argument("--" + k.replace("_", "-"), required=True)
    s.add_argument("--time-budget", type=float, default=0)
    g = sp.add_parser("agg"); g.add_argument("parts", nargs="+"); g.add_argument("--out")
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
