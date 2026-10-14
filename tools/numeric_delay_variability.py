# SPDX-License-Identifier: MIT
"""Are the numeric delays fixed? Per case and per 30-min block, with the algorithm held fixed. This machine only.

    python tools/numeric_delay_variability.py scan --root R --cases 1,2 --units units.csv --step-rates rates.json --out part.jsonl
    python tools/numeric_delay_variability.py agg part*.jsonl --root R --devices va/p?.jsonl --summary summary.json

The pilot (tools/numeric_derivation_pilot.py) let each case choose its own window, so its delays trade off with
the window. Here every case uses the consensus algorithm of the pilot, so the delay d (from the window's close,
uniform phase model, same likelihood) is comparable across cases and blocks:

    HR       60/median(R-R) over 6 s, truncated      ART_SBP  median of beat max over 3 s
    ART_MBP  EWMA of beat mean, 1/4                  ART_DBP  EWMA of beat min, 1/2
    CVP      mean of the waveform over 4 s           ETCO2, PIP  last breath
    RR_CO2   60/mean(breath interval) over 30 s, rounded

Phase. The likelihood lets each value have been computed at an unknown moment within a phase interval before
the stamp's delay. For the Solar8000 numerics and RR_CO2, which the monitor recomputes on its own schedule and
the recorder polls, that interval is the gap since the previous record. ETCO2 and PIP are values of the last
completed breath, held until the Primus is polled (about every 7 s); the reference series already holds the
last breath, so a gap-wide phase counted the hold twice and pulled their delays about half a gap too low,
below zero for ETCO2 (an availability-rule impossibility). Their phase is one Primus record (0.48 s). A
150-case comparison (13 October) found per-record phase better for PIP in 78% of cases (median log-likelihood
+51) and for ETCO2 in 54% (+3.5), and the gap phase better for RR_CO2 in 99%.

scan: per track, d over the whole case (profile 95% interval: log-likelihood within 1.92 of its maximum, which
treats records as independent and so is too narrow) and d per 30-min block (>= 150 records), with the block's
median event interval and elapsed time.
agg: variance components of the block delays (one-way random effects: between-case tau, within-case sigma_w,
ICC), the spread of case delays against what within-case scatter alone predicts, drift with elapsed time,
dependence on R-R, correlation of case delays between tracks, and case delays by recording set-up (Solar8000
port, port set, file version, stamp grid, Primus CO2 scale, department, case-ID tertile; Kruskal-Wallis).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import numeric_derivation_pilot as ndp  # noqa: E402
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import TRACK_NUMERIC  # noqa: E402

warnings.filterwarnings("ignore")

BLOCK_S = 1800.0
# phase width per track where it is not the gap since the previous record (see the module docstring)
PRIMUS_RECORD_S = 0.48
PHASE_RECORD = {"ETCO2", "PIP"}
D = np.round(np.arange(-3.0, 15.0001, 0.05), 3)
# track: (algorithm, quantisation, free bias, window (s, for the window centre), reference series key)
ALGO = {
    "HR": ("median|t|6", "floor", False, 6.0, "rr"),
    "ART_SBP": ("median|t|3", "round", True, 3.0, "sbp"),
    "ART_MBP": ("ewma|n|4", "round", True, None, "mbp"),
    "ART_DBP": ("ewma|n|2", "round", True, None, "dbp"),
    "CVP": ("wave_mean|t|4", "round", True, 4.0, "cvp"),
    "ETCO2": ("mean|n|1", "round", True, None, "etco2"),
    "PIP": ("mean|n|1", "round", True, None, "pip"),
    "RR_CO2": ("60/mean_int|t|30", "round", False, 30.0, "rr_int"),
}


def series(algo, tc, x, wave=None):
    fn, wt, w = algo.split("|")
    w = float(w)
    if fn == "median" and wt == "t" and algo.startswith("median|t|6"):
        return tc, 60.0 / ndp.roll_t(tc, x, w, "median")
    if fn == "median":
        return tc, ndp.roll_t(tc, x, w, "median")
    if fn == "ewma":
        return tc, ndp.ewma(x, int(w))
    if fn == "mean" and wt == "n" and w == 1:
        return tc, x
    if fn == "60/mean_int":
        return tc, 60.0 / ndp.roll_t(tc, x, w, "mean")
    if fn == "wave_mean":
        t0, s, n = wave
        cs, cn = np.r_[0.0, np.cumsum(s)], np.r_[0, np.cumsum(n)]
        tg = t0 + 0.1 * (np.arange(s.size) + 1)
        k = int(round(w / 0.1))
        j = np.arange(s.size) + 1
        j0 = np.maximum(j - k, 0)
        num, den = cs[j] - cs[j0], cn[j] - cn[j0]
        y = np.where(den > 0.6 * k * (n.sum() / max(np.count_nonzero(n), 1)), num / np.maximum(den, 1), np.nan)
        return tg, y
    raise ValueError(algo)


def profile(ot, ov, og, tc, y, q, quant, free, span, Dg, dd):
    ll, b, s = ndp.fit(ot, ov, og, tc, y, q, quant, free, span, Dg, dd)
    k = int(np.nanargmax(ll))
    ok = ll >= ll[k] - 1.92
    lo = k
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = k
    while hi < ok.size - 1 and ok[hi + 1]:
        hi += 1
    return float(Dg[k]), (float(Dg[lo]), float(Dg[hi])), float(b[k]), float(s[k]), bool(lo == 0 or hi == ok.size - 1)


def track_delays(name, ot, ov, og, tc, y, ev):
    algo, quant, free, _, _ = ALGO[name]
    lo_v, hi_v = ndp.VALID.get(name, (0, 100))
    span = hi_v - lo_v
    q = ndp.step_res(ov)
    okc = np.isfinite(y)
    tc, y = tc[okc], y[okc]
    if tc.size < 100:
        return None
    a = ndp.subsample(ot, ov, og, 8000)
    d, ci, b, s, edge = profile(*a, tc, y, q, quant, free, span, D, 0.05)
    out = {"n": int(ot.size), "q": q, "d": d, "ci": ci, "bias": round(b, 3), "sigma": round(s, 3), "edge": edge,
           "blocks": []}
    Db = np.round(np.arange(d - 4.0, d + 4.0001, 0.05), 3)
    t0 = ot[0]
    for k0 in np.arange(ot[0], ot[-1], BLOCK_S):
        m = (ot >= k0) & (ot < k0 + BLOCK_S)
        if m.sum() < 150:
            continue
        bt = ndp.subsample(ot[m], ov[m], og[m], 3000)
        db, cib, _, _, eb = profile(*bt, tc, y, q, quant, free, span, Db, 0.05)
        et, ei = ev
        mm = (et >= k0) & (et < k0 + BLOCK_S)
        out["blocks"].append({"h": round(float((k0 + BLOCK_S / 2 - t0) / 3600), 3), "d": db, "ci": cib,
                              "edge": eb, "n": int(m.sum()),
                              "interval": round(float(np.median(ei[mm])), 3) if mm.sum() > 20 else None})
    return out


ONLY: set[str] | None = None   # set by --only: scan just these tracks


def scan_case(root, c, u, sr):
    solar = ONLY is None or bool(ONLY & {"HR", "ART_SBP", "ART_MBP", "ART_DBP", "CVP"})
    names = {"Primus/AWP", "Primus/CO2"}
    if solar:
        names |= {"SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/CVP"}
    names |= {ndp.SOLAR[k] for k in ("HR", "ART_SBP", "ART_MBP", "ART_DBP", "CVP")}
    names |= {ndp.PRIMUS[k] for k in ("ETCO2", "PIP", "RR_CO2")}
    v = ndp.read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names=names)
    res = {"case": c, "tracks": {}, "spans": {}}
    for n in names:
        tr = v.tracks.get(n)
        if tr is not None and tr.track_type == TRACK_NUMERIC and tr.record_times.size:
            res["spans"][n] = [float(tr.record_times.min()), float(tr.record_times.max())]
    W = {}
    for k in ("ecg", "art", "cvp"):
        tr = v.tracks.get(ndp.WAVES[k])
        if k == "ecg" and (tr is None or not tr.record_times.size):
            tr = v.tracks.get("SNUADC/ECG_V5")
        if tr is None or not tr.record_times.size:
            continue
        t, clk = ndp.snu_times(tr, u, sr)
        W[k] = (t, tr.physical().astype(np.float64)[: t.size], float(tr.srate))
        if k == "ecg":
            res["snuadc"] = {"rate_ppm": round(clk["rate_ppm"], 1), "source": clk["rate_source"], "srate": W[k][2]}
    if "ecg" in W:
        te, xe, fs = W["ecg"]
        bs = ndp.beat_series(te, xe, fs, W["art"][:2] if "art" in W else None, W["cvp"][:2] if "cvp" in W else None)
        t0 = float(np.floor(te[0]))
        nbin = int((te[-1] - t0) / 0.1) + 1
        cvp_bins = ndp.wave_bins(W["cvp"][0], W["cvp"][1], t0, nbin) if "cvp" in W else None
        W.clear()
        for name in ("HR", "ART_SBP", "ART_MBP", "ART_DBP", "CVP"):
            if ONLY is not None and name not in ONLY:
                continue
            algo, _, _, _, key = ALGO[name]
            ser = bs.get(key)
            tr = v.tracks.get(ndp.SOLAR[name])
            if tr is None or ser is None or ser[0].size < 200:
                continue
            if name == "CVP" and cvp_bins is None:
                continue
            ot, ov, og = ndp.numeric_obs(tr, *ndp.VALID[name])
            m = (ot > ser[0][0] + 60) & (ot < ser[0][-1])
            ot, ov, og = ot[m], ov[m], og[m]
            if ot.size < 300:
                continue
            tc, y = series(algo, ser[0], ser[1], cvp_bins)
            r = track_delays(name, ot, ov, og, tc, y, bs["rr"])
            if r:
                res["tracks"][name] = r
        del bs
    W.clear()
    for key in ("awp", "co2"):
        tr = v.tracks.get(ndp.WAVES[key])
        if tr is None or not tr.record_times.size:
            continue
        t = ndp.sample_times(tr)
        x = np.nan_to_num(tr.physical().astype(np.float64))[: t.size]
        starts = ndp.stretches_of(tr)
        ser = ndp.awp_breaths(t, x, float(tr.srate)) if key == "awp" else ndp.co2_breaths(t, x, float(tr.srate))
        del t, x
        if ser is None:
            continue
        for name in (("PIP",) if key == "awp" else ("ETCO2", "RR_CO2")):
            if ONLY is not None and name not in ONLY:
                continue
            algo, _, _, _, sk = ALGO[name]
            ntr = v.tracks.get(ndp.PRIMUS[name])
            tc0, xv = ser[sk]
            if ntr is None or tc0.size < 100:
                continue
            tc = ndp.primus_to_recorder(tc0, starts, ndp.PRIMUS_PPM)
            ot, ov, og = ndp.numeric_obs(ntr, *ndp.VALID[name])
            if name in PHASE_RECORD:
                og = np.full_like(og, PRIMUS_RECORD_S)
            m = (ot > tc[0] + 60) & (ot < tc[-1])
            if m.sum() < 300:
                continue
            tcs, y = series(algo, tc, xv)
            iv = ser["rr_int"] if "rr_int" in ser else (tc0[1:], np.diff(tc0))
            r = track_delays(name, ot[m], ov[m], og[m], tcs, y,
                             (ndp.primus_to_recorder(iv[0], starts, ndp.PRIMUS_PPM), iv[1]))
            if r:
                res["tracks"][name] = r
    return res


HR_WINDOWS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]


def scan_hr_window(root, c, u, sr):
    """HR only: best window T (60/median of R-R over T s, truncated) and its delay, for the two-group check."""
    v = ndp.read_vital(Path(root) / "vital_files" / f"{c:04d}.vital",
                       track_names={"SNUADC/ECG_II", "SNUADC/ECG_V5", ndp.SOLAR["HR"]})
    tr = v.tracks.get("SNUADC/ECG_II")
    if tr is None or not tr.record_times.size:
        tr = v.tracks.get("SNUADC/ECG_V5")
    hr = v.tracks.get(ndp.SOLAR["HR"])
    if tr is None or hr is None or not tr.record_times.size:
        return {"case": c, "skip": "no ECG or HR"}
    t, _ = ndp.snu_times(tr, u, sr)
    bs = ndp.beat_series(t, tr.physical().astype(np.float64)[: t.size], float(tr.srate), None, None)
    del t
    tc, x = bs["rr"]
    ot, ov, og = ndp.numeric_obs(hr, *ndp.VALID["HR"])
    m = (ot > tc[0] + 60) & (ot < tc[-1])
    ot, ov, og = ndp.subsample(ot[m], ov[m], og[m], 6000)
    if ot.size < 300:
        return {"case": c, "skip": "too few records"}
    q = ndp.step_res(ov)
    rows = []
    for T in HR_WINDOWS:
        for fn in ("median", "60/mean_int"):
            y = 60.0 / ndp.roll_t(tc, x, T, "median" if fn == "median" else "mean")
            ok = np.isfinite(y)
            d, ci, _, s_, edge = profile(ot, ov, og, tc[ok], y[ok], q, "floor", False, 225.0, D, 0.05)
            ll, _, _ = ndp.fit(ot, ov, og, tc[ok], y[ok], q, "floor", False, 225.0, np.array([d]), 0.05)
            rows.append({"fn": fn, "T": T, "d": d, "ll": round(float(ll[0]), 2), "edge": edge})
    rows.sort(key=lambda r: -r["ll"])
    return {"case": c, "n": int(ot.size), "best": rows[0], "rows": rows}


# ----------------------------------------------------------------------------------------------- aggregate

def kruskal(groups):
    from scipy.stats import kruskal as kw  # noqa: PLC0415
    g = [np.asarray(x) for x in groups if len(x) >= 5]
    if len(g) < 2:
        return None
    h, p = kw(*g)
    n = sum(x.size for x in g)
    return {"groups": len(g), "n": int(n), "H": round(float(h), 2), "p": float(f"{p:.3g}"),
            "eta2": round(float(max(0.0, (h - len(g) + 1) / (n - len(g)))), 3)}


def var_components(blocks_by_case):
    """One-way random effects on block delays (unbalanced ANOVA estimator)."""
    groups = [np.asarray(b) for b in blocks_by_case if len(b) >= 2]
    if len(groups) < 5:
        return None
    k = len(groups)
    n_i = np.array([g.size for g in groups])
    N = n_i.sum()
    gm = np.concatenate(groups).mean()
    means = np.array([g.mean() for g in groups])
    ssb = float((n_i * (means - gm) ** 2).sum())
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    tau2 = max(0.0, (msb - msw) / n0)
    from scipy.stats import f as fdist  # noqa: PLC0415
    p = float(fdist.sf(msb / msw, k - 1, N - k)) if msw > 0 else 0.0
    return {"cases": k, "blocks": int(N), "sigma_within_s": round(float(np.sqrt(msw)), 3),
            "tau_between_s": round(float(np.sqrt(tau2)), 3), "icc": round(float(tau2 / (tau2 + msw)), 3) if msw + tau2 else None,
            "F_p": float(f"{p:.3g}")}


def agg(paths, root, dev_paths, out):
    R = {}
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            if "tracks" in r:
                R[r["case"]] = r
    devs = {}
    for p in dev_paths or []:
        for l in open(p):
            r = json.loads(l)
            if r["case"] in R:
                devs[r["case"]] = r
    cl = {}
    cf = Path(root) / "clinical_data.csv"
    if cf.exists():
        cl = {int(r["caseid"]): r for r in csv.DictReader(open(cf, encoding="utf-8-sig"))}
    S = {"cases": len(R), "block_s": BLOCK_S, "tracks": {}}
    case_d = defaultdict(dict)
    for name in ALGO:
        rows = [(c, r["tracks"][name]) for c, r in R.items() if name in r["tracks"]]
        rows = [(c, t) for c, t in rows if not t["edge"]]
        if len(rows) < 10:
            continue
        d = np.array([t["d"] for _, t in rows])
        ciw = np.array([t["ci"][1] - t["ci"][0] for _, t in rows])
        for c, t in rows:
            case_d[name][c] = t["d"]
        blocks = []
        within_sd, slopes, rr_slopes = [], [], []
        for _, t in rows:
            b = [x for x in t["blocks"] if not x["edge"]]
            if len(b) >= 2:
                bd = np.array([x["d"] for x in b])
                blocks.append(bd)
                within_sd.append(bd.std(ddof=1))
                h = np.array([x["h"] for x in b])
                if len(b) >= 3 and h.std() > 0:
                    slopes.append(np.polyfit(h, bd, 1)[0])
                iv = np.array([x["interval"] if x["interval"] is not None else np.nan for x in b])
                ok = np.isfinite(iv)
                if ok.sum() >= 3 and iv[ok].std() > 0.02:
                    rr_slopes.append(np.polyfit(iv[ok], bd[ok], 1)[0])
        vc = var_components(blocks)
        nb = np.array([len(b) for b in blocks]) if blocks else np.array([1])
        pred_sd = float(np.median(within_sd) / np.sqrt(np.median(nb))) if within_sd else None
        o = {"cases": len(rows), "edge_excluded": sum(1 for c, r in R.items() if name in r["tracks"] and r["tracks"][name]["edge"]),
             "d_p10_25_50_75_90": np.percentile(d, [10, 25, 50, 75, 90]).round(2).tolist(),
             "case_sd_s": round(float(d.std(ddof=1)), 3),
             "case_iqr_s": round(float(np.subtract(*np.percentile(d, [75, 25]))), 3),
             "profile_ci_width_p50": round(float(np.median(ciw)), 3),
             "within_case_block_sd_p50": round(float(np.median(within_sd)), 3) if within_sd else None,
             "case_sd_expected_from_within": round(pred_sd, 3) if pred_sd else None,
             "variance_components": vc,
             "drift_s_per_h_p10_50_90": np.percentile(slopes, [10, 50, 90]).round(3).tolist() if len(slopes) >= 5 else None,
             "delay_per_s_interval_p10_50_90": np.percentile(rr_slopes, [10, 50, 90]).round(2).tolist() if len(rr_slopes) >= 5 else None}
        if ALGO[name][3]:
            o["centre_lag_p50_s"] = round(float(np.median(d) + ALGO[name][3] / 2), 2)
        # set-up covariates
        cov = {}
        def fp(c):
            dv = devs.get(c, {})
            de = dv.get("devices", {})
            mk = dv.get("markers", {})
            co2 = dv.get("tracks", {}).get("Primus/CO2", {}).get("gain")
            return {"solar_port": (de.get("Solar8000") or [None, None])[1],
                    "port_set": "|".join(f"{k}:{v[1]}" for k, v in sorted(de.items())),
                    "version": dv.get("version"),
                    "stamp_1ms": None if "stamp_on_1ms" not in mk else bool(mk["stamp_on_1ms"] > 0.5),
                    "primus_co2_gain": None if co2 is None else round(co2, 4),
                    "snuadc_srate": R[c].get("snuadc", {}).get("srate"),
                    "department": cl.get(c, {}).get("department"),
                    "caseid_tertile": int(np.digitize(c, [2130, 4260]))}
        F = {c: fp(c) for c, _ in rows}
        for key in ("solar_port", "port_set", "version", "stamp_1ms", "primus_co2_gain", "snuadc_srate", "department",
                    "caseid_tertile"):
            g = defaultdict(list)
            for c, t in rows:
                g[F[c][key]].append(t["d"])
            g.pop(None, None)
            kr = kruskal(list(g.values()))
            if kr:
                big = sorted(((k_, v_) for k_, v_ in g.items() if len(v_) >= 5), key=lambda kv: -len(kv[1]))[:8]
                kr["medians"] = {str(k_): [round(float(np.median(v_)), 2), len(v_)] for k_, v_ in big}
                cov[key] = kr
        o["by_setup"] = cov
        S["tracks"][name] = o
    # correlation of case delays between tracks
    from scipy.stats import spearmanr  # noqa: PLC0415
    names = list(case_d)
    corr = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            cs = sorted(set(case_d[a]) & set(case_d[b]))
            if len(cs) >= 15:
                rho, p = spearmanr([case_d[a][c] for c in cs], [case_d[b][c] for c in cs])
                corr[f"{a}~{b}"] = {"n": len(cs), "rho": round(float(rho), 3), "p": float(f"{p:.3g}")}
    S["between_track_spearman"] = corr
    json.dump(S, open(out, "w"), indent=1)
    print(json.dumps(S, indent=1))


def hrwin_agg(paths, dev_paths, out):
    """Best HR window and delay by case group (case delay with the fixed 6 s window: low < 3.1 s <= high)."""
    R = [json.loads(l) for p in paths for l in open(p)]
    G = [r for r in R if "best" in r]
    dev = {}
    for p in dev_paths or []:
        for l in open(p):
            r = json.loads(l)
            dev[r["case"]] = (r.get("devices", {}).get("Solar8000") or [None, None])[1]
    out_d = {"cases": len(R), "fitted": len(G), "groups": {}}
    for r in G:
        six = [x for x in r["rows"] if x["fn"] == "median" and x["T"] == 6][0]
        r["group"] = "high" if six["d"] >= 3.1 else "low"
    for g in ("low", "high"):
        rs = [r for r in G if r["group"] == g]
        if not rs:
            continue
        b = [r["best"] for r in rs]
        T = np.array([x["T"] for x in b], float)
        dd = np.array([x["d"] for x in b])
        out_d["groups"][g] = {
            "cases": len(rs),
            "best_fn": Counter(x["fn"] for x in b).most_common(),
            "best_T": sorted(Counter(int(t) for t in T).items()),
            "delay_from_close_p25_50_75": np.percentile(dd, [25, 50, 75]).round(2).tolist(),
            "centre_lag_p25_50_75": np.percentile(dd + T / 2, [25, 50, 75]).round(2).tolist(),
            "solar_ports": Counter(dev.get(r["case"]) for r in rs).most_common(8),
        }
    json.dump(out_d, open(out, "w"), indent=1)
    print(json.dumps(out_d, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan", "agg", "hrwin", "hrwin-agg"])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--units")
    ap.add_argument("--step-rates")
    ap.add_argument("--devices", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--summary")
    ap.add_argument("--only", help="comma-separated tracks to scan (e.g. ETCO2,PIP); default all")
    a = ap.parse_args()
    global ONLY
    ONLY = {x for x in a.only.split(",") if x} if a.only else None
    if a.mode == "agg":
        agg(a.inputs, a.root, a.devices, a.summary)
        return
    if a.mode == "hrwin-agg":
        hrwin_agg(a.inputs, a.devices, a.summary)
        return
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    sr = json.loads(Path(a.step_rates).read_text())
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in open(a.out)}
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        try:
            r = (scan_hr_window if a.mode == "hrwin" else scan_case)(a.root, c, U.get(c, {}), sr)
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:300]}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
