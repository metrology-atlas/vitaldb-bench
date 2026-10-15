# SPDX-License-Identifier: MIT
"""How are the monitor numerics derived from their waveforms? Pilot, per case, this machine only.

    python tools/numeric_derivation_pilot.py --root R --cases 1,2 --units units.csv --step-rates rates.json --out out.jsonl
    python tools/numeric_derivation_pilot.py --root R --select-from cases.txt --n 24 --seed 3 --out cases.txt

Numerics and the reference series they are compared with:

    Solar8000/HR                    R-R intervals (SNUADC/ECG_II R peaks)
    Solar8000/ART_SBP, _MBP, _DBP   per-beat max / mean / min of SNUADC/ART (R+0.1 s to the next R+0.1 s)
    Solar8000/CVP                   per-beat mean of SNUADC/CVP
    Primus/ETCO2, Primus/RR_CO2     per-breath max of Primus/CO2 and breath interval (CO2 upstrokes)
    Primus/PIP_MBAR, PEEP_MBAR      per-breath max of Primus/AWP and its median over the last 0.5 s before the next breath

Clocks: Solar8000 and Primus numerics carry recorder stamps. SNUADC samples are put on the stamp-model
line (physiosync.stamp_clock, rules as tools/vent_sync_pilot.py), i.e. recorder time. Primus waveform stamps
follow its sample count within a stretch (re-anchored at gaps), so they are mapped to recorder time with a
drift r (default 115 ppm, refined over 0-170 ppm for the best candidate; r = 0 winning would mean the numeric
shares the waveform's clock).

Model of one reported value v at stamp t (the uniform model agreed for this pilot):
    the monitor computed it at u = t - d - phi, phi ~ Uniform[0, g), g = the gap since the previous record
    (the value is stamped at the close of its availability interval);
    the value is f(window ending at u) of the reference events completed by u;
    the internal value lies in the reporting bin (v +- q/2 when rounding, [v, v+q) when truncating), blurred by
    our own measurement error sigma (robust, per delay), plus a 2% outlier floor; a free bias b for pressures and
    gases (our per-beat/breath values are not the monitor's), none for rates.
Candidates: f in {mean, median, EWMA (alpha = 1/N)}, plus for rates 60/mean(interval) vs mean(60/interval)
and, for RR, a breath count over T s; windows of N events or T seconds; for MBP and CVP also a plain time mean
of the waveform over T s. The delay d is searched over -3..15 s (negative = the value appears before the
data it describes, i.e. front-stamping).

Per track out: the best candidate overall and per family (log-likelihood, delay, bias, sigma); rounding vs
truncation for rates; the delay of the best seconds- and events-window candidates by tertile of the event
interval (a window of N beats has a delay that grows with R-R, a window of T seconds does not); repeats
between consecutive records against the predicted change (a value held between updates repeats even when
the prediction moves); a hold test for Solar numerics (value refreshed every U s at a fitted phase per
30 min, U = 2..12 s; compared by BIC with the uniform model).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
from pat import r_peaks, sample_times  # noqa: E402
from vent_sync_pilot import breath_onsets, snu_times, stretches_of  # noqa: E402
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: E402

warnings.filterwarnings("ignore")

SOLAR = {"HR": "Solar8000/HR", "ART_SBP": "Solar8000/ART_SBP", "ART_MBP": "Solar8000/ART_MBP",
         "ART_DBP": "Solar8000/ART_DBP", "CVP": "Solar8000/CVP"}
PRIMUS = {"ETCO2": "Primus/ETCO2", "RR_CO2": "Primus/RR_CO2", "PIP": "Primus/PIP_MBAR", "PEEP": "Primus/PEEP_MBAR"}
WAVES = {"ecg": "SNUADC/ECG_II", "art": "SNUADC/ART", "cvp": "SNUADC/CVP", "awp": "Primus/AWP", "co2": "Primus/CO2"}
VALID = {"HR": (25, 250), "ART_SBP": (30, 300), "ART_MBP": (15, 250), "ART_DBP": (5, 200), "CVP": (-10, 40),
         "ETCO2": (5, 100), "RR_CO2": (3, 60), "PIP": (3, 80), "PEEP": (0, 40)}
RATE = {"HR", "RR_CO2"}
N_BEATS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]
T_BEATS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]
N_BREATHS = [1, 2, 3, 4, 5, 6, 8, 10, 12]
T_BREATHS = [10, 15, 20, 30, 45, 60]
T_WAVE = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]
EPS = 0.02
STALE = 30.0          # no reference event this long before the query time: no prediction
PRIMUS_PPM = 115.0


# ----------------------------------------------------------------------------------------------- windows

def roll_n(x, N, how):
    if x.size < N:
        return np.full(x.size, np.nan)
    if how == "mean":
        c = np.r_[0.0, np.cumsum(x)]
        y = (c[N:] - c[:-N]) / N
    else:
        y = np.median(sliding_window_view(x, N), axis=1)
    return np.r_[np.full(N - 1, np.nan), y]


def roll_t(tc, x, T, how):
    j0 = np.searchsorted(tc, tc - T, side="right")
    n = np.arange(tc.size) + 1 - j0
    if how == "mean":
        c = np.r_[0.0, np.cumsum(x)]
        return (c[1:] - c[j0]) / n
    if how == "count":
        return n * 60.0 / T
    return np.array([np.median(x[a : i + 1]) for i, a in enumerate(j0)])


def ewma(x, N):
    a = 1.0 / N
    return signal.lfilter([a], [1, -(1 - a)], x, zi=[(1 - a) * x[0]])[0]


def candidates(kind, tc, x, wave=None):
    """kind: 'rate' (x = event intervals in s, output per minute), 'value'. Returns {name: (tc, y)}."""
    out = {}
    beats = kind in ("rate_beat", "value_beat")
    NL, TL = (N_BEATS, T_BEATS) if beats else (N_BREATHS, T_BREATHS)
    if kind.startswith("rate"):
        inst = 60.0 / x
        for N in NL:
            out[f"60/mean_int|n|{N}"] = 60.0 / roll_n(x, N, "mean")
            out[f"mean_rate|n|{N}"] = roll_n(inst, N, "mean")
            out[f"median|n|{N}"] = 60.0 / roll_n(x, N, "median")
            if N > 1:
                out[f"ewma_rate|n|{N}"] = ewma(inst, N)
        for T in TL:
            out[f"60/mean_int|t|{T}"] = 60.0 / roll_t(tc, x, T, "mean")
            out[f"mean_rate|t|{T}"] = roll_t(tc, inst, T, "mean")
            out[f"median|t|{T}"] = 60.0 / roll_t(tc, x, T, "median")
            if not beats:
                out[f"count|t|{T}"] = roll_t(tc, x, T, "count")
    else:
        for N in NL:
            out[f"mean|n|{N}"] = roll_n(x, N, "mean")
            if N > 2:
                out[f"median|n|{N}"] = roll_n(x, N, "median")
            if N > 1:
                out[f"ewma|n|{N}"] = ewma(x, N)
        for T in TL:
            out[f"mean|t|{T}"] = roll_t(tc, x, T, "mean")
            out[f"median|t|{T}"] = roll_t(tc, x, T, "median")
    res = {k: (tc, v) for k, v in out.items()}
    if wave is not None:                       # (t0, 0.1 s bin sums, counts)
        t0, s, n = wave
        cs, cn = np.r_[0.0, np.cumsum(s)], np.r_[0, np.cumsum(n)]
        tg = t0 + 0.1 * (np.arange(s.size) + 1)
        for T in T_WAVE:
            k = int(round(T / 0.1))
            j = np.arange(s.size) + 1
            j0 = np.maximum(j - k, 0)
            num, den = cs[j] - cs[j0], cn[j] - cn[j0]
            y = np.where(den > 0.6 * k * (n.sum() / max(np.count_nonzero(n), 1)), num / np.maximum(den, 1), np.nan)
            res[f"wave_mean|t|{T}"] = (tg, y)
    return res


# ----------------------------------------------------------------------------------------------- likelihood

def point_lik(v, Y, q, quant, bias, sigma, span):
    if quant == "round":
        hi, lo = v[:, None] + q / 2, v[:, None] - q / 2
    else:
        hi, lo = v[:, None] + q, v[:, None]
    m = Y + bias
    P = ndtr((hi - m) / sigma) - ndtr((lo - m) / sigma)
    P = np.where(np.isfinite(P), P, 0.0)
    return (1 - EPS) * P + EPS * q / span


def lookup(tc, y, Q):
    i = np.searchsorted(tc, Q, side="right") - 1
    ok = i >= 0
    ic = np.clip(i, 0, tc.size - 1)
    Y = np.where(ok & (Q - tc[ic] <= STALE), y[ic], np.nan)
    return Y


def fit(ot, ov, og, tc, y, q, quant, free_bias, span, D, dd, extra=None):
    """Phase-averaged log-likelihood per delay in D. og = per-record phase width (s)."""
    M = int(np.ceil(og.max() / dd)) + 1
    Dext = D[0] + dd * np.arange(D.size + M)
    Q = ot[:, None] - Dext[None, :]
    Y = lookup(tc, y, Q)
    R = ov[:, None] - Y
    if free_bias:
        b = np.nanmedian(R, axis=0)
    else:
        b = np.zeros(Dext.size)
    b = np.where(np.isfinite(b), b, 0.0)
    s = 1.4826 * np.nanmedian(np.abs(R - b[None, :]), axis=0)
    s = np.where(np.isfinite(s), np.maximum(s, 0.25 * q), 10 * q)
    L = point_lik(ov, Y, q, quant, b[None, :], s[None, :], span)
    cs = np.concatenate([np.zeros((L.shape[0], 1)), np.cumsum(L, axis=1)], axis=1)
    m = np.clip(np.round(og / dd).astype(int), 1, M)
    j = np.arange(D.size)[None, :]
    rows = np.arange(L.shape[0])[:, None]
    La = (cs[rows, j + m[:, None]] - cs[rows, j]) / m[:, None]
    ll = np.log(La).sum(axis=0)
    # bias and sigma reported at the middle of each phase interval
    mid = np.clip(np.arange(D.size) + int(np.median(m)) // 2, 0, Dext.size - 1)
    return ll, b[mid], s[mid]


def best_of(ll):
    k = int(np.nanargmax(ll))
    return k, float(ll[k])


# ----------------------------------------------------------------------------------------------- reference series

def beat_series(te, xe, fs, art, cvp):
    ok = np.isfinite(xe)
    xe2 = np.where(ok, xe, np.nanmedian(xe))
    rp = r_peaks(xe2, fs)
    del xe2
    R = te[rp]
    rr = np.diff(R)
    good = (rr >= 0.25) & (rr <= 2.5)
    out = {"rr": (R[1:][good], rr[good]), "n_r": int(R.size)}
    for key, tr in (("art", art), ("cvp", cvp)):
        if tr is None:
            continue
        t, x = tr
        idx = np.searchsorted(t, R + 0.1)
        okb = (idx[1:] > idx[:-1]) & (idx[1:] < x.size) & good
        okb &= (t[np.minimum(idx[1:], x.size - 1)] - t[np.minimum(idx[:-1], x.size - 1)]) < 3.0
        xf = np.nan_to_num(x, nan=0.0)
        fin = np.isfinite(x).astype(float)
        st = np.clip(idx[:-1], 0, x.size - 1)
        en = np.clip(idx[1:], 0, x.size - 1)
        csx, csn = np.r_[0.0, np.cumsum(xf)], np.r_[0.0, np.cumsum(fin)]
        mean = (csx[en] - csx[st]) / np.maximum(csn[en] - csn[st], 1)
        tcb = R[1:] + 0.1
        if key == "art":
            xm = np.where(np.isfinite(x), x, -np.inf)
            xn = np.where(np.isfinite(x), x, np.inf)
            sb = np.full(st.size, np.nan)
            db = np.full(st.size, np.nan)
            w = np.flatnonzero(okb)
            if w.size:
                ix = np.r_[st[w], en[w][-1]]
                # reduceat over consecutive windows: build a sorted unique start list
                mx = np.maximum.reduceat(xm, st[w])
                mn = np.minimum.reduceat(xn, st[w])
                # reduceat runs from st[w][i] to st[w][i+1]; correct where the window end differs
                nxt = np.r_[st[w][1:], x.size]
                same = nxt == en[w]
                sb[w[same]], db[w[same]] = mx[same], mn[same]
                for i in w[~same]:
                    seg = x[st[i] : en[i]]
                    if seg.size:
                        sb[i], db[i] = np.nanmax(seg), np.nanmin(seg)
                del ix
            v = okb & (sb > 30) & (sb < 300) & (db > 5) & (sb - db > 8) & (mean > db) & (mean < sb)
            out["sbp"] = (tcb[v], sb[v])
            out["dbp"] = (tcb[v], db[v])
            out["mbp"] = (tcb[v], mean[v])
        else:
            v = okb & (mean > -10) & (mean < 40)
            out["cvp"] = (tcb[v], mean[v])
    return out


def wave_bins(t, x, t0, n):
    b = np.floor((t - t0) / 0.1).astype(np.int64)
    k = (b >= 0) & (b < n) & np.isfinite(x)
    return t0, np.bincount(b[k], x[k], minlength=n), np.bincount(b[k], minlength=n).astype(float)


def co2_breaths(t, x, fs):
    n = int(30 * fs)
    up, dn = [], []
    for s in range(0, x.size, n):
        seg = x[s : s + n]
        if seg.size < fs * 10:
            continue
        lo, hi = np.nanpercentile(seg, [10, 90])
        if not np.isfinite(hi) or hi - lo < 10:
            continue
        lev = lo + 0.5 * (hi - lo)
        a, b = seg[:-1], seg[1:]
        up += list(t[s + np.flatnonzero((a < lev) & (b >= lev)) + 1])
        dn += list(t[s + np.flatnonzero((a >= lev) & (b < lev)) + 1])
    up, dn = np.unique(up), np.unique(dn)
    if up.size < 3:
        return None
    up = up[np.r_[True, np.diff(up) > 1.0]]
    j = np.searchsorted(dn, up, side="right")
    ok = (j < dn.size)
    nxt = np.r_[up[1:], np.inf]
    ok &= np.where(ok, dn[np.minimum(j, dn.size - 1)] < nxt, False)
    ti, et, tf = [], [], []
    for i in np.flatnonzero(ok):
        f = dn[j[i]]
        m = (t >= up[i]) & (t <= f)
        if m.sum() < 3:
            continue
        e = np.nanmax(x[m])
        if 5 <= e <= 100:
            ti.append(i)
            et.append(e)
            tf.append(f)
    iv = np.diff(up)
    okr = (iv >= 1.0) & (iv <= 20.0)
    return {"etco2": (np.array(tf), np.array(et)), "rr_int": (up[1:][okr], iv[okr])}


def awp_breaths(t, x, fs):
    on = breath_onsets(t, x, fs)
    if on.size < 3:
        return None
    pt, pv, et, ev = [], [], [], []
    i0 = np.searchsorted(t, on)
    for k in range(on.size - 1):
        dur = on[k + 1] - on[k]
        if dur < 1.5 or dur > 20:
            continue
        a, b = i0[k], i0[k + 1]
        seg = x[a:b]
        if seg.size < 10:
            continue
        j = int(np.nanargmax(seg))
        pip = seg[j]
        peep = float(np.nanmedian(x[max(b - int(0.5 * fs), a) : b]))
        half = peep + 0.5 * (pip - peep)
        after = np.flatnonzero(seg[j:] < half)
        tend = t[a + j + (after[0] if after.size else 0)]
        if 3 <= pip <= 80:
            pt.append(tend)
            pv.append(pip)
        if 0 <= peep <= 40:
            et.append(on[k + 1])
            ev.append(peep)
    return {"pip": (np.array(pt), np.array(pv)), "peep": (np.array(et), np.array(ev))}


def primus_to_recorder(tc, starts, ppm):
    s = np.r_[-np.inf, np.sort(starts)]
    k = np.searchsorted(s, tc, side="right") - 1
    base = np.where(np.isfinite(s[k]), s[k], tc[0] if tc.size else 0.0)
    return tc + ppm * 1e-6 * (tc - base)


# ----------------------------------------------------------------------------------------------- per track

def numeric_obs(tr, lo, hi, cap=None):
    t = np.asarray(tr.record_times, dtype=float)
    v = np.asarray(tr.physical(), dtype=float)[: t.size]
    o = np.argsort(t, kind="stable")
    t, v = t[o], v[o]
    keep = np.r_[True, np.diff(t) > 0.05]
    t, v = t[keep], v[keep]
    g = np.r_[np.nan, np.diff(t)]
    ok = np.isfinite(v) & (v >= lo) & (v <= hi) & np.isfinite(g) & (g < 15)
    return t[ok], v[ok], g[ok]


def subsample(ot, ov, og, cap):
    if ot.size <= cap:
        return ot, ov, og
    k = np.linspace(0, ot.size - 1, cap).astype(int)
    return ot[k], ov[k], og[k]


def step_res(v):
    d = np.abs(np.diff(v))
    d = np.round(d[d > 0], 3)
    if not d.size:
        return 1.0
    u, c = np.unique(d, return_counts=True)
    return float(u[np.argmax(c)])


def analyse_track(name, ot, ov, og, cands, *, solar, span, ev_int=None):
    q = step_res(ov)
    rate = name in RATE
    quants = ("round", "floor") if rate else ("round",)
    free = not rate
    D = np.round(np.arange(-3.0, 15.0001, 0.2), 3)
    s_t, s_v, s_g = subsample(ot, ov, og, 3000)
    rows = []
    for cn, (tc, y) in cands.items():
        okc = np.isfinite(y)
        if okc.sum() < 50:
            continue
        tcc, yc = tc[okc], y[okc]
        for qm in quants:
            ll, b, s = fit(s_t, s_v, s_g, tcc, yc, q, qm, free, span, D, 0.2)
            k, L = best_of(ll)
            rows.append({"cand": cn, "quant": qm, "ll": L, "d": float(D[k]), "b": float(b[k]), "s": float(s[k])})
    if not rows:
        return None
    rows.sort(key=lambda r: -r["ll"])
    # refine the top 3 on all records (<= 8000), 0.05 s delay steps
    a_t, a_v, a_g = subsample(ot, ov, og, 8000)
    top = []
    for r in rows[:3]:
        tc, y = cands[r["cand"]]
        okc = np.isfinite(y)
        Dr = np.round(np.arange(r["d"] - 0.6, r["d"] + 0.6001, 0.05), 3)
        ll, b, s = fit(a_t, a_v, a_g, tc[okc], y[okc], q, r["quant"], free, span, Dr, 0.05)
        k, L = best_of(ll)
        top.append({**r, "ll_all": round(L, 2), "d": float(Dr[k]), "b": round(float(b[k]), 3), "s": round(float(s[k]), 3),
                    "n_all": int(a_t.size)})
    top.sort(key=lambda r: -r["ll_all"])
    best = top[0]
    fam = {}
    for r in rows:
        fn, wt, _ = r["cand"].split("|")
        key = f"{fn}|{wt}|{r['quant']}"
        if key not in fam:
            fam[key] = {"cand": r["cand"], "ll": round(r["ll"], 2), "d": r["d"], "b": round(r["b"], 3), "s": round(r["s"], 3)}
    out = {"n": int(ot.size), "n_fit": int(s_t.size), "q": q, "gap_median": float(np.median(og)),
           "best": best, "top": top, "families": fam, "ll_best_stage1": round(rows[0]["ll"], 2)}
    if rate:
        lr = {qm: max(r["ll"] for r in rows if r["quant"] == qm) for qm in quants}
        out["round_minus_floor_ll"] = round(lr["round"] - lr["floor"], 2)
    # delay by tertile of the local event interval, for the best n- and t-window candidates
    if ev_int is not None:
        et, ei = ev_int
        loc = np.interp(ot, et, ndimage_median(ei, 9)) if et.size > 20 else None
        if loc is not None:
            terc = np.nanpercentile(loc, [33.3, 66.7])
            out["tertiles"] = {}
            for wt in ("n", "t"):
                cand = [r for r in rows if r["cand"].split("|")[1] == wt]
                if not cand:
                    continue
                r0 = cand[0]
                tc, y = cands[r0["cand"]]
                okc = np.isfinite(y)
                Dr = np.round(np.arange(r0["d"] - 3.0, r0["d"] + 3.0001, 0.1), 3)
                res = []
                for lo_, hi_ in ((-np.inf, terc[0]), (terc[0], terc[1]), (terc[1], np.inf)):
                    m = (loc > lo_) & (loc <= hi_)
                    if m.sum() < 100:
                        res.append(None)
                        continue
                    tt, vv, gg = subsample(ot[m], ov[m], og[m], 3000)
                    ll, _, _ = fit(tt, vv, gg, tc[okc], y[okc], q, r0["quant"], free, span, Dr, 0.1)
                    k, _ = best_of(ll)
                    res.append({"interval_s": round(float(np.median(loc[m])), 3), "d": float(Dr[k]), "n": int(m.sum())})
                out["tertiles"][wt] = {"cand": r0["cand"], "by_tertile": res}
    # repeats against the predicted change (best candidate, middle of the phase interval)
    tc, y = cands[best["cand"]]
    okc = np.isfinite(y)
    pred = lookup(tc[okc], y[okc], ot - best["d"] - og / 2)
    cons = np.r_[False, (og[1:] < 1.5 * np.median(og))]
    dp = np.abs(np.r_[np.nan, np.diff(pred)])
    rep = np.r_[False, np.diff(ov) == 0]
    edges = [0, 0.5, 1, 2, 4, 1e9]
    rt = []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = cons & np.isfinite(dp) & (dp / q >= lo_) & (dp / q < hi_)
        rt.append([int(m.sum()), int((m & rep).sum())])
    out["repeats_by_pred_change_q"] = {"edges": edges[:-1], "n_rep": rt}
    if solar:
        out["hold"] = hold_test(ot, ov, og, tc[okc], y[okc], best, q, span, free)
    return out


def ndimage_median(x, k):
    from scipy import ndimage
    return ndimage.median_filter(x, size=k, mode="nearest")


def hold_test(ot, ov, og, tc, y, best, q, span, free):
    """Value refreshed every U s at phase phi (per 30 min block), shown until the next refresh."""
    tt, vv, gg = subsample(ot, ov, og, 4000)
    b, s, quant = best["b"] if free else 0.0, max(best["s"], 0.25 * q), best["quant"]
    blocks = np.floor((tt - tt[0]) / 1800).astype(int)
    nb = int(blocks.max()) + 1
    # baseline: the uniform-phase model on the same records
    D0 = np.array([best["d"]])
    ll0, _, _ = fit(tt, vv, gg, tc, y, q, quant, free, span, D0, 0.05)
    res = {"ll_uniform": round(float(ll0[0]), 2), "n": int(tt.size), "blocks": nb, "U": []}
    n = tt.size
    for U in (2, 4, 6, 8, 10, 12):
        phis = np.arange(0, U, 0.2)
        best_U = None
        for d in np.arange(best["d"] - U - 1.0, best["d"] + 1.0001, 0.2):
            upd = phis[None, :] + U * np.floor((tt[:, None] - phis[None, :]) / U)
            Y = lookup(tc, y, upd - d)
            L = np.log(point_lik(vv, Y, q, quant, b, s, span))
            tot = 0.0
            for k in range(nb):
                m = blocks == k
                if m.any():
                    tot += float(L[m].sum(axis=0).max())
            if best_U is None or tot > best_U[0]:
                best_U = (tot, float(d))
        bic_gain = 2 * (best_U[0] - float(ll0[0])) - (nb + 1) * np.log(n)
        res["U"].append({"U": U, "ll": round(best_U[0], 2), "d": round(best_U[1], 2), "delta_bic": round(bic_gain, 1)})
    return res


# ----------------------------------------------------------------------------------------------- main

def run_case(root, c, u, step_rates):
    names = set(SOLAR.values()) | set(PRIMUS.values()) | set(WAVES.values()) | {"SNUADC/ECG_V5"}
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names=names)
    res = {"case": c, "tracks": {}}
    W = {}
    for k in ("ecg", "art", "cvp"):
        tr = v.tracks.get(WAVES[k])
        if k == "ecg" and (tr is None or not tr.record_times.size):
            tr = v.tracks.get("SNUADC/ECG_V5")
        if tr is None or not tr.record_times.size:
            continue
        t, clk = snu_times(tr, u, step_rates)
        W[k] = (t, tr.physical().astype(np.float64)[: t.size], float(tr.srate))
        if k == "ecg":
            res["snuadc_clock"] = {"rate_ppm": round(clk["rate_ppm"], 1), "source": clk["rate_source"]}
    # Solar side
    if "ecg" in W:
        te, xe, fs = W["ecg"]
        bs = beat_series(te, xe, fs, W.get("art", (None, None))[:2] if "art" in W else None,
                         W["cvp"][:2] if "cvp" in W else None)
        res["beats"] = bs["n_r"]
        t0 = float(np.floor(te[0]))
        nbin = int((te[-1] - t0) / 0.1) + 1
        plan = [("HR", "rate_beat", bs.get("rr"), None)]
        if "art" in W:
            wb = wave_bins(W["art"][0], W["art"][1], t0, nbin)
            plan += [("ART_SBP", "value_beat", bs.get("sbp"), None), ("ART_DBP", "value_beat", bs.get("dbp"), None),
                     ("ART_MBP", "value_beat", bs.get("mbp"), wb)]
        if "cvp" in W:
            wb = wave_bins(W["cvp"][0], W["cvp"][1], t0, nbin)
            plan += [("CVP", "value_beat", bs.get("cvp"), wb)]
        for name, kind, ser, wave in plan:
            tr = v.tracks.get(SOLAR[name])
            if tr is None or ser is None or ser[0].size < 200:
                continue
            ot, ov, og = numeric_obs(tr, *VALID[name])
            ot_m = (ot > ser[0][0] + 60) & (ot < ser[0][-1])
            ot, ov, og = ot[ot_m], ov[ot_m], og[ot_m]
            if ot.size < 300:
                continue
            cands = candidates(kind, ser[0], ser[1], wave)
            iv = bs["rr"]
            t1 = time.time()
            r = analyse_track(name, ot, ov, og, cands, solar=True, span=VALID[name][1] - VALID[name][0], ev_int=iv)
            if r:
                r["secs"] = round(time.time() - t1, 1)
                res["tracks"][name] = r
        del bs
    W.clear()
    # Primus side
    aw, co = v.tracks.get(WAVES["awp"]), v.tracks.get(WAVES["co2"])
    for key, tr in (("awp", aw), ("co2", co)):
        if tr is None or not tr.record_times.size:
            continue
        t = sample_times(tr)
        x = np.nan_to_num(tr.physical().astype(np.float64))[: t.size]
        starts = stretches_of(tr)
        ser = awp_breaths(t, x, float(tr.srate)) if key == "awp" else co2_breaths(t, x, float(tr.srate))
        if ser is None:
            continue
        plan = [("PIP", "value_breath", "pip"), ("PEEP", "value_breath", "peep")] if key == "awp" else \
               [("ETCO2", "value_breath", "etco2"), ("RR_CO2", "rate_breath", "rr_int")]
        for name, kind, sk in plan:
            ntr = v.tracks.get(PRIMUS[name])
            tc0, xv = ser[sk]
            if ntr is None or tc0.size < 100:
                continue
            ot, ov, og = numeric_obs(ntr, *VALID[name])
            span = VALID[name][1] - VALID[name][0]
            out = None
            t1 = time.time()
            for ppm in (PRIMUS_PPM,):
                tc = primus_to_recorder(tc0, starts, ppm)
                m = (ot > tc[0] + 60) & (ot < tc[-1])
                if m.sum() < 150:
                    break
                cands = candidates(kind, tc, xv)
                iv = ser["rr_int"] if "rr_int" in ser else (tc[1:], np.diff(tc))
                out = analyse_track(name, ot[m], ov[m], og[m], cands, solar=False, span=span,
                                    ev_int=(primus_to_recorder(iv[0], starts, ppm), iv[1]))
            if out:
                # Primus drift: refine on the best candidate
                bc = out["best"]
                dr = []
                for ppm in (0.0, 60.0, 115.0, 170.0):
                    tc = primus_to_recorder(tc0, starts, ppm)
                    cands = candidates(kind, tc, xv)
                    tcc, y = cands[bc["cand"]]
                    okc = np.isfinite(y)
                    m = (ot > tc[0] + 60) & (ot < tc[-1])
                    Dr = np.round(np.arange(bc["d"] - 1.0, bc["d"] + 1.0001, 0.1), 3)
                    ll, _, _ = fit(*subsample(ot[m], ov[m], og[m], 8000), tcc[okc], y[okc], out["q"], bc["quant"],
                                   name not in RATE, span, Dr, 0.1)
                    k, L = best_of(ll)
                    dr.append({"ppm": ppm, "ll": round(L, 2), "d": float(Dr[k])})
                out["primus_drift"] = dr
                out["primus_stretches"] = int(len(starts) + 1)
                out["secs"] = round(time.time() - t1, 1)
                res["tracks"][name] = out
    return res


def select(root, path, n, seed, out):
    cs = [int(x) for x in open(path).read().replace("\n", ",").split(",") if x.strip()]
    random.Random(seed).shuffle(cs)
    need = {"Solar8000/CVP", "Solar8000/ART_MBP", "Solar8000/HR", "Primus/ETCO2", "Primus/PIP_MBAR", "SNUADC/ECG_II"}
    got = []
    t0 = time.time()
    for c in cs:
        if len(got) >= n or time.time() - t0 > 150:
            break
        try:
            v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names=need)
        except Exception:
            continue
        ok = all(k in v.tracks and v.tracks[k].record_times.size > 600 for k in need)
        if ok:
            cv = np.asarray(v.tracks["Solar8000/CVP"].physical(), float)
            ok = np.mean((cv > 0) & (cv < 30)) > 0.5
        if ok:
            got.append(c)
    Path(out).write_text(",".join(map(str, sorted(got))))
    print(len(got), "selected")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases")
    ap.add_argument("--units")
    ap.add_argument("--step-rates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--select-from")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    if a.select_from:
        select(a.root, a.select_from, a.n, a.seed, a.out)
        return
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    step_rates = json.loads(Path(a.step_rates).read_text())
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in Path(a.out).read_text().splitlines() if l.strip()}
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        t1 = time.time()
        try:
            res = run_case(a.root, c, U.get(c, {}), step_rates)
        except Exception as e:  # noqa: BLE001
            res = {"case": c, "error": repr(e)[:300]}
        res["secs"] = round(time.time() - t1, 1)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
