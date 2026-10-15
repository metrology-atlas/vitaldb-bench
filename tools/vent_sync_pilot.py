# SPDX-License-Identifier: MIT
"""Ventilator (Primus) against SNUADC: breath-locked timing from CVP, R-R, R amplitude and arterial pressure.
Pilot, per case, this machine only.

    python tools/vent_sync_pilot.py --root R --cases 1,2 --units units.csv --step-rates rates.json --out out.jsonl

SNUADC times are corrected with the stamp model (physiosync.stamp_clock, rules as tools/sa_corrected.py);
Primus times are its own stamps. Per case:

1. Breath onsets in Primus/AWP: rising crossings of 30% between the local 10th and 90th percentile (30 s
   blocks), confirmed by reaching 70% within 1.5 s; mechanical breaths only (amplitude >= 4 hPa).
2. Coarse lag per 60 min (+-30 s): cross-correlation of the slow respiratory envelopes of AWP and CVP
   (0.1-0.7 Hz band, 20 s RMS, log, less a 5 min median): changes in ventilation are not periodic, so this
   fixes which breath is which.
3. Fine lag per 10 min window (step 5 min): breath-onset-triggered averages of CVP (SNUADC, 50 Hz grid) and
   of AWP (Primus); lag = the shift that best correlates the two averaged curves, searched within half a
   breath of the previous good window's lag (tracking, not across a Primus
   stamp jump), else of the nearest coarse lag; precision from the split of the breaths into odd and even halves.
4. Breath-phase modulation per window of beat series: R-R, R amplitude, systolic and pulse pressure. Each beat
   gets its phase in the breath (Primus onsets); a first-harmonic fit gives amplitude, R^2 and phase, the
   phase as a time lag (mod one breath; unwrapped over the case).
5. Breath matching (breath_match): per 30 min block of a Primus stretch, which breath is which, from the
   breath-to-breath size of the CVP (and, independently, ART) response against peak airway pressure.
6. Per signal: lag against time, one slope (ppm) shared by the Primus stretches (stamp jumps > 1 s) with one
   offset per stretch; residual SD.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

sys.path.insert(0, str(Path(__file__).parent))
from pat import r_peaks, sample_times
from physiosync import corrected_times, stamp_clock
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

warnings.filterwarnings("ignore")
SNU = {"ecg": "SNUADC/ECG_II", "cvp": "SNUADC/CVP", "art": "SNUADC/ART"}
G = 0.02  # fine grid (s)


def snu_times(tr, u, step_rates):
    st = tr.record_offsets[: tr.record_times.size]
    clk = stamp_clock(tr.record_times, st, float(tr.srate), correction_sign=-1)
    if clk["rate_source"] == "fit":
        ev = clk["corrections"]["median_event_s"]
        fall = step_rates.get(u.get("snuadc_port"), step_rates.get("*", 0.0)) if ev is not None and ev <= -0.4 else 0.0
        clk = stamp_clock(tr.record_times, st, float(tr.srate), rate_ppm=fall, correction_sign=-1)
    return corrected_times(clk, st, tr.record_sizes[: st.size]), clk


def binned(t, y, t0, dt, n):
    b = np.floor((t - t0) / dt).astype(np.int64)
    k = (b >= 0) & (b < n)
    s = np.bincount(b[k], y[k], minlength=n)
    c = np.bincount(b[k], minlength=n)
    with np.errstate(invalid="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def breath_onsets(t, x, fs):
    n = int(30 * fs)
    on = []
    for s in range(0, x.size, n):
        seg = x[s : s + n]
        if seg.size < fs * 10:
            continue
        lo, hi = np.percentile(seg, [10, 90])
        if hi - lo < 4:
            continue
        lev, conf = lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo)
        i = np.flatnonzero((seg[:-1] < lev) & (seg[1:] >= lev))
        for j in i:
            if np.max(seg[j + 1 : j + 1 + int(1.5 * fs)], initial=-np.inf) < conf:
                continue
            f = (lev - seg[j]) / (seg[j + 1] - seg[j])
            on.append(t[s + j] + f * (t[s + j + 1] - t[s + j]))
    on = np.array(on)
    if on.size:
        on = on[np.r_[True, np.diff(on) > 1.5]]
    return on


def stretches_of(tr):
    rt = tr.record_times
    res = rt - rt[0] - tr.record_offsets[: rt.size] / float(tr.srate)
    j = np.flatnonzero(np.abs(np.diff(res)) > 1.0)
    return rt[j + 1]  # stamp times where a new Primus stretch starts


def env_series(y):
    sos = signal.butter(2, (0.1, 0.7), btype="bandpass", fs=10, output="sos")
    ok = np.isfinite(y)
    if ok.mean() < 0.3:
        return None
    f = np.interp(np.arange(y.size), np.flatnonzero(ok), y[ok])
    f = signal.sosfiltfilt(sos, f)
    e = np.log(np.sqrt(ndimage.uniform_filter1d(f * f, 200)) + 1e-9)
    e = e - ndimage.median_filter(e, 3001)
    e[~ok] = np.nan
    return e


def xlag(a, b, L):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.mean() < 0.5:
        return None, None
    a = np.where(ok, a - np.nanmean(a[ok]), 0)
    b = np.where(ok, b - np.nanmean(b[ok]), 0)
    if a.std() == 0 or b.std() == 0:
        return None, None
    F = 1 << int(np.ceil(np.log2(2 * a.size)))
    r = np.fft.irfft(np.conj(np.fft.rfft(a, F)) * np.fft.rfft(b, F), F)
    r = np.r_[r[-L:], r[: L + 1]] / (a.std() * b.std() * a.size)
    i = int(np.argmax(r))
    return (i - L), float(r[i])


def trig_avg(grid, t0, onsets, taus):
    idx = np.round((onsets[:, None] + taus[None, :] - t0) / G).astype(np.int64)
    ok = (idx >= 0) & (idx < grid.size)
    v = np.full(idx.shape, np.nan)
    v[ok] = grid[idx[ok]]
    v = v - np.nanmean(v[:, (taus > -0.6) & (taus < -0.1)], axis=1, keepdims=True)
    return np.nanmean(v, axis=0)


def steepest(avg, taus, lo, hi):
    d = np.gradient(ndimage.uniform_filter1d(np.nan_to_num(avg), 5), G)
    m = (taus >= lo) & (taus <= hi)
    if m.sum() < 3:
        return None, None
    i = np.flatnonzero(m)[int(np.argmax(d[m]))]
    if 0 < i < d.size - 1:
        y0, y1, y2 = d[i - 1], d[i], d[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        off = 0.0
    noise = np.std(d[~m]) if (~m).sum() > 5 else np.nan
    return float(taus[i] + off * G), float(d[i] / noise) if noise and np.isfinite(noise) and noise > 0 else None


def shape_lag(aa, cv, taus, lo, hi):
    """Lag (s) that best maps the AWP average onto the CVP average (correlation of the two curves, 0.2 s
    smoothing), searched in [lo, hi]; returns (lag, r)."""
    a = ndimage.uniform_filter1d(np.nan_to_num(aa), 10)
    c = ndimage.uniform_filter1d(np.nan_to_num(cv), 10)
    best = (None, -2.0)  # noqa: F841 - unused; kept as written
    rs = []
    lags = np.arange(int(np.floor(lo / G)), int(np.ceil(hi / G)) + 1)
    core = (taus >= -1.0) & (taus <= 4.0)
    ia = np.flatnonzero(core)
    for L in lags:
        ic = ia + L
        if ic[0] < 0 or ic[-1] >= c.size:
            rs.append(np.nan)
            continue
        r = np.corrcoef(a[ia], c[ic])[0, 1]
        rs.append(r)
    rs = np.array(rs)
    if not np.isfinite(rs).any():
        return None, None
    i = int(np.nanargmax(rs))
    off = 0.0
    if 0 < i < rs.size - 1 and np.isfinite(rs[i - 1]) and np.isfinite(rs[i + 1]):
        den = rs[i - 1] - 2 * rs[i] + rs[i + 1]
        off = 0.5 * (rs[i - 1] - rs[i + 1]) / den if den != 0 else 0.0
    return float((lags[i] + off) * G), float(rs[i])


def harmonic(beat_t, y, onsets):
    k = np.searchsorted(onsets, beat_t) - 1
    ok = (k >= 0) & (k < onsets.size - 1) & np.isfinite(y)
    if ok.sum() < 30:
        return None
    k, bt, yy = k[ok], beat_t[ok], y[ok]
    T = onsets[k + 1] - onsets[k]
    good = (T > 1.5) & (T < 12)
    k, bt, yy, T = k[good], bt[good], yy[good], T[good]
    if yy.size < 30:
        return None
    ph = 2 * np.pi * (bt - onsets[k]) / T
    yy = yy - ndimage.median_filter(yy, 31)
    X = np.c_[np.ones_like(ph), np.cos(ph), np.sin(ph)]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    fit = X @ beta
    r2 = 1 - np.sum((yy - fit) ** 2) / np.sum((yy - yy.mean()) ** 2)
    ang = np.arctan2(beta[2], beta[1])  # phase of the maximum
    Tm = float(np.median(T))
    return {"amp": float(np.hypot(beta[1], beta[2])), "r2": float(r2), "lag_s": float(ang / (2 * np.pi) * Tm), "T": Tm, "n": int(yy.size)}


def fit_slope(tt, lag, stretch_starts, w=None):
    if len(tt) < 4:
        return None
    tt, lag = np.asarray(tt), np.asarray(lag)
    w = np.ones_like(tt) if w is None else np.asarray(w)
    seg = np.searchsorted(stretch_starts, tt)
    X = np.c_[tt / 3600.0, (seg[:, None] == np.unique(seg)[None, :]).astype(float)]
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], lag * sw, rcond=None)
    res = lag - X @ beta
    return {"slope_ms_h": float(beta[0] * 1000), "ppm": float(beta[0] * 1e6 / 3600),
            "resid_sd_ms": float(np.sqrt(np.average(res ** 2, weights=w)) * 1000), "n": int(len(tt)), "stretches": int(np.unique(seg).size)}


def _hp(x, k=9):
    m = np.isfinite(x)
    if m.sum() < k:
        return x * np.nan
    f = np.interp(np.arange(x.size), np.flatnonzero(m), x[m])
    out = f - ndimage.median_filter(f, k)
    out[~m] = np.nan
    return out


def _grid_mean(grid, t0, a, b):
    """Mean of a 50 Hz grid over [a_i, b_i) for arrays of absolute times."""
    ia = np.clip(np.round((a - t0) / G).astype(np.int64), 0, grid.size - 1)
    ib = np.clip(np.round((b - t0) / G).astype(np.int64), 0, grid.size - 1)
    c = np.r_[0.0, np.cumsum(np.nan_to_num(grid))]
    n = np.r_[0, np.cumsum(np.isfinite(grid))]
    cnt = n[ib] - n[ia]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[ib] - c[ia]) / cnt, np.nan)


def breath_match(on, awp50, ref50, t0, rows_cvp, sst, *, block_s=1800, kmax=3):
    """Which breath is which. For each 30 min block of one Primus stretch, the tracked lag l(t) (good CVP windows,
    interpolated) places each breath's response in the SNUADC signal; the breath-to-breath size of that
    response (high-passed over 9 breaths) is correlated with the breath-to-breath peak airway pressure of
    breath i+k for k in -kmax..kmax. If the tracking is k breaths off, the correlation peaks at k; the lag is
    then l - median(o[i+k] - o[i]). Returns one row per block: [mid, stretch, k, c_best, c_second, n, shift_ms]."""
    g = np.array([r for r in rows_cvp if r[2] >= 0.8 and r[3] is not None and r[3] < 100], dtype=float)
    if g.size == 0 or on.size < 50:
        return []
    o = on - t0
    seg_on = np.searchsorted(sst, o)
    seg_g = np.searchsorted(sst, g[:, 0])
    peak = _grid_mean(awp50, t0, on + 0.3, on + 1.2) - _grid_mean(awp50, t0, on - 0.5, on - 0.05)
    out = []
    for sidx in np.unique(seg_on):
        gs = g[seg_g == sidx]
        if gs.shape[0] < 2:
            continue
        ib = np.flatnonzero(seg_on == sidx)
        for b0 in np.arange(o[ib[0]], o[ib[-1]], block_s):
            ii = ib[(o[ib] >= b0) & (o[ib] < b0 + block_s)]
            if ii.size < 150:
                continue
            lag = np.interp(o[ii], gs[:, 0], gs[:, 1] / 1000.0)
            resp = _grid_mean(ref50, t0, on[ii] + lag + 0.2, on[ii] + lag + 1.5) - _grid_mean(ref50, t0, on[ii] + lag - 0.6, on[ii] + lag - 0.1)
            r = _hp(resp)
            a = _hp(peak)
            cs = []
            for k in range(-kmax, kmax + 1):
                j = ii + k
                ok = (j >= 0) & (j < on.size)
                x, y = r[ok], a[j[ok]]
                m = np.isfinite(x) & np.isfinite(y)
                cs.append(np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 50 and np.std(x[m]) > 0 and np.std(y[m]) > 0 else np.nan)
            cs = np.array(cs)
            if not np.isfinite(cs).any():
                continue
            kb = int(np.nanargmax(cs))
            srt = np.sort(cs[np.isfinite(cs)])
            k = kb - kmax
            j = np.clip(ii + k, 0, on.size - 1)
            shift = float(np.median(on[j] - on[ii])) * 1000
            out.append([round(float(b0 + block_s / 2), 1), int(sidx), k, round(float(srt[-1]), 3),
                        round(float(srt[-2]) if srt.size > 1 else np.nan, 3), int(ii.size), round(-shift, 1)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--step-rates", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    step_rates = json.loads(Path(a.step_rates).read_text())
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in Path(a.out).read_text().splitlines() if l.strip()}
    with open(a.out, "a") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            t1 = time.time()
            res: dict = {"case": c}
            try:
                v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={*SNU.values(), "Primus/AWP"})
                u = U.get(c, {})
                aw = v.tracks["Primus/AWP"]
                ta = sample_times(aw)
                xa = np.nan_to_num(aw.physical().astype(np.float64))
                on = breath_onsets(ta, xa, float(aw.srate))
                st_starts = stretches_of(aw)
                res["breaths"] = int(on.size)
                tracks = {}
                for k, n in SNU.items():
                    tr = v.tracks.get(n)
                    if tr is None or not tr.record_times.size:
                        continue
                    t, clk = snu_times(tr, u, step_rates)
                    x = tr.physical().astype(np.float64)
                    tracks[k] = (t, x, float(tr.srate))
                    if k == "ecg":
                        res["snuadc_clock"] = {"rate_ppm": round(clk["rate_ppm"], 1), "source": clk["rate_source"]}
                del v
                t0 = float(np.floor(min(ta[0], tracks["ecg"][0][0])))
                t_end = float(max(ta[-1], tracks["ecg"][0][-1]))
                n50 = int((t_end - t0) / G) + 1
                tc, xc, _ = tracks["cvp"]
                okc = np.isfinite(xc)
                cvp50 = binned(tc[okc], xc[okc], t0, G, n50)
                awp50 = binned(ta, xa, t0, G, n50)
                # coarse: 10 Hz envelopes, per 60 min
                cvp10 = np.nanmean(np.r_[cvp50, np.full((-n50) % 5, np.nan)].reshape(-1, 5), axis=1)
                awp10 = np.nanmean(np.r_[awp50, np.full((-n50) % 5, np.nan)].reshape(-1, 5), axis=1)
                ea, ec = env_series(awp10), env_series(cvp10)
                coarse = []
                if ea is not None and ec is not None:
                    W = 36000
                    for s in range(0, ea.size - W // 2, W // 2):
                        l, r = xlag(ea[s : s + W], ec[s : s + W], 300)
                        if l is not None:
                            coarse.append([round((s + W / 2) / 10, 1), l / 10, round(r, 3)])
                res["coarse"] = coarse
                cg = [x for x in coarse if x[2] >= 0.3]
                L0 = float(np.median([x[1] for x in cg])) if cg else None
                res["coarse_lag_s"] = L0
                # beat series
                te, xe, fs = tracks["ecg"]
                ok = np.isfinite(xe)
                xe2 = np.where(ok, xe, np.nanmedian(xe))
                rp = r_peaks(xe2, fs)
                bp = signal.sosfiltfilt(signal.butter(2, (5, 25), btype="bandpass", fs=fs, output="sos"), xe2)
                rt_ = te[rp]
                series = {"rr": (rt_[1:], np.diff(rt_)), "ramp": (rt_, np.abs(bp[rp]))}
                del bp, xe2
                if "art" in tracks:
                    tA, xA, fA = tracks["art"]
                    idx = np.searchsorted(tA, rt_)
                    sbp, dbp = [], []
                    for i in idx:
                        s0, s1 = i + int(0.1 * fA), i + int(0.5 * fA)
                        d0, d1 = max(i - int(0.15 * fA), 0), i + int(0.15 * fA)
                        if s1 < xA.size:
                            sbp.append(np.nanmax(xA[s0:s1])); dbp.append(np.nanmin(xA[d0:d1]))
                        else:
                            sbp.append(np.nan); dbp.append(np.nan)
                    sbp, dbp = np.array(sbp), np.array(dbp)
                    series["sbp"] = (rt_, sbp)
                    series["pp"] = (rt_, sbp - dbp)
                taus = np.arange(-4.0, 12.0, G)
                rows = {"cvp": [], **{k: [] for k in series}}
                prev = None
                sst_all = np.asarray(st_starts) - t0
                for ws in np.arange(on[0] if on.size else 0, (on[-1] if on.size else 0) - 600, 300):
                    o = on[(on >= ws) & (on < ws + 600)]
                    if o.size < 40:
                        continue
                    T = float(np.median(np.diff(o)))
                    mid = round(float(ws + 300 - t0), 1)
                    # search centre: the previous window's lag if it was good and recent (tracking), else the
                    # nearest coarse lag, else the case's
                    centre = None
                    same_stretch = prev is not None and not np.any((sst_all > prev[0] - 300) & (sst_all <= mid + 300))
                    if prev is not None and mid - prev[0] <= 900 and same_stretch:
                        centre = prev[1]
                    elif cg:
                        centre = min(cg, key=lambda x: abs(x[0] - mid))[1]
                    elif L0 is not None:
                        centre = L0
                    if centre is not None and T > 1.5:
                        lo, hi = centre - T / 2, centre + T / 2
                        aa = trig_avg(awp50, t0, o, taus)
                        lag, r = shape_lag(aa, trig_avg(cvp50, t0, o, taus), taus, lo, hi)
                        h1 = shape_lag(trig_avg(awp50, t0, o[0::2], taus), trig_avg(cvp50, t0, o[0::2], taus), taus, lo, hi)[0]
                        h2 = shape_lag(trig_avg(awp50, t0, o[1::2], taus), trig_avg(cvp50, t0, o[1::2], taus), taus, lo, hi)[0]
                        if lag is not None and r >= 0.8 and h1 is not None and h2 is not None and abs(h1 - h2) < 0.1:
                            prev = (mid, lag)
                        if lag is not None:
                            rows["cvp"].append([mid, round(lag * 1000, 1), round(r, 3),
                                                None if h1 is None or h2 is None else round(abs(h1 - h2) * 1000, 1), round(T, 2)])
                    for k, (bt, y) in series.items():
                        m = (bt >= ws) & (bt < ws + 600)
                        h = harmonic(bt[m], y[m], o)
                        if h:
                            rows[k].append([mid, round(h["lag_s"] * 1000, 1), round(h["r2"], 4), round(h["amp"], 5), round(h["T"], 2), h["n"]])
                res["rows"] = rows
                sstm = np.asarray(st_starts) - t0
                res["match_cvp"] = breath_match(on, awp50, cvp50, t0, rows["cvp"], sstm)
                if "art" in tracks:
                    tA_, xA_, _ = tracks["art"]
                    okA = np.isfinite(xA_)
                    art50 = binned(tA_[okA], xA_[okA], t0, G, n50)
                    res["match_art"] = breath_match(on, awp50, art50, t0, rows["cvp"], sstm)
                sst = np.asarray(st_starts) - t0
                fits = {}
                g = [r for r in rows["cvp"] if r[3] is not None and r[3] < 100 and (r[2] or 0) >= 0.8]
                f_ = fit_slope([r[0] for r in g], [r[1] / 1000 for r in g], sst) if g else None
                if f_:
                    f_["good_windows"] = len(g)
                    f_["lag_ms_median"] = float(np.median([r[1] for r in g]))
                    fits["cvp"] = f_
                for k in series:
                    rr = [r for r in rows[k] if r[2] >= 0.05]
                    if len(rr) >= 4:
                        T = np.median([r[4] for r in rr]) * 1000
                        lag = np.unwrap(np.array([r[1] for r in rr]) / T * 2 * np.pi) / (2 * np.pi) * T
                        f_ = fit_slope([r[0] for r in rr], lag / 1000, sst, [r[2] for r in rr])
                        if f_:
                            f_["good_windows"] = len(rr)
                            fits[k] = f_
                res["fits"] = fits
                res["primus_stretches"] = int(len(st_starts) + 1)
                res["stretch_starts_s"] = [round(float(x), 1) for x in sst]
                res["hours"] = round((t_end - t0) / 3600, 2)
            except Exception as ex:  # noqa: BLE001
                import traceback
                res["error"] = (f"{type(ex).__name__}: {ex} " + traceback.format_exc()[-300:])[:600]
            res["seconds"] = round(time.time() - t1, 1)
            fh.write(json.dumps(res) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
