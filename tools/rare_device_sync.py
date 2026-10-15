# SPDX-License-Identifier: MIT
"""Clock offset and drift of the rarely used VitalDB devices against the reference (SNUADC stamp-model time).
Per case, this machine only; aggregates leave.

    python tools/rare_device_sync.py --root R --cases 1,2 --units units.csv --step-rates rates.json --out out.jsonl
    python tools/rare_device_sync.py --agg out*.jsonl --summary summary.json

Reference: SNUADC samples on the stamp-model line (tools/vent_sync_pilot.snu_times). Lag = device time stamp
minus reference time of the same physiological event (+ = the device stamps late).

CardioQ (oesophageal Doppler; ABP wave at 180 Hz, the same arterial pressure as SNUADC/ART): its stamps
  follow its sample count but re-anchor often (hundreds of jumps per case), so windows lie inside stretches
  between jumps (>= 60 s, cut to <= 5 min). Beats matched one to one: systolic peaks in both (0.5-8 Hz band);
  per window a coarse lag from the
  cross-correlation of 1 Hz MAP series (+-60 s), then the beat shift (-3..3 beats) whose R-R series agree
  best, then lag = median of matched peak-time differences; precision = their MAD.
EV1000 (ART_MBP numeric, values change every ~20 s): 1 s MAP from SNUADC/ART; predicted value = mean over a
  20 s window ending L s before the stamp; L (-120..120 s) maximising the correlation of detrended values
  over 30-min windows, at the stamps where the value changes.
Vigileo (SVV, changes every ~20 s): as EV1000, predicted by the pulse-pressure variation of SNUADC/ART over the
  trailing 20 s (per-second pulse pressure, (max - min) / mean). Stroke volume against pulse pressure did not
  align (correlations 0.1-0.8, lags at the search limits).
Vigilance (HR_AVG): ECG heart rate (R peaks, 1 s), 60 s predictor window, L in -300..300 s.
Per device and case: lag per window, median offset, drift (ppm, line through the windows), residual SD,
windows used, and steps between consecutive windows larger than 5 x the residual SD.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from pat import r_peaks, sample_times  # noqa: E402
from vent_sync_pilot import snu_times  # noqa: E402
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: E402


def peaks(t, x, fs):
    ok = np.isfinite(x)
    if ok.mean() < 0.5:
        return np.zeros(0)
    xf = np.where(ok, x, np.nanmedian(x))
    y = signal.sosfiltfilt(signal.butter(2, (0.5, 8), btype="bandpass", fs=fs, output="sos"), xf)
    pk, _ = signal.find_peaks(y, distance=int(0.3 * fs), prominence=0.3 * np.nanstd(y))
    pk = pk[ok[pk]]
    return t[pk]


def per_second(t, x, t0, n, how="mean"):
    b = np.floor(t - t0).astype(np.int64)
    k = (b >= 0) & (b < n) & np.isfinite(x)
    if how == "mean":
        s = np.bincount(b[k], x[k], minlength=n)
        c = np.bincount(b[k], minlength=n)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)
    hi = np.full(n, -np.inf)
    lo = np.full(n, np.inf)
    np.maximum.at(hi, b[k], x[k])
    np.minimum.at(lo, b[k], x[k])
    out = hi - lo
    out[~np.isfinite(out)] = np.nan
    return out


def xcorr_lag(a, b, maxlag):
    """Lag (samples) of b against a (b delayed by +lag) maximising correlation; NaN-aware."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.mean() < 0.5:
        return None, None
    a = np.where(ok, a - np.nanmean(a), 0)
    b = np.where(ok, b - np.nanmean(b), 0)
    if a.std() == 0 or b.std() == 0:
        return None, None
    F = 1 << int(np.ceil(np.log2(2 * a.size)))
    r = np.fft.irfft(np.conj(np.fft.rfft(a, F)) * np.fft.rfft(b, F), F)
    r = np.r_[r[-maxlag:], r[: maxlag + 1]] / (a.std() * b.std() * a.size)
    i = int(np.argmax(r))
    return i - maxlag, float(r[i])


def stretches(rt, starts, fs, jump=0.02, min_len=60.0, max_len=300.0):
    """Windows (start, end, in device stamp time) inside stretches where the stamps follow the sample count."""
    res = (rt - rt[0]) - starts / fs
    cut = np.r_[0, np.flatnonzero(np.abs(np.diff(res)) > jump) + 1, rt.size]
    out = []
    for a, b in zip(cut[:-1], cut[1:]):
        t0, t1 = rt[a], rt[b - 1]
        if t1 - t0 < min_len:
            continue
        n = int(np.ceil((t1 - t0) / max_len))
        edges = np.linspace(t0, t1, n + 1)
        out += list(zip(edges[:-1], edges[1:]))
    return out


def cardioq(v, ref_t, ref_x, fs_ref):
    tr = v.tracks.get("CardioQ/ABP")
    if tr is None or not tr.record_times.size:
        return None
    fs = float(tr.srate)
    tq = sample_times(tr)
    xq = np.asarray(tr.physical(), float)[: tq.size]
    pq_ = peaks(tq, xq, fs)
    pr = peaks(ref_t, ref_x, fs_ref)
    if pq_.size < 200 or pr.size < 200:
        return {"status": "too few beats"}
    rt = np.asarray(tr.record_times, float)
    wins = stretches(rt, tr.record_offsets[: rt.size].astype(float), fs)
    t0 = float(np.floor(min(tq[0], ref_t[0])))
    n = int(max(tq[-1], ref_t[-1]) - t0) + 1
    mq = per_second(tq, xq, t0, n)
    mr = per_second(ref_t, ref_x, t0, n)
    rows = []
    prev = None
    for ws, we in wins:
        i0, i1 = int(ws - t0), int(we - t0)
        a, b = mr[max(i0 - 90, 0): i1 + 90], mq[max(i0 - 90, 0): i1 + 90]
        L0, r0 = xcorr_lag(a - np.nanmedian(a), b - np.nanmedian(b), 60)
        if L0 is None or r0 < 0.3:
            if prev is None:
                continue
            L0 = prev
        A = pq_[(pq_ >= ws) & (pq_ < we)]
        B = pr[(pr >= ws - L0 - 5) & (pr < we - L0 + 5)]
        if A.size < 40 or B.size < 40:
            continue
        j = np.clip(np.searchsorted(B, A - L0), 1, B.size - 1)
        j = np.where(np.abs(B[j - 1] - (A - L0)) < np.abs(B[j] - (A - L0)), j - 1, j)
        rra = np.diff(A)
        best = None
        for k in range(-3, 4):
            jj = j[1:] + k
            ok = (jj >= 1) & (jj < B.size)
            rrb = np.full(rra.size, np.nan)
            rrb[ok] = B[jj[ok]] - B[jj[ok] - 1]
            g = ok & (rrb > 0.25) & (rrb < 2.5) & (rra > 0.25) & (rra < 2.5)
            if g.sum() < 30 or np.std(rra[g]) == 0 or np.std(rrb[g]) == 0:
                continue
            c = float(np.corrcoef(rra[g], rrb[g])[0, 1])
            if best is None or c > best[0]:
                best = (c, k)
        if best is None:
            continue
        jj = np.clip(j + best[1], 0, B.size - 1)
        d = A - B[jj]
        med = float(np.median(d))
        mad = float(1.4826 * np.median(np.abs(d - med)))
        prev = med
        rows.append([round(float((ws + we) / 2), 1), round(med, 4), round(mad, 4), round(best[0], 3), int(A.size),
                     round(float(we - ws), 1)])
    return {"windows": rows, "stretches": len(wins)}


def numeric_vs(v, name, pred_t, pred, win_avg, maxlag, block=1800.0):
    tr = v.tracks.get(name)
    if tr is None or not tr.record_times.size:
        return None
    t = np.asarray(tr.record_times, float)
    x = np.asarray(tr.physical(), float)[: t.size]
    ok = np.isfinite(x) & (x > 0)
    t, x = t[ok], x[ok]
    ch = np.r_[True, np.diff(x) != 0]
    t, x = t[ch], x[ch]
    if t.size < 50:
        return {"status": "too few values"}
    t0 = pred_t[0]
    c = np.r_[0.0, np.nancumsum(np.nan_to_num(pred))]
    cnt = np.r_[0, np.cumsum(np.isfinite(pred))]
    rows = []
    lags = np.arange(-maxlag, maxlag + 1, 1)
    for bs in np.arange(t[0], t[-1] - block / 2, block / 2):
        m = (t >= bs) & (t < bs + block)
        if m.sum() < 30:
            continue
        tt, xx = t[m], x[m]
        xx = xx - np.median(xx)
        rs = []
        for L in lags:
            e = np.floor(tt - L - t0).astype(int)
            s = e - int(win_avg)
            okk = (s >= 0) & (e < c.size - 1)
            if okk.sum() < 20:
                rs.append(np.nan)
                continue
            n = cnt[e[okk]] - cnt[s[okk]]
            p = np.where(n > 0.6 * win_avg, (c[e[okk]] - c[s[okk]]) / np.maximum(n, 1), np.nan)
            g = np.isfinite(p)
            if g.sum() < 20 or np.std(p[g]) == 0:
                rs.append(np.nan)
                continue
            rs.append(float(np.corrcoef(xx[okk][g], p[g] - np.median(p[g]))[0, 1]))
        rs = np.array(rs)
        if not np.isfinite(rs).any():
            continue
        i = int(np.nanargmax(rs))
        L = float(lags[i])
        if 0 < i < rs.size - 1 and np.isfinite(rs[i - 1]) and np.isfinite(rs[i + 1]):
            den = rs[i - 1] - 2 * rs[i] + rs[i + 1]
            if den < 0:
                L += 0.5 * (rs[i - 1] - rs[i + 1]) / den
        rows.append([round(float(bs + block / 2), 1), round(L, 2), round(float(rs[i]), 3), int(m.sum())])
    return {"windows": rows}


def fit(rows, lag_i=1, w_i=None, min_r=0.5, r_i=3):
    R = [r for r in rows if r[r_i] is not None and r[r_i] >= min_r]
    if len(R) < 3:
        return {"windows": len(R)}
    t = np.array([r[0] for r in R])
    y = np.array([r[lag_i] for r in R])
    A = np.c_[np.ones_like(t), t - t.mean()]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    sd = float(1.4826 * np.median(np.abs(res - np.median(res))))
    steps = int(np.sum(np.abs(np.diff(y)) > max(5 * sd, 1e-3)))
    return {"windows": len(R), "offset_median_s": round(float(np.median(y)), 4), "drift_ppm": round(float(coef[1] * 1e6), 1),
            "residual_sd_s": round(sd, 4), "steps": steps, "span_h": round(float((t[-1] - t[0]) / 3600), 2)}


def run_case(root, c, u, step_rates):
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital")
    res = {"case": c, "devices": [d for d in v.devices]}
    art = v.tracks.get("SNUADC/ART")
    ref = None
    if art is not None and art.record_times.size:
        ta, _ = snu_times(art, u, step_rates)
        xa = np.asarray(art.physical(), float)[: ta.size]
        ref = (ta, xa, float(art.srate))
    if "CardioQ" in v.devices and ref is not None:
        out = cardioq(v, *ref)
        if out and "windows" in out:
            out["fit"] = fit(out["windows"], lag_i=1, r_i=3, min_r=0.8)
        res["CardioQ"] = out
    if ref is not None and ("EV1000" in v.devices or "Vigileo" in v.devices or "EV-1000" in v.devices):
        ta, xa, _ = ref
        t0 = float(np.floor(ta[0]))
        n = int(ta[-1] - t0) + 1
        mapp = per_second(ta, xa, t0, n)
        mapp[(mapp < 20) | (mapp > 200)] = np.nan
        pp = per_second(ta, xa, t0, n, how="range")
        pp[(pp < 10) | (pp > 150) | ~np.isfinite(mapp)] = np.nan
        pt = t0 + np.arange(n) + 0.5
        # pulse-pressure variation over the trailing 20 s, from per-second pulse pressure
        from numpy.lib.stride_tricks import sliding_window_view  # noqa: PLC0415
        ppv = np.full(n, np.nan)
        if n > 20:
            W = sliding_window_view(pp, 20)
            with np.errstate(all="ignore"):
                val = (np.nanmax(W, axis=1) - np.nanmin(W, axis=1)) / np.nanmean(W, axis=1) * 100
            ppv[19:] = np.where(np.sum(np.isfinite(W), axis=1) >= 15, val, np.nan)
        for dev, name, pred, wa in (("EV1000", "EV1000/ART_MBP", mapp, 20), ("EV-1000", "EV-1000/ART_MBP", mapp, 20),
                                    ("Vigileo", "Vigileo/SVV", ppv, 1), ("EV1000", "EV1000/SVV", ppv, 1)):
            if dev in v.devices and name in v.tracks:
                out = numeric_vs(v, name, pt, pred, wa, 120)
                if out and "windows" in out:
                    out["fit"] = fit(out["windows"], lag_i=1, r_i=2, min_r=0.9 if name.endswith("MBP") else 0.8)
                res[name] = out
    if "Vigilance" in v.devices and "Vigilance/HR_AVG" in v.tracks:
        ecg = v.tracks.get("SNUADC/ECG_II") or v.tracks.get("SNUADC/ECG_V5")
        if ecg is not None and ecg.record_times.size:
            te, _ = snu_times(ecg, u, step_rates)
            xe = np.asarray(ecg.physical(), float)[: te.size]
            ok = np.isfinite(xe)
            rp = r_peaks(np.where(ok, xe, np.nanmedian(xe)), float(ecg.srate))
            rt = te[rp]
            rr = np.diff(rt)
            g = (rr > 0.25) & (rr < 2.5)
            t0 = float(np.floor(te[0]))
            n = int(te[-1] - t0) + 1
            hr = per_second(rt[1:][g], 60 / rr[g], t0, n)
            out = numeric_vs(v, "Vigilance/HR_AVG", t0 + np.arange(n) + 0.5, hr, 60, 300)
            if out and "windows" in out:
                out["fit"] = fit(out["windows"], lag_i=1, r_i=2, min_r=0.8)
            res["Vigilance/HR_AVG"] = out
    return res


def agg(paths, out):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    S = {"cases": len(rows)}
    for key in ("CardioQ", "EV1000/ART_MBP", "EV-1000/ART_MBP", "EV1000/SVV", "Vigileo/SVV", "Vigilance/HR_AVG"):
        F = [r[key]["fit"] for r in rows if isinstance(r.get(key), dict) and "fit" in r[key]]
        G = [f for f in F if f.get("windows", 0) >= 3 and "offset_median_s" in f]
        if not F:
            continue

        def q(k, G=G):
            x = [f[k] for f in G if f.get(k) is not None]
            return [round(float(v_), 4) for v_ in np.percentile(x, [5, 25, 50, 75, 95])] if x else None

        d = {"cases_tried": len(F), "cases_fitted": len(G), "offset_s_p5_25_50_75_95": q("offset_median_s"),
             "abs_offset_s_p50_p95": None, "drift_ppm": q("drift_ppm"), "residual_sd_s": q("residual_sd_s"),
             "steps_cases": sum(1 for f in G if f["steps"] > 0), "windows_median": float(np.median([f["windows"] for f in G])) if G else None}
        if G:
            a = np.abs([f["offset_median_s"] for f in G])
            d["abs_offset_s_p50_p95"] = [round(float(np.percentile(a, 50)), 4), round(float(np.percentile(a, 95)), 4)]
        if key == "CardioQ":
            W = [w for r in rows if isinstance(r.get(key), dict) for w in r[key].get("windows", []) if w[3] >= 0.8]
            if W:
                lag = np.array([w[1] for w in W])
                d["window_lag_s_p5_25_50_75_95"] = [round(float(x), 3) for x in np.percentile(lag, [5, 25, 50, 75, 95])]
                d["stretches_median"] = float(np.median([r[key].get("stretches", 0) for r in rows if isinstance(r.get(key), dict)]))
                d["beat_match_mad_s_p50"] = round(float(np.median([w[2] for w in W])), 4)
                d["rr_corr_p50"] = round(float(np.median([w[3] for w in W])), 3)
        S[key] = d
    json.dump(S, open(out, "w"), indent=1)
    print(json.dumps(S, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--units")
    ap.add_argument("--step-rates")
    ap.add_argument("--out")
    ap.add_argument("--agg", nargs="+")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.agg:
        agg(a.agg, a.summary)
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
            r = run_case(a.root, c, U.get(c, {}), sr)
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:300]}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
