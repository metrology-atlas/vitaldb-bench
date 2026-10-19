# SPDX-License-Identifier: MIT
"""Pilot: the heartbeat in the Primus airway pressure and CO2, as a beat-by-beat link between Primus and SNUADC.

    python tools/airway_heartbeat_pilot.py scan --root R --cases 1,2 --modes M --ports P --step-rates S --out part.jsonl
    python tools/airway_heartbeat_pilot.py agg part*.jsonl [--out DIR]

Each heartbeat moves the lungs a little, which can show in the airway pressure and the capnogram as small
oscillations at the heart rate (cardiogenic oscillations), clearest when flow pauses in expiration. Averaged on
the ECG's R peaks, as the cardiac trace in the BIS EEG was, they would time the Primus against the SNUADC beat by
beat rather than breath by breath.

Per 10-minute window (stepped by 5 min): R peaks from ECG II on the SNUADC stamp model; AWP and CO2 on the
Primus's own stamps, high-passed at 0.6 Hz, with their breath-locked average shape removed (otherwise the heart
rate's own breathing rhythm, sinus arrhythmia, makes the breath waveform itself look heart-locked), and kept only
in expiration (AWP within its lowest 40% in the window, CO2 on its plateau: above 70% of the window's range); the
R-triggered average over -1.5 to +2.0 s. Lag: the largest positive peak within -0.5 to +1.5 s, after the first
clear window the peak nearest the previous window's lag (the oscillation repeats every beat, so the lag is
otherwise ambiguous by one beat; the largest peak regardless is kept too, as lag_free, because following the
previous window can also lock onto noise). Strength: that peak against the same average with the R times jittered by up to
half a beat (peak-to-noise). The breath-matching lag (AWP to CVP) for the same windows is kept for comparison.

agg: how often the trace is clear (peak-to-noise >= 5), the cardiac lag's spread within a stretch between AWP
pauses (residual SD of one line per stretch) against the breath lag's, and whether the two lags change together
(their slopes per stretch, and the correlation of window-to-window changes). A usable beat-level link needs a clear
trace in most windows, a spread well under the breath lag's, and changes that follow the breath lag's: the Primus
drifts about +100 ppm against the SNUADC, so a real cardiac trace on the Primus's stamps must ramp with the breath
lag; a lag that stays put while the breath lag ramps is not on the Primus's clock.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppv_by_breath as P  # noqa: E402
from pat import r_peaks, sample_times  # noqa: E402

WIN_S, STEP_S = 600.0, 300.0
TAUS = np.arange(-1.5, 2.0, 0.016)
SEARCH = (-0.5, 1.5)
PNR_MIN = 5.0


def trig(t_sig, x, r_times, taus):
    """Average of x at r + tau (nearest sample within 10 ms), ignoring nan."""
    idx = np.searchsorted(t_sig, r_times[:, None] + taus[None, :])
    idx = np.clip(idx, 1, t_sig.size - 1)
    left = np.abs(t_sig[idx - 1] - (r_times[:, None] + taus)) < np.abs(t_sig[idx] - (r_times[:, None] + taus))
    idx = idx - left.astype(int)
    v = x[idx]
    far = np.abs(t_sig[idx] - (r_times[:, None] + taus)) > 0.01
    v = np.where(far, np.nan, v)
    return np.nanmean(v, axis=0), np.sum(np.isfinite(v), axis=0)


def remove_breath(t_sig, x, onsets):
    """x minus its breath-locked average shape: each sample loses the mean of the samples at the same time after
    their breath's onset, so whatever follows the breath cycle (and the heart only through breathing) is gone."""
    k = np.searchsorted(onsets, t_sig, side="right") - 1
    ok = k >= 0
    ph = np.full(t_sig.size, -1, dtype=np.int64)
    ph[ok] = np.round((t_sig[ok] - onsets[k[ok]]) / 0.016).astype(np.int64)
    good = ok & (ph >= 0) & (ph < 1000)
    tmpl = np.bincount(ph[good], x[good], minlength=1000) / np.maximum(np.bincount(ph[good], minlength=1000), 1)
    out = np.full(x.size, np.nan)
    out[good] = x[good] - tmpl[ph[good]]
    return out


def cardiac_lag(t_sig, x_res, mask, r_times, rr, rng, prev=None):
    """R-triggered average of the breath-removed signal; lag of its largest positive peak (nearest the previous
    window's lag, if any, to stay on one beat); strength against R times jittered by up to half a beat."""
    xm = np.where(mask, x_res, np.nan)
    avg, n = trig(t_sig, xm, r_times, TAUS)
    ctl = [trig(t_sig, xm, r_times + rng.uniform(-rr / 2, rr / 2, r_times.size), TAUS)[0] for _ in range(3)]
    s = (TAUS >= SEARCH[0]) & (TAUS <= SEARCH[1])
    if np.isfinite(avg[s]).sum() < 10:
        return None
    pk, _ = signal.find_peaks(np.nan_to_num(np.where(s, avg, -np.inf), neginf=-1e9))
    if pk.size == 0:
        return None
    i_free = int(pk[np.argmax(avg[pk])])
    i = int(pk[np.argmin(np.abs(TAUS[pk] - prev))]) if prev is not None else i_free
    noise = float(np.nanstd(np.concatenate(ctl)))
    return (float(TAUS[i]), float(avg[i] / noise) if noise > 0 else None, int(np.nanmedian(n)), float(TAUS[i_free]),
            float(avg[i_free] / noise) if noise > 0 else None)


def scan_case(c, a, modes, ports, step_rates):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital",
                   track_names={"SNUADC/ECG_II", "SNUADC/CVP", "Primus/AWP", "Primus/CO2"})
    ecg, aw, co2, cv = (v.tracks.get(n) for n in ("SNUADC/ECG_II", "Primus/AWP", "Primus/CO2", "SNUADC/CVP"))
    if ecg is None or aw is None or not ecg.record_times.size or not aw.record_times.size or float(ecg.srate) < 400:
        return {"case": c, "skip": "no 500 Hz ECG or no AWP"}
    te = P.model_times(ecg, ports.get(c), modes.get(c), step_rates)
    xe = ecg.physical().astype(np.float64)[: te.size]
    pk = r_peaks(np.nan_to_num(xe, nan=float(np.nanmedian(xe))), float(ecg.srate))
    rt = te[pk]
    ta = sample_times(aw)
    xa = np.nan_to_num(aw.physical().astype(np.float64))[: ta.size]
    fs_a = float(aw.srate)
    sos = signal.butter(2, 0.6, btype="highpass", fs=fs_a, output="sos")
    xa_hp = signal.sosfiltfilt(sos, xa)
    on_all = P.breath_onsets(ta, xa, fs_a)
    sig = {"awp": (ta, xa, remove_breath(ta, xa_hp, on_all))}
    if co2 is not None and co2.record_times.size:
        tco = sample_times(co2)
        xc = np.nan_to_num(co2.physical().astype(np.float64))[: tco.size]
        xc_hp = signal.sosfiltfilt(signal.butter(2, 0.6, btype="highpass", fs=float(co2.srate), output="sos"), xc)
        sig["co2"] = (tco, xc, remove_breath(tco, xc_hp, on_all))
    rng = np.random.default_rng(c)
    prev = {}
    res = {"case": c, "z": float(ta[0]), "pauses": (ta[1:][np.diff(ta) > 5] - ta[0]).round(1).tolist(), "windows": []}
    for ws in np.arange(max(ta[0], rt[0]), min(ta[-1], rt[-1]) - WIN_S, STEP_S):
        r = rt[(rt >= ws + 2) & (rt < ws + WIN_S - 2)]
        if r.size < 100:
            continue
        rr = float(np.median(np.diff(r)))
        row = {"t": round(float(ws + WIN_S / 2 - ta[0]), 1), "beats": int(r.size), "rr": round(rr, 3)}
        for k, (ts, x, xh) in sig.items():
            m = (ts >= ws) & (ts < ws + WIN_S)
            if m.sum() < 100:
                continue
            lo, hi = np.percentile(x[m], [2, 98])
            keep = (x < lo + 0.4 * (hi - lo)) if k == "awp" else (x > lo + 0.7 * (hi - lo))
            out = cardiac_lag(ts[m], xh[m], keep[m], r, rr, rng, prev.get(k))
            if out:
                row[f"{k}_lag"], row[f"{k}_pnr"], row[f"{k}_n"] = round(out[0], 3), round(out[1], 2) if out[1] else None, out[2]
                row[f"{k}_lag_free"], row[f"{k}_pnr_free"] = round(out[3], 3), round(out[4], 2) if out[4] else None
                if out[1] and out[1] >= PNR_MIN:
                    prev[k] = out[0]
        res["windows"].append(row)
    if cv is not None and cv.record_times.size:
        tc = P.model_times(cv, ports.get(c), modes.get(c), step_rates)
        tl, ll = P.cvp_lag(ta, xa, tc, cv.physical().astype(np.float64)[: tc.size], on_all)
        res["breath"] = {"t": (tl - ta[0]).round(1).tolist(), "lag": ll.round(4).tolist()}
    return res


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


def stretch_resid(t, y, pauses):
    st = np.searchsorted(np.asarray(pauses, float), t)
    out, slopes = [], []
    for s in np.unique(st):
        m = st == s
        if m.sum() >= 4 and np.ptp(t[m]) > 1800:
            b = np.polyfit(t[m], y[m], 1)
            out += list(y[m] - np.polyval(b, t[m]))
            slopes.append((s, b[0]))
    return np.array(out), dict(slopes)


def agg(a):
    R = [json.loads(l) for p in a.parts for l in open(p, encoding="utf-8") if l.strip()]
    out = {"recordings": len(R), "errors": sum(1 for r in R if "error" in r)}
    for k in ("awp", "co2"):
        pn = np.array([w.get(f"{k}_pnr") for r in R for w in r.get("windows", []) if w.get(f"{k}_pnr") is not None], float)
        res_c, res_b, slope_pairs, dpairs = [], [], [], []
        per_rec = []
        for r in R:
            W = [w for w in r.get("windows", []) if (w.get(f"{k}_pnr") or 0) >= PNR_MIN]
            if len(W) < 6:
                continue
            t = np.array([w["t"] for w in W]); y = np.array([w[f"{k}_lag"] for w in W])
            rc, sc = stretch_resid(t, y, r["pauses"])
            res_c += list(rc)
            per_rec.append(r["case"])
            B = r.get("breath")
            if B and len(B["t"]) >= 6:
                tb, yb = np.array(B["t"]), np.array(B["lag"])
                rb, sb = stretch_resid(tb, yb, r["pauses"])
                res_b += list(rb)
                for s in set(sc) & set(sb):
                    slope_pairs.append((sc[s] * 1e6, sb[s] * 1e6))
                for i in range(1, t.size):
                    j0, j1 = np.argmin(np.abs(tb - t[i - 1])), np.argmin(np.abs(tb - t[i]))
                    if abs(tb[j0] - t[i - 1]) < 60 and abs(tb[j1] - t[i]) < 60 and j0 != j1:
                        dpairs.append((y[i] - y[i - 1], yb[j1] - yb[j0]))
        sp = np.array(slope_pairs) if slope_pairs else np.empty((0, 2))
        dp = np.array(dpairs) if dpairs else np.empty((0, 2))
        out[k] = {"windows": int(pn.size), "share_pnr_ge_5": round(float(np.mean(pn >= PNR_MIN)), 3) if pn.size else None,
                  "pnr_median": round(float(np.median(pn)), 2) if pn.size else None,
                  "recordings_usable": len(per_rec),
                  "cardiac_resid_sd_ms": round(float(np.std(res_c)) * 1000, 1) if res_c else None,
                  "cardiac_resid_mad_ms": round(float(1.4826 * np.median(np.abs(res_c))) * 1000, 1) if res_c else None,
                  "breath_resid_mad_ms_same_recordings": round(float(1.4826 * np.median(np.abs(res_b))) * 1000, 1) if res_b else None,
                  "stretch_slopes_ppm_cardiac_vs_breath": {"stretches": int(len(sp)),
                                                           "corr": round(float(np.corrcoef(sp.T)[0, 1]), 3) if len(sp) > 3 else None,
                                                           "median_diff_ppm": round(float(np.median(sp[:, 0] - sp[:, 1])), 1) if len(sp) else None},
                  "window_change_corr": round(float(np.corrcoef(dp.T)[0, 1]), 3) if len(dp) > 10 else None}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1))


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
