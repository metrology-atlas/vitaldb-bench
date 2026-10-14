# SPDX-License-Identifier: MIT
"""Shared artefacts as timing references between VitalDB devices and channels. Feasibility check, per case,
this machine only (per-case output stays here; only aggregates are reported).

    python tools/shared_artifacts.py --root R --cases 1,2,3 --out out.jsonl [--window 600 --step 300]

Three probes, each in sliding windows over the whole recording, on the files' own time stamps:

1. Broadband bursts (electrocautery, muscle, movement). Log power envelope (20 Hz grid) of SNUADC ECG_II
   30-60 Hz, BIS EEG1/EEG2 30-55 Hz, SNUADC ART and PLETH 30-240 Hz; 0.55 s running median (removes QRS),
   less a 10 s running median (keeps edges, drops trends). Cross-correlation per window (lag +-max_lag_s):
   peak r, lag (parabolic), and the peak's margin over the best lag more than 1 s away.
2. Mains hum (60 Hz). Complex 60 Hz demodulation in 1 s blocks of ART/CVP/PLETH (SNUADC) and EEG1 (BIS) on
   absolute time; the phase difference of two channels, in ms (mod 16.7 ms): block-to-block jitter, slow
   change over the recording (unwrapped), and jumps.
3. Ventilation. Primus AWP against SNUADC CVP (else ART), 0.05-0.6 Hz, 10 Hz grid: lag per window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

sys.path.insert(0, str(Path(__file__).parent))
from pat import sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

warnings.filterwarnings("ignore")
TR = {"ecg": "SNUADC/ECG_II", "pleth": "SNUADC/PLETH", "art": "SNUADC/ART", "cvp": "SNUADC/CVP",
      "eeg1": "BIS/EEG1_WAV", "eeg2": "BIS/EEG2_WAV", "awp": "Primus/AWP"}
BAND = {"ecg": (30, 60), "pleth": (30, 240), "art": (30, 240), "cvp": (30, 240), "eeg1": (30, 55), "eeg2": (30, 55)}
DT = 0.05
PAIRS_BURST = [("ecg", "eeg1"), ("ecg", "eeg2"), ("eeg1", "eeg2"), ("ecg", "art"), ("ecg", "pleth"), ("art", "eeg1")]
HUM = 60.0


def binned(t, y, t0, dt, n):
    b = np.floor((t - t0) / dt).astype(np.int64)
    k = (b >= 0) & (b < n)
    s = np.bincount(b[k], y[k], minlength=n)
    c = np.bincount(b[k], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan), c


def envelope(t, x, fs, band, t0, n):
    sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x).astype(np.float32) ** 2
    m, _ = binned(t, y, t0, DT, n)
    e = np.log10(m + 1e-12)
    ok = np.isfinite(m)
    e[~ok] = np.nanmedian(e)
    e = ndimage.median_filter(e, 11)
    e = e - ndimage.median_filter(e, 201)
    e[~ok] = np.nan
    return e


def xcorr_window(a, b, maxlag):
    """Normalised cross-correlation r(lag) for lag in [-maxlag, maxlag] bins; b delayed by lag -> peak at +lag."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.mean() < 0.5:
        return None
    a = np.where(ok, a - np.nanmean(a[ok]), 0.0)
    b = np.where(ok, b - np.nanmean(b[ok]), 0.0)
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return None
    n = a.size
    F = 1 << int(np.ceil(np.log2(2 * n)))
    r = np.fft.irfft(np.conj(np.fft.rfft(a, F)) * np.fft.rfft(b, F), F)
    r = np.r_[r[-maxlag:], r[: maxlag + 1]] / (sa * sb * n)
    return r


def peak(r, dt, excl_bins):
    i = int(np.argmax(r))
    lag = i
    if 0 < i < r.size - 1:
        y0, y1, y2 = r[i - 1], r[i], r[i + 1]
        den = y0 - 2 * y1 + y2
        lag = i + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
    far = np.abs(np.arange(r.size) - i) > excl_bins
    second = float(r[far].max()) if far.any() else np.nan
    maxlag = (r.size - 1) // 2
    return (lag - maxlag) * dt, float(r[i]), second


def windows(n, win, step):
    return [(s, s + win) for s in range(0, max(1, n - win + 1), step)]


def burst_probe(env, win_s, step_s, maxlag_s):
    out = {}
    n = min(e.size for e in env.values())
    W, S, L = int(win_s / DT), int(step_s / DT), int(maxlag_s / DT)
    for a, b in PAIRS_BURST:
        if a not in env or b not in env:
            continue
        rows = []
        for s, e in windows(n, W, S):
            ea, eb = env[a][s:e], env[b][s:e]
            act = float(np.nanmean((ea > 0.5) & (eb > 0.5)))
            r = xcorr_window(ea, eb, L)
            if r is None:
                continue
            lag, pk, sec = peak(r, DT, int(1.0 / DT))
            rows.append([round((s + (e - s) / 2) * DT, 1), round(lag * 1000, 1), round(pk, 3), round(sec, 3), round(act, 4)])
        out[f"{a}~{b}"] = rows
    return out


def hum_probe(tracks, t0, n_s):
    """1 s complex demodulation at 60 Hz on absolute time."""
    Z = {}
    for k in ("art", "cvp", "pleth", "eeg1", "eeg2", "ecg"):
        if k not in tracks:
            continue
        t, x, fs = tracks[k]
        tt = t - t0
        x = x - ndimage.uniform_filter1d(x, int(fs * 0.2))  # remove slow content
        z_re, _ = binned(t, x * np.cos(2 * np.pi * HUM * tt), t0, 1.0, n_s)
        z_im, _ = binned(t, -x * np.sin(2 * np.pi * HUM * tt), t0, 1.0, n_s)
        o_re, _ = binned(t, x * np.cos(2 * np.pi * 57.0 * tt), t0, 1.0, n_s)
        o_im, _ = binned(t, -x * np.sin(2 * np.pi * 57.0 * tt), t0, 1.0, n_s)
        z = z_re + 1j * z_im
        snr = np.abs(z) / (np.nanmedian(np.abs(o_re + 1j * o_im)) + 1e-12)
        Z[k] = (z, snr)
    res = {"snr_median": {k: round(float(np.nanmedian(v[1])), 2) for k, v in Z.items()}}
    period_ms = 1000.0 / HUM
    for a, b in [("art", "eeg1"), ("cvp", "eeg1"), ("pleth", "eeg1"), ("art", "cvp"), ("eeg1", "eeg2")]:
        if a not in Z or b not in Z:
            continue
        (za, sa), (zb, sb) = Z[a], Z[b]
        ok = np.isfinite(za) & np.isfinite(zb) & (sa > 5) & (sb > 5)
        if ok.sum() < 300:
            res[f"{a}~{b}"] = {"blocks": int(ok.sum())}
            continue
        d = np.angle(za * np.conj(zb))           # radians, a ahead of b
        idx = np.flatnonzero(ok)
        dd = d[idx]
        step = np.angle(np.exp(1j * np.diff(dd)))  # block-to-block change
        contiguous = np.diff(idx) == 1
        jit = step[contiguous] / (2 * np.pi) * period_ms
        un = np.unwrap(dd) / (2 * np.pi) * period_ms
        tt = idx.astype(float)
        slope = np.polyfit(tt, un, 1)[0] if tt.size > 10 else np.nan
        res[f"{a}~{b}"] = {
            "blocks": int(ok.sum()),
            "jitter_ms_sd": round(float(np.std(jit)), 3) if jit.size else None,
            "jitter_ms_mad": round(float(1.4826 * np.median(np.abs(jit - np.median(jit)))), 3) if jit.size else None,
            "jumps_gt2ms": int(np.sum(np.abs(jit) > 2.0)),
            "unwrapped_range_ms": round(float(np.ptp(un)), 2),
            "slope_ms_per_h": round(float(slope * 3600), 3),
            "phase_ms_median": round(float(np.angle(np.mean(np.exp(1j * dd))) / (2 * np.pi) * period_ms), 2),
            "resultant": round(float(np.abs(np.mean(np.exp(1j * dd)))), 3),
            "series_10s": [round(float(v), 2) for v in un[::10]],
        }
    return res


def vent_probe(tracks, t0, n10, win_s, step_s, maxlag_s):
    if "awp" not in tracks:
        return {"skip": "no AWP"}
    ref = "cvp" if "cvp" in tracks else ("art" if "art" in tracks else None)
    if ref is None:
        return {"skip": "no CVP/ART"}
    dt = 0.1
    sos = signal.butter(2, (0.05, 0.6), btype="bandpass", fs=1 / dt, output="sos")
    ser = {}
    for k in ("awp", ref):
        t, x, _ = tracks[k]
        m, _ = binned(t, x, t0, dt, n10)
        ok = np.isfinite(m)
        if ok.mean() < 0.2:
            return {"skip": f"{k} sparse"}
        f = np.interp(np.arange(m.size), np.flatnonzero(ok), m[ok])
        f = signal.sosfiltfilt(sos, f)
        f[~ok] = np.nan
        ser[k] = f
    W, S, L = int(win_s / dt), int(step_s / dt), int(maxlag_s / dt)
    rows = []
    for s, e in windows(n10, W, S):
        r = xcorr_window(ser["awp"][s:e], ser[ref][s:e], L)
        if r is None:
            continue
        lag, pk, sec = peak(r, dt, int(1.5 / dt))
        rows.append([round((s + (e - s) / 2) * dt, 1), round(lag * 1000, 1), round(pk, 3), round(sec, 3)])
    return {"ref": ref, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=600)
    ap.add_argument("--step", type=float, default=300)
    ap.add_argument("--max-lag", type=float, default=20)
    ap.add_argument("--vent-max-lag", type=float, default=2.0, help="within half a breath: ventilation is periodic")
    a = ap.parse_args()
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
                v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names=set(TR.values()))
                tracks = {}
                for k, n in TR.items():
                    tr = v.tracks.get(n)
                    if tr is None or not tr.record_times.size:
                        continue
                    x = tr.physical().astype(np.float32)
                    x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
                    tracks[k] = (sample_times(tr), x, float(tr.srate))
                res["tracks"] = sorted(tracks)
                t0 = float(np.floor(min(v_[0][0] for v_ in tracks.values())))
                t_end = max(v_[0][-1] for v_ in tracks.values())
                n = int((t_end - t0) / DT) + 1
                env = {k: envelope(tracks[k][0], tracks[k][1], tracks[k][2], BAND[k], t0, n) for k in BAND if k in tracks}
                res["burst"] = burst_probe(env, a.window, a.step, a.max_lag)
                del env
                res["hum"] = hum_probe(tracks, t0, int(t_end - t0) + 1)
                res["vent"] = vent_probe(tracks, t0, int((t_end - t0) / 0.1) + 1, a.window, a.step, a.vent_max_lag)
                res["hours"] = round((t_end - t0) / 3600, 2)
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            res["seconds"] = round(time.time() - t1, 1)
            fh.write(json.dumps(res) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
