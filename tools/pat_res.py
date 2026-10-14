# SPDX-License-Identifier: MIT
"""Effective time resolution of SNUADC ECG and PLETH, and the PAT sawtooth measured beat by beat.

    python tools/pat_res.py --root R --cases cases.json --out pat_res.jsonl [--minutes 60] [--time-budget 150]

Both tracks are stored at 500 Hz. That is the SNUADC's sampling rate; the signal it samples may have
been produced at a lower rate upstream (the monitor regenerates its pleth from its own samples). Three
checks, on the first ``--minutes`` of continuous overlap in each case:

1. Sample-and-hold: lengths of runs of identical consecutive stored values. A signal regenerated at
   F Hz and sampled at 500 Hz repeats each value about 500/F times.
2. Fiducial quantisation (as in Ruffolo et al. 2025, figure 4): fiducials are taken at whole samples
   (R-peak = extreme of the band-passed ECG; pulse onset = the sample of steepest rise of the raw PPG,
   and the last minimum before it), and the change between consecutive inter-beat intervals is
   histogrammed in samples. A coarser upstream rate shows as IBI changes that are multiples of k samples.
3. Beat-matched PAT: each R-peak is paired with the PPG onset nearest to R + the case's typical delay
   (the mode of all onset-minus-R differences in 0.2-1.6 s), so delays longer than one beat are matched
   correctly. The per-beat PAT series is then scanned for resets as in the paper's Kalman approach: a
   running line through the current tooth; a reset when two consecutive beats sit more than
   ``--reset-ms`` above it. Reports tooth slopes, reset sizes and periods.
Output: histograms and per-case summaries only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from pat import ECG, PPG, r_peaks, sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import (
    read_vital,
)


def hold_runs(raw: np.ndarray, maxlen: int = 40) -> np.ndarray:
    """Histogram of lengths of runs of identical consecutive values (1..maxlen)."""
    change = np.flatnonzero(np.diff(raw) != 0)
    lengths = np.diff(change)
    return np.bincount(np.clip(lengths, 0, maxlen), minlength=maxlen + 1)[1:]


def kinks(raw: np.ndarray, kmax: int = 40) -> dict:
    """Signs of a lower upstream rate that was interpolated (not held) up to 500 Hz.

    A signal made at 500/k Hz and linearly interpolated is piecewise linear: its second difference is
    non-zero only every k samples. So: the autocorrelation of |second difference| at lags 1..kmax, and
    for each k the share of |d2| in the strongest of the k sample phases (1/k if there is no grid)."""
    d2 = np.abs(np.diff(raw.astype(np.float64), 2))
    a = d2 - d2.mean()
    ac = [float(np.dot(a[:-lag], a[lag:]) / np.dot(a, a)) for lag in range(1, kmax + 1)]
    phase = {}
    for k in range(2, 33):
        n = d2.size // k * k
        m = d2[:n].reshape(-1, k).sum(axis=0)
        phase[k] = round(float(m.max() / m.sum()) * k, 3)  # 1.0 = no grid, k = all energy on one phase
    return {"d2_zero_frac": round(float(np.mean(d2 == 0)), 4), "d2_acf": [round(x, 3) for x in ac], "phase_conc": phase}


def spectrum(raw: np.ndarray, fs: float) -> dict:
    """Welch PSD in 2 Hz bins up to fs/2 (dB relative to the peak), to see where the content ends."""
    f, p = signal.welch(raw.astype(np.float64) - np.mean(raw), fs=fs, nperseg=4096)
    edges = np.arange(0, fs / 2 + 2, 2.0)
    idx = np.digitize(f, edges) - 1
    band = np.array([p[idx == i].mean() if np.any(idx == i) else np.nan for i in range(edges.size - 1)])
    return {"hz": edges[:-1].tolist(), "db": [round(float(x), 1) for x in 10 * np.log10(band / np.nanmax(band))]}


def parabolic(y: np.ndarray, i: np.ndarray) -> np.ndarray:
    """Sub-sample position of an extreme at integer index i (three-point parabola)."""
    i = np.clip(i, 1, y.size - 2)
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    return i + np.clip(off, -0.5, 0.5)


def ppg_onsets(raw: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """PPG upstroke: the sample of steepest rise of the 0.5-10 Hz band-passed signal (integer index),
    and its sub-sample position from a parabola through the derivative."""
    x = raw.astype(np.float64)
    sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)
    d = np.gradient(y)
    up, _ = signal.find_peaks(d, distance=int(0.3 * fs), prominence=0.3 * (np.quantile(d, 0.99) - np.quantile(d, 0.5)))
    return up, parabolic(d, up)


def ecg_subsample(x: np.ndarray, rp: np.ndarray, fs: float) -> np.ndarray:
    sos = signal.butter(2, (5, 25), btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)
    sign = 1.0 if np.median(y[rp]) >= 0 else -1.0
    return parabolic(sign * y, rp)


def divisibility(idx: np.ndarray, lim: int = 40) -> dict:
    """Share of IBI changes divisible by k, divided by 1/k (1.0 = no grid; k = all on the grid)."""
    dd = np.diff(np.diff(idx))
    dd = dd[np.abs(dd) <= lim]
    return {k: round(float(np.mean(dd % k == 0)) * k, 3) for k in range(2, 17)} if dd.size else {}


def ibi_diff_hist(idx: np.ndarray, lim: int = 40) -> np.ndarray:
    ibi = np.diff(idx)
    dd = np.diff(ibi)
    dd = dd[np.abs(dd) <= lim]
    return np.bincount(dd + lim, minlength=2 * lim + 1)


def resets(t: np.ndarray, pat: np.ndarray, thr: float, smooth: int = 15, consec: int = 5) -> dict:
    """Tooth tracker on the running median of ``smooth`` beats (as the paper's Kalman filter does, the
    beat noise is averaged before deciding): a running line through the current tooth; a reset when
    ``consec`` consecutive smoothed beats sit more than ``thr`` above it. Reports, per tooth, its slope
    (ppm, positive = PAT shrinking) and the size of the jump that ends it."""
    if pat.size < 4 * smooth:
        return {"n_resets": 0, "sizes_ms": [], "steps_ms": [], "reset_t_s": [], "slopes_ppm": [], "periods_s": []}
    ps = signal.medfilt(pat, smooth)
    starts, sizes, slopes, steps = [0], [], [], []
    i0, i, above = 0, 1, 0
    while i < ps.size:
        if i - i0 >= 2 * smooth:
            k = np.polyfit(t[i0:i], ps[i0:i], 1)
            pred = np.polyval(k, t[i])
        else:
            pred = np.median(ps[i0:i])
        if ps[i] - pred > thr:
            above += 1
            if above >= consec:
                j = i - consec + 1  # first beat of the new tooth
                if j - i0 >= 2 * smooth:
                    k = np.polyfit(t[i0:j], pat[i0:j], 1)
                    slopes.append(float(k[0]))
                    sizes.append(float(np.median(pat[j : j + smooth]) - np.polyval(k, t[j])))
                if j >= smooth and j + smooth <= pat.size:
                    # local step: 15 beats after minus 15 before (about 10 s each side)
                    steps.append(float(np.median(pat[j : j + smooth]) - np.median(pat[j - smooth : j])))
                starts.append(j)
                i0, above = j, 0
                i = j + 1
                continue
        else:
            above = 0
        i += 1
    per = np.diff(t[starts]) if len(starts) > 1 else np.array([])
    return {"n_resets": len(starts) - 1, "sizes_ms": [round(s * 1000, 2) for s in sizes],
            "steps_ms": [round(s * 1000, 2) for s in steps], "reset_t_s": [round(float(t[k]), 1) for k in starts[1:]],
            "slopes_ppm": [round(-s * 1e6, 1) for s in slopes], "periods_s": [round(float(p), 1) for p in per]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--reset-ms", type=float, default=16.0)
    ap.add_argument("--time-budget", type=float, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t_start, slowest = time.time(), 0.0
    with out.open("a") as fh:
        for c in json.loads(Path(a.cases).read_text())["cases"]:
            if c in done:
                continue
            if time.time() - t_start + max(2 * slowest, 5) > a.time_budget:
                break
            t1 = time.time()
            res: dict = {"case": c}
            case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG})
            e, p = case.tracks.get(ECG), case.tracks.get(PPG)
            if e is None or p is None or not e.record_times.size or not p.record_times.size:
                res["skip"] = "no ECG_II or PLETH"
            else:
                fs = float(e.srate)
                res["srate"] = {"ecg": float(e.srate), "ppg": float(p.srate)}
                te, tp = sample_times(e), sample_times(p)
                # longest stretch where both tracks are continuous, capped at --minutes
                gap_e = np.flatnonzero(np.abs(np.diff(te) - 1 / fs) > 0.01 / fs)
                be = np.r_[0, gap_e + 1, te.size]
                k = int(np.argmax(np.diff(be)))
                ea, eb = be[k], be[k + 1]
                eb = min(eb, ea + int(a.minutes * 60 * fs))
                pa, pb = np.searchsorted(tp, te[ea]), np.searchsorted(tp, te[eb - 1])
                if eb - ea < 600 * fs or np.max(np.abs(np.diff(tp[pa:pb]) - 1 / fs)) > 0.01 / fs:
                    res["skip"] = "no 10 min continuous stretch"
                else:
                    raw_e, raw_p = np.asarray(e.values[ea:eb]), np.asarray(p.values[pa:pb])
                    res["fmt"] = {"ecg": e.fmt, "ppg": p.fmt}
                    res["hold_ecg"] = hold_runs(raw_e).tolist()
                    res["hold_ppg"] = hold_runs(raw_p).tolist()
                    res["kinks_ecg"], res["kinks_ppg"] = kinks(raw_e), kinks(raw_p)
                    res["psd_ecg"], res["psd_ppg"] = spectrum(raw_e, fs), spectrum(raw_p, fs)
                    xe = e.physical()[ea:eb].astype(np.float64)
                    rp = r_peaks(xe, fs)
                    up_i, up_f = ppg_onsets(raw_p, fs)
                    res["ibi_dd_ecg"] = ibi_diff_hist(rp).tolist()
                    res["ibi_dd_ppg"] = ibi_diff_hist(up_i).tolist()
                    res["div_ecg"], res["div_ppg"] = divisibility(rp), divisibility(up_i)
                    if rp.size < 100 or up_i.size < 100:
                        res["skip"] = "too few beats"
                    else:
                        t0e, t0p = te[ea], tp[pa]
                        rf = ecg_subsample(xe, rp, fs)
                        for tag, (tr, ts) in {"int": (t0e + rp / fs, t0p + up_i / fs),
                                              "sub": (t0e + rf / fs, t0p + up_f / fs)}.items():
                            # typical delay: mode of all onset-minus-R differences between 0.2 and 1.6 s
                            j0 = np.searchsorted(ts, tr + 0.2)
                            diffs = np.concatenate([ts[np.clip(j0 + m, 0, ts.size - 1)] - tr for m in range(4)])
                            diffs = diffs[(diffs > 0.2) & (diffs < 1.6)]
                            h, edges = np.histogram(diffs, np.arange(0.2, 1.6, 0.004))
                            delay = float(edges[np.argmax(h)] + 0.002)
                            j = np.clip(np.searchsorted(ts, tr + delay), 1, ts.size - 1)
                            j = np.where(np.abs(ts[j - 1] - tr - delay) < np.abs(ts[j] - tr - delay), j - 1, j)
                            pat = ts[j] - tr
                            rr = float(np.median(np.diff(tr)))
                            good = np.abs(pat - delay) < 0.25 * rr  # matched to its own beat, not a neighbour
                            pg, tg = pat[good], tr[good] - tr[0]
                            dp = np.diff(pg)
                            r = {"delay_mode_s": delay, "beats": int(good.sum()), "rr_median_s": rr,
                                 "pat_q": [float(x) for x in np.quantile(pg, [0.05, 0.25, 0.5, 0.75, 0.95])],
                                 # beat-to-beat PAT noise: robust SD of successive differences / sqrt(2)
                                 "sd_beat_ms": float(1.4826 * np.median(np.abs(dp - np.median(dp))) / np.sqrt(2) * 1000),
                                 "sawtooth": resets(tg, pg, a.reset_ms / 1000),
                                 # control: the sign-flipped series has no real upward resets, so what the
                                 # tracker finds there is the false-detection rate at this threshold
                                 "null": resets(tg, 2 * np.median(pg) - pg, a.reset_ms / 1000)}
                            if tag == "int":
                                r["pat_ms_mod16"] = np.bincount(np.round(pg * 1000).astype(int) % 16, minlength=16).tolist()
                            else:
                                # 15-beat medians, every 15th beat: for plots on this machine only
                                n = pg.size // 15 * 15
                                r["series"] = [np.round(tg[:n].reshape(-1, 15).mean(1), 1).tolist(),
                                               np.round(np.median(pg[:n].reshape(-1, 15), 1) * 1000, 2).tolist()]
                            res[tag] = r
            res["seconds"] = time.time() - t1
            fh.write(json.dumps(res) + "\n")
            slowest = max(slowest, res["seconds"])
    print(json.dumps({"done": sum(1 for _ in out.open()), "seconds": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
