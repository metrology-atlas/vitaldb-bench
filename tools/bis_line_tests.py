# SPDX-License-Identifier: MIT
"""BIS 60 Hz line: (A) does its minute-by-minute frequency follow the grid seen through SNUADC's 120 Hz line?
(B) is the BIS-minus-SNUADC phase continuous across BIS stamp steps when BIS samples are placed by count?
This machine only; per-case printout.

Usage: python tools/bis_line_tests.py ROOT RATES.csv CASE [CASE ...]
"""
import csv, sys
from pathlib import Path
import numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

def demod(tr, f_line, sub=10.0):
    """Phase per `sub`-second block on the sample count; block times from the first stamp + count (whole track),
    plus each block's nominal stamp time (by contiguous runs) for alignment with the other device."""
    rt, rs, fs = tr.record_times, tr.record_sizes, float(tr.srate)
    x = np.nan_to_num(tr.physical().astype(float))
    n = int(fs * sub); nb = x.size // n
    z = (x[: nb * n] * np.exp(-2j * np.pi * f_line * np.arange(nb * n) / fs)).reshape(nb, n).mean(1)
    starts = np.r_[0, np.cumsum(rs)[:-1]]
    centre = np.arange(nb) * n + n / 2
    rec = np.searchsorted(starts, centre, side="right") - 1
    t_stamp = rt[rec] + (centre - starts[rec]) / fs                  # stamp-based time of each block
    return t_stamp, centre / fs, z                                    # count time from the track start

def steps_of(tr):
    rt, rs, fs = tr.record_times, tr.record_sizes, float(tr.srate)
    d = np.diff(rt) - rs[:-1] / fs
    i = np.where(np.abs(d) > 0.02)[0]
    return rt[i + 1], d[i]

root = Path(sys.argv[1]); rates = {x['case']: x for x in csv.DictReader(open(sys.argv[2]))}
for c in sys.argv[3:]:
    v = read_vital(root / "vital_files" / f"{int(c):04d}.vital",
                   track_names={"SNUADC/ECG_V5", "SNUADC/ECG_II", "SNUADC/CVP", "BIS/EEG1_WAV", "BIS/EEG2_WAV"})
    model = float(rates[c]['r_eeg']) - float(rates[c]['r_ecg'])
    # strongest channel of each device by median block amplitude
    def best(keys, f):
        cand = []
        for k in keys:
            tr = v.tracks.get(k)
            if tr is None or not tr.record_times.size: continue
            ts, tc, z = demod(tr, f); cand.append((np.median(np.abs(z)) / (np.median(np.abs(np.diff(z))) + 1e-12), k, ts, tc, z, tr))
        return max(cand, key=lambda e: e[0]) if cand else None
    S = best(("SNUADC/ECG_V5", "SNUADC/ECG_II", "SNUADC/CVP"), 120.0)
    B = best(("BIS/EEG1_WAV", "BIS/EEG2_WAV"), 60.0)
    if not S or not B: print(c, "missing"); continue
    # --- A: frequency per 60 s (6 blocks), ppm of the line, on each device's count; paired by stamp time
    def freq(ts, tc, z, h):
        ph = np.unwrap(np.angle(z)); out_t, out_f = [], []
        for i in range(0, ph.size - 6, 6):
            seg_t = tc[i:i + 6]
            if np.ptp(ts[i:i + 6]) > 70: continue                    # crosses a stamp gap: skip
            p = np.polyfit(seg_t, ph[i:i + 6], 1); out_t.append(ts[i + 3]); out_f.append(p[0] / (2 * np.pi * h * 60) * 1e6)
        return np.array(out_t), np.array(out_f)
    tS, fS = freq(S[2], S[3], S[4], 2); tB, fB = freq(B[2], B[3], B[4], 1)
    j = np.searchsorted(tS, tB).clip(1, tS.size - 1)
    j = np.where(np.abs(tS[j - 1] - tB) < np.abs(tS[j] - tB), j - 1, j)
    ok = np.abs(tS[j] - tB) < 30
    a, b = fS[j[ok]], fB[ok]
    lim = np.percentile(np.abs(np.r_[a, b] - np.median(np.r_[a, b])), 95)
    keep = (np.abs(a - np.median(a)) < lim) & (np.abs(b - np.median(b)) < lim)
    r = np.corrcoef(a[keep], b[keep])[0, 1]; slope = np.polyfit(a[keep], b[keep], 1)[0]
    print(f"case {c:>5} [{S[1].split('/')[1]} {B[1].split('/')[1]}] A: {keep.sum()} minutes; grid swing (SNUADC) SD {np.std(a[keep]):.0f} ppm; "
          f"BIS SD {np.std(b[keep]):.0f}; r = {r:+.2f}; slope {slope:+.2f}; median BIS-SNUADC {np.median(b[keep]-a[keep]):+.1f} (model {model:+.1f})")
    # --- B: BIS placed by count through stamp steps; phase difference to SNUADC (by its own stamps) at BIS block times
    tr = B[5]; t_count = tr.record_times[0] + B[3]                    # BIS time by count from its first stamp
    phS = np.unwrap(np.angle(S[4])) / 2.0
    dphi = np.angle(B[4]) - np.interp(t_count, S[2], phS)             # wrapped, then unwrapped below
    good = np.abs(B[4]) > 0.3 * np.median(np.abs(B[4]))
    d = np.unwrap(dphi[good]); tt = t_count[good]
    st_t, st_d = steps_of(tr)
    jumps = []
    for t0, dd in zip(st_t, st_d):
        pre = (tt > t0 - 300) & (tt < t0 - 20); post = (tt > t0 + 20) & (tt < t0 + 300)
        if pre.sum() > 10 and post.sum() > 10:
            jumps.append((dd * 1000, (np.median(d[post]) - np.median(d[pre])) / (2 * np.pi * 60) * 1000))
    if jumps:
        J = np.array(jumps)
        print(f"            B: {len(J)} BIS stamp steps (median size {np.median(J[:,0]):+.0f} ms); phase change across them, as time: "
              f"median {np.median(J[:,1]):+.1f} ms, |median| {np.median(np.abs(J[:,1])):.1f} ms; r(step, change) = "
              f"{np.corrcoef(J[:,0], J[:,1])[0,1] if len(J) > 2 else float('nan'):+.2f}")
    else:
        print("            B: no BIS stamp steps with clean phase on both sides")
