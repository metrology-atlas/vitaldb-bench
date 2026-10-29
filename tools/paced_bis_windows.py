# SPDX-License-Identifier: MIT
"""Per 3-min window: is the BIS EEG average around paced beats beyond 40 random-time averages? And does its sharpest
feature repeat at the pacing period (a narrow spike train) and how wide is it (samples)? This machine only.

Usage: python tools/paced_bis_windows.py ROOT   (per 3-min window, against 40 random-time averages)
"""
import sys, json
from pathlib import Path
import numpy as np
exec(open(Path(__file__).resolve().parent / "paced_bis_spikes_control.py").read().split("for c, P in")[0])
for c, P in [(830, 1.002), (1219, 0.998), (2701, 0.860), (4368, 1.204), (4631, 1.000), (6378, 0.860)]:
    v = read_vital(root / "vital_files" / f"{c:04d}.vital", track_names={"SNUADC/ECG_II", "BIS/EEG1_WAV", "BIS/EEG2_WAV"})
    e = v.tracks["SNUADC/ECG_II"]; te = sample_times(e); xe = np.nan_to_num(e.physical().astype(float))
    beats = te[r_peaks(xe, 500.0)]; pb = beats[np.r_[False, np.abs(np.diff(beats) - P) < 0.012]]
    out = {"case": c}
    for k in ("BIS/EEG1_WAV", "BIS/EEG2_WAV"):
        b = v.tracks[k]; tb = sample_times(b); xb = np.nan_to_num(b.physical().astype(float))
        xb = signal.sosfiltfilt(signal.butter(2, 1, "highpass", fs=128.0, output="sos"), xb)
        n = sig = spike = 0; widths = []
        rng = np.random.default_rng(1)
        for w0 in np.arange(pb[0], pb[-1], 180):
            sel = pb[(pb >= w0) & (pb < w0 + 180)]
            if sel.size < 60: continue
            A = trig_avg(tb, xb, 128.0, sel)
            if not A: continue
            n += 1; a = A[0] - np.median(A[0])
            C = [trig_avg(tb, xb, 128.0, np.sort(rng.uniform(sel[0], sel[-1], sel.size)))[0] for _ in range(40)]
            cp = np.array([np.ptp(x) for x in C])
            if np.ptp(a) > np.percentile(cp, 97.5) * 1.0 and np.ptp(a) > np.max(cp):
                sig += 1
                d2 = np.abs(np.diff(a, 2)); i = int(np.argmax(d2)) + 1          # sharpest point
                j = i + int(round(P * 128)); jm = i - int(round(P * 128))
                rep = [x for x in (j, jm) if 1 <= x < a.size - 1]
                if rep and max(d2[x - 1] for x in rep) > 0.5 * d2[i - 1]:       # repeats one period away
                    spike += 1
                    half = np.abs(a[i]) / 2; L = R = i
                    while L > 0 and np.abs(a[L]) > half: L -= 1
                    while R < a.size - 1 and np.abs(a[R]) > half: R += 1
                    widths.append(R - L)
        out[k.split("/")[1]] = {"windows": n, "beat_locked_beyond_all_controls": sig, "with_a_sharp_feature_repeating_at_the_period": spike,
                                "feature_width_samples_median": float(np.median(widths)) if widths else None}
    print(json.dumps(out))
