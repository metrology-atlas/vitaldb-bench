# SPDX-License-Identifier: MIT
"""Figure: BIS EEG averaged around paced beats (SNUADC ECG), one 3-minute window per recording (the one with the
largest beat-locked swing), against the spread of averages at random times (grey band, 5-95%). Recordings are
lettered; ids stay in the study's records list.

Usage: python tools/paced_bis_plot.py ROOT   (writes figs/paced-bis-averages.png)
"""
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open(Path(__file__).resolve().parent / "paced_bis_spikes_control.py").read().split("for c, P in")[0])
GREY, BLUE, INK, MUTED, GRID = "#9b9a94", "#2a78d6", "#1f2430", "#5a6170", "#e4e3df"
CASES = [(830, 1.002), (1219, 0.998), (2701, 0.860), (4368, 1.204), (4631, 1.000), (6378, 0.860)]
fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.2), sharex=True)
lags = np.arange(int(2.2 * 128)) / 128 - 1.6
for ax, (c, P), lab in zip(axs.flat, CASES, "ABCDEF"):
    v = read_vital(root / "vital_files" / f"{c:04d}.vital", track_names={"SNUADC/ECG_II", "BIS/EEG1_WAV"})
    e = v.tracks["SNUADC/ECG_II"]; te = sample_times(e); xe = np.nan_to_num(e.physical().astype(float))
    beats = te[r_peaks(xe, 500.0)]; pb = beats[np.r_[False, np.abs(np.diff(beats) - P) < 0.012]]
    b = v.tracks["BIS/EEG1_WAV"]; tb = sample_times(b); xb = np.nan_to_num(b.physical().astype(float))
    xb = signal.sosfiltfilt(signal.butter(2, 1, "highpass", fs=128.0, output="sos"), xb)
    best = None
    for w0 in np.arange(pb[0], pb[-1], 180):
        sel = pb[(pb >= w0) & (pb < w0 + 180)]
        if sel.size < 60: continue
        A = trig_avg(tb, xb, 128.0, sel)
        if not A: continue
        r0 = np.random.default_rng(int(w0))
        cp = max(np.ptp(trig_avg(tb, xb, 128.0, np.sort(r0.uniform(sel[0], sel[-1], sel.size)))[0]) for _ in range(12))
        score = np.ptp(A[0]) / cp
        if best is None or score > best[2]: best = (A[0], sel, score)
    a, sel, score = best
    rng = np.random.default_rng(0)
    C = np.stack([trig_avg(tb, xb, 128.0, np.sort(rng.uniform(sel[0], sel[-1], sel.size)))[0] for _ in range(40)])
    lo, hi = np.percentile(C - np.median(C, 1, keepdims=True), [5, 95], axis=0)
    ax.fill_between(lags * 1000, lo, hi, color=GREY, alpha=0.35, lw=0)
    ax.plot(lags * 1000, a - np.median(a), color=BLUE, lw=1.4)
    ax.set_title(f"{lab}: {sel.size} beats; {score:.1f}x random", fontsize=8.5, loc="left", fontweight="bold")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
for ax in axs[1]: ax.set_xlabel("ms from the paced beat (SNUADC ECG, raw stamps)", fontsize=7.5)
for ax in axs[:, 0]: ax.set_ylabel("BIS EEG1 average (µV)", fontsize=7.5)
fig.text(0.01, 0.985, "BIS EEG averaged around paced beats (blue), in each recording's clearest 3-minute window, against averages at random times (grey, 5-95%)",
         fontsize=9, fontweight="bold", va="top")
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.11, top=0.88, hspace=0.35, wspace=0.22)
fig.savefig("figs/paced-bis-averages.png", dpi=150, facecolor="white")
