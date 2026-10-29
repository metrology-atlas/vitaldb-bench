# SPDX-License-Identifier: MIT
"""Figure: stored samples at the steepest moments of each SNUADC channel in one recording; samples that repeat the
previous value marked. Shows the 240 Hz output holds directly.

Usage: python tools/waveform_staircase_plot.py ROOT CASE   (writes figs/staircase.png)
"""
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
GREY, BLUE, ORANGE, INK, MUTED, GRID = "#9b9a94", "#2a78d6", "#e65100", "#1f2430", "#5a6170", "#e4e3df"
root, case = Path(sys.argv[1]), int(sys.argv[2])
v = read_vital(root / "vital_files" / f"{case:04d}.vital", track_names={"SNUADC/ECG_II", "SNUADC/ART", "SNUADC/PLETH", "SNUADC/CVP"})
order = [("SNUADC/ECG_II", "ECG lead II"), ("SNUADC/ART", "Arterial pressure"), ("SNUADC/PLETH", "Plethysmogram"), ("SNUADC/CVP", "Central venous pressure")]
order = [o for o in order if v.tracks.get(o[0]) is not None and v.tracks[o[0]].record_times.size]
fig, axs = plt.subplots(1, len(order), figsize=(3.1 * len(order), 3.6))
N = 24                                                          # 24 samples = 48 ms
for ax, (k, name) in zip(np.atleast_1d(axs), order):
    x = np.asarray(v.tracks[k].values, float); n0 = 600 * 500; seg = x[n0: n0 + 500 * 600]
    d = np.abs(np.diff(seg)); w = np.convolve(d, np.ones(N), "valid")
    i = int(np.argmax(w)); y = seg[i: i + N]; t = np.arange(N) * 2.0
    rep = np.r_[False, np.diff(y) == 0]
    ax.step(t, y, where="post", color=GREY, lw=1)
    ax.plot(t[~rep], y[~rep], "o", color=BLUE, ms=5, mec="white", mew=0.8, label="new value")
    ax.plot(t[rep], y[rep], "o", color=ORANGE, ms=5, mec="white", mew=0.8, label="same as the previous sample")
    for u in np.arange(0, t[-1] + 2, 1000 / 240): ax.axvline(u, color=GRID, lw=0.8, zorder=0)
    ax.set_title(name, fontsize=9, loc="left", fontweight="bold", pad=16)
    ax.text(0, 1.0, f"{rep.sum()} of {N} samples repeat", transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
    ax.set_xlabel("ms (one sample = 2 ms)", fontsize=7.5); ax.tick_params(labelsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
np.atleast_1d(axs)[0].set_ylabel("stored value (converter code)", fontsize=7.5)
h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
fig.text(0.01, 0.98, "The steepest 48 ms of each channel, sample by sample (faint vertical lines: a 240 Hz grid, for scale)",
         fontsize=9, fontweight="bold", va="top")
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.78, wspace=0.35)
fig.savefig("figs/staircase.png", dpi=150, facecolor="white")
