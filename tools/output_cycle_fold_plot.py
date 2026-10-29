# SPDX-License-Identifier: MIT
"""Figure: how much each channel changes at each point of the monitor's 240 Hz output cycle.
The cycle (frequency and phase) is measured from the arterial pressure's change pattern in the same recording; every
channel's step size (relative to its local average step) is then averaged by position in that cycle. A flat line:
the channel ignores the 240 Hz cycle; a swing: it changes mainly at one point of each cycle.

Usage: python tools/output_cycle_fold_plot.py ROOT CASE [CASE ...]   (writes figs/fold-240.png)
"""
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import signal
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
GREY, BLUE, ORANGE, GREEN, PURPLE, INK, MUTED, GRID = "#9b9a94", "#2a78d6", "#e65100", "#2e7d32", "#7b3fa0", "#1f2430", "#5a6170", "#e4e3df"
root = Path(sys.argv[1]); cases = [int(c) for c in sys.argv[2:]]
CH = [("SNUADC/ECG_II", "ECG II", BLUE), ("SNUADC/ART", "Arterial", ORANGE), ("SNUADC/PLETH", "Pleth", GREEN), ("SNUADC/CVP", "CVP", PURPLE)]
fig, axs = plt.subplots(1, len(cases), figsize=(3.4 * len(cases), 3.6), sharey=True)
for ax, case in zip(np.atleast_1d(axs), cases):
    v = read_vital(root / "vital_files" / f"{case:04d}.vital", track_names={k for k, _, _ in CH})
    N = 500 * 600; n0 = 500 * 900
    ref = np.asarray(v.tracks["SNUADC/ART"].values, float)[n0:n0 + N]
    rel = lambda x: np.abs(np.diff(x)) / (np.convolve(np.abs(np.diff(x)), np.ones(25) / 25, "same") + 1e-9)
    r = rel(ref); n = np.arange(r.size)
    # find the cycle: strongest line of the relative-step series near 240 Hz (step 0.001 Hz), then its phase
    best = None
    for f in np.arange(239.95, 240.05, 0.0005):
        c = np.abs(np.mean((r - r.mean()) * np.exp(-2j * np.pi * f * n / 500)))
        if best is None or c > best[0]: best = (c, f)
    f = best[1]; ph0 = np.angle(np.mean((r - r.mean()) * np.exp(-2j * np.pi * f * n / 500)))
    bins = np.linspace(0, 1, 13)
    for k, name, col in CH:
        t = v.tracks.get(k)
        if t is None or not t.record_times.size: continue
        x = np.asarray(t.values, float)[n0:n0 + N]
        if np.ptp(x) == 0: continue
        rr = rel(x); m = np.abs(np.diff(x)) > 0
        phase = ((f * n / 500 + ph0 / (2 * np.pi)) % 1.0)
        idx = np.digitize(phase, bins) - 1
        prof = [np.mean(rr[(idx == b)]) for b in range(12)]
        ax.plot((bins[:-1] + bins[1:]) / 2 * 1000 / 240, prof, "-o", color=col, ms=3.5, lw=1.6, label=name)
    ax.axhline(1, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax.set_title(f"recording {chr(65 + cases.index(case))}  (cycle {f:.3f} Hz)", fontsize=8.5, loc="left", fontweight="bold")
    ax.set_xlabel("position in the 240 Hz cycle (ms, one cycle = 4.17 ms)", fontsize=7.5)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
np.atleast_1d(axs)[0].set_ylabel("step size relative to its local average", fontsize=7.5)
h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
fig.text(0.01, 0.98, "How much each channel changes at each point of the monitor's 240 Hz output cycle (10 minutes per recording)",
         fontsize=9, fontweight="bold", va="top")
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.27, top=0.83, wspace=0.12)
fig.savefig("figs/fold-240.png", dpi=150, facecolor="white")
