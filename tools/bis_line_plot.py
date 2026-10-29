# SPDX-License-Identifier: MIT
"""Figure: the BIS 60 Hz line against the grid seen through SNUADC, minute by minute.

Usage: python tools/bis_line_plot.py   (reads bis_line_minutes.npz; writes figs/bis-line-not-mains.png)
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
GREY, BLUE, INK, MUTED, GRID = "#9b9a94", "#2a78d6", "#1f2430", "#5a6170", "#e4e3df"
d = np.load("bis_line_minutes.npz"); ctl, bis = d["ctl"], d["bis"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, (x, y), title, ylab, lim in [(axs[0], ctl, "a  Control: two SNUADC channels, both 120 Hz", "second SNUADC channel (ppm)", 80),
                                     (axs[1], bis, "b  BIS EEG 60 Hz line against SNUADC 120 Hz", "BIS EEG line (ppm)", 300)]:
    rho = spearmanr(x, y)[0]; inl = (np.abs(x) < 60) & (np.abs(y) < 60); slope = np.polyfit(x[inl], y[inl], 1)[0] if inl.sum() > 20 else np.nan
    ax.plot(x, y, "o", ms=3, color=BLUE, alpha=0.35, mec="none")
    ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1, ls=(0, (4, 3)))
    ax.set_xlim(-80, 80); ax.set_ylim(-lim, lim)
    ax.set_title(title, fontsize=9, loc="left", fontweight="bold", pad=18)
    ax.text(0, 1.01, f"{x.size:,} minutes; rank correlation {rho:+.2f}; spread (SD) {np.std(x):.0f} and {np.std(y):.0f} ppm",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    ax.set_xlabel("grid frequency via SNUADC, per minute (ppm about the recording's median)", fontsize=7.5)
    ax.set_ylabel(ylab + ", same minute", fontsize=8)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    print(title, x.size, round(rho, 2), round(slope, 2))
axs[1].text(78, 280, "dashed: equal change\non both axes", fontsize=7, color=MUTED, ha="right", va="top")
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.84, wspace=0.3)
fig.savefig("figs/bis-line-not-mains.png", dpi=150, facecolor="white")
