# SPDX-License-Identifier: MIT
"""Figure: BIS EEG values lie on a 0.025 uV grid. Left: a few seconds of stored values against the grid. Right: where
values fall between grid lines (0 = on a line), for all samples of 10 minutes, against the same values jittered.

Usage: python tools/bis_grid_plot.py ROOT CASE   (writes figs/bis-grid.png)
"""
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
GREY, BLUE, ORANGE, INK, MUTED, GRID = "#9b9a94", "#2a78d6", "#e65100", "#1f2430", "#5a6170", "#e4e3df"
root, case = Path(sys.argv[1]), int(sys.argv[2])
x = np.asarray(read_vital(root / "vital_files" / f"{case:04d}.vital", track_names={"BIS/EEG1_WAV"}).tracks["BIS/EEG1_WAV"].values, np.float64)
x = x[np.isfinite(x)][128 * 900: 128 * 1500]
g = 0.05
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
d = np.abs(np.diff(x)); d = d[(d > 0) & (d < 0.3)]
a.hist(d, bins=np.arange(0, 0.3001, 0.0025), color=BLUE)
for k in np.arange(1, 7) * g: a.axvline(k, color=GRID, lw=0.8, zorder=0)
a.set_title("Small steps between successive samples", fontsize=9, loc="left", fontweight="bold", pad=16)
a.text(0, 1.0, f"steps under 0.3 µV, 10 minutes; faint lines every 0.05 µV", transform=a.transAxes, fontsize=7.5, color=MUTED, va="bottom")
a.set_xlabel("step size (µV)", fontsize=7.5); a.set_ylabel("steps", fontsize=7.5); a.tick_params(labelsize=7)
frac = (x / g) - np.round(x / g)
jit = ((x + np.random.default_rng(0).uniform(-0.5, 0.5, x.size) * g) / g); jit = jit - np.round(jit)
bins = np.linspace(-0.5, 0.5, 51)
b.hist(jit, bins=bins, color=GREY, alpha=0.6, label="same values, randomly shifted")
b.hist(frac, bins=bins, color=BLUE, label="stored values")
b.set_yscale("log"); b.set_xlabel("position between two grid lines (0 = exactly on a line)", fontsize=7.5)
b.set_ylabel("samples (log scale)", fontsize=7.5); b.tick_params(labelsize=7)
b.set_title(f"Where all values fall (10 minutes, {x.size:,} samples)", fontsize=9, loc="left", fontweight="bold", pad=16)
b.legend(fontsize=7.5, frameon=False, loc="upper left")
for ax in (a, b):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.86, wspace=0.3)
fig.savefig("figs/bis-grid.png", dpi=150, facecolor="white")
print(np.mean(np.abs(frac) < 0.02))
