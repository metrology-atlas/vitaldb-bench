# SPDX-License-Identifier: MIT
"""Figure: Primus CO2 and airway pressure samples (62.5 Hz) - how long each stored value lasts.

Usage: python tools/primus_holds_plot.py ROOT CASE   (writes figs/primus-holds.png)
"""
import sys, json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
GREY, BLUE, ORANGE, INK, MUTED, GRID = "#9b9a94", "#2a78d6", "#e65100", "#1f2430", "#5a6170", "#e4e3df"
root, case = Path(sys.argv[1]), int(sys.argv[2])
v = read_vital(root / "vital_files" / f"{case:04d}.vital", track_names={"Primus/CO2", "Primus/AWP"})
fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1, 0.9]})
for ax, (k, name) in zip(axs[:2], [("Primus/CO2", "CO2"), ("Primus/AWP", "Airway pressure")]):
    x = np.asarray(v.tracks[k].values, float); n0 = int(62.5 * 900); seg = x[n0: n0 + int(62.5 * 300)]
    d = np.abs(np.diff(seg)); w = np.convolve(d, np.ones(40), "valid"); i = int(np.argmax(w)); y = seg[i: i + 40]
    t = np.arange(y.size) * 16.0; rep = np.r_[False, np.diff(y) == 0]
    ax.step(t, y, where="post", color=GREY, lw=1)
    ax.plot(t[~rep], y[~rep], "o", color=BLUE, ms=5, mec="white", mew=0.8, label="new value")
    ax.plot(t[rep], y[rep], "o", color=ORANGE, ms=5, mec="white", mew=0.8, label="same as the previous sample")
    ax.set_title(f"{name}: its steepest 640 ms", fontsize=9, loc="left", fontweight="bold", pad=16)
    ax.text(0, 1.0, f"{rep.sum()} of {y.size} samples repeat", transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
    ax.set_xlabel("ms (one sample = 16 ms)", fontsize=7.5); ax.tick_params(labelsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("stored value (code)", fontsize=7.5)
# how long a value lasts, pooled over the 200-recording test
H = {"Primus/CO2": np.zeros(6), "Primus/AWP": np.zeros(6)}
for f in ("mid_0.jsonl", "mid_1.jsonl"):  # from waveform_deltas.py on the 200-recording test (every 32nd case), in the current folder
    for l in open(f):
        d = json.loads(l)
        for k in H:
            s = d["tracks"].get(k)
            if s and s["held_runs_1_to_6plus"]: H[k] += np.array(s["held_runs_1_to_6plus"])
b = axs[2]; xs = np.arange(1, 6)
for off, (k, col, name) in zip((-0.2, 0.2), [("Primus/CO2", BLUE, "CO2"), ("Primus/AWP", ORANGE, "Airway pressure")]):
    h = H[k][1:6] / H[k][1:].sum() * 100
    b.bar(xs + off, h, width=0.38, color=col, label=name)
b.set_xticks(xs, ["2", "3", "4", "5", "6+"]); b.set_xlabel("a repeated value lasts this many samples", fontsize=7.5)
b.set_ylabel("share of repeated values (%)", fontsize=7.5); b.tick_params(labelsize=7)
b.set_title("How long repeats last", fontsize=9, loc="left", fontweight="bold", pad=16)
b.text(0, 1.0, "200 recordings", transform=b.transAxes, fontsize=7.5, color=MUTED, va="bottom")
b.legend(fontsize=7.5, frameon=False); b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True)
for s in ("top", "right"): b.spines[s].set_visible(False)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower left", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.05, 0))
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.25, top=0.86, wspace=0.3)
fig.savefig("figs/primus-holds.png", dpi=150, facecolor="white")
