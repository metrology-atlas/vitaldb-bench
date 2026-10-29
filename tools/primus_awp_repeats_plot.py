# SPDX-License-Identifier: MIT
"""Figure: Primus AWP repeated samples. A: an inspiratory upstroke with repeated samples marked. B: spectrum of the
repeated-sample pattern in three recordings (lines coincide). C: each line's frequency in 30 recordings, as the
difference from its median (mHz).

Usage: python tools/primus_awp_repeats_plot.py ROOT   (reads awp.jsonl; writes figs/awp-repeat-pattern.png)
"""
import sys, json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
GREY, BLUE, ORANGE, GREEN, PURPLE, INK, MUTED, GRID = "#9b9a94", "#2a78d6", "#e65100", "#2e7d32", "#7b3fa0", "#1f2430", "#5a6170", "#e4e3df"
root = Path(sys.argv[1]); F2 = 62.5
def style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
def title(ax, s): ax.set_title(s, fontsize=8.5, loc="left", fontweight="bold")
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.7), gridspec_kw=dict(width_ratios=[1.1, 1.5, 1]))
spec = {}
for c, col in ((497, PURPLE), (17, ORANGE), (2929, BLUE)):
    x = np.asarray(read_vital(root / "vital_files" / f"{c:04d}.vital", track_names={"Primus/AWP"}).tracks["Primus/AWP"].values, float)
    x = x[int(F2 * 600):int(F2 * 600) + int(F2 * 3600)]; d = np.diff(x)
    sw = ((d[1:-1] == 0) & (d[:-2] != 0) & (d[2:] != 0)).astype(float)
    if c == 497:
        st = np.abs(x[3:] - x[:-3]) >= 8; i = np.nonzero(st & (d[1:-1] == 0))[0]; i0 = int(i[150]) - 20; s = x[i0:i0 + 45]; tt = np.arange(s.size) * 16
        ax = axs[0]; ax.plot(tt, s, "-", color=GREY, lw=0.8); ax.plot(tt, s, "o", color=PURPLE, ms=3)
        rep = np.nonzero(np.diff(s) == 0)[0] + 1; ax.plot(tt[rep], s[rep], "o", color=ORANGE, ms=6, mfc="none", mew=1.4, label="repeats the previous value")
        ax.legend(fontsize=6.8, frameon=False, loc="upper left"); ax.set_xlabel("ms (stored samples, 16 ms apart)", fontsize=7.5)
        ax.set_ylabel("AWP (stored code)", fontsize=7.5); style(ax); title(ax, "A  A breath: some stored samples\nrepeat the previous value")
    c0 = sw - sw.mean(); F = np.abs(np.fft.rfft(c0, n=c0.size)); f = np.fft.rfftfreq(c0.size, 1 / F2)
    F = np.convolve(F, np.ones(25) / 25, "same"); F /= np.median(F[f > 2])
    axs[1].plot(f, F, color=col, lw=0.9, label=f"recording {c}")
ax = axs[1]; ax.set_xlim(2, 31.25); ax.set_xlabel("Hz", fontsize=7.5); ax.set_ylabel("strength of the repeat pattern\n(relative to background)", fontsize=7.5)
ax.legend(fontsize=6.8, frameon=False); style(ax); title(ax, "B  The repeat pattern has the same lines\nin every recording (not one clean rate)")
R = [json.loads(l) for l in open("awp.jsonl")]; R = [r for r in R if "lines" in r]
keys = ["4.23", "8.52", "12.76", "16.58", "24.58", "25.105", "28.93", "29.34"]
ax = axs[2]
for j, k in enumerate(keys):
    v = []
    for r in R:
        L = np.array(r["lines"][k]); g = L[L[:, 1] > 5]
        if g.size: v.append(np.median(g[:, 0]))
    v = np.array(v); ax.plot((v - np.median(v)) * 1000, np.full(v.size, j) + np.random.default_rng(j).uniform(-0.2, 0.2, v.size), "o", color=PURPLE, ms=2.5, alpha=0.7)
ax.set_yticks(range(len(keys))); ax.set_yticklabels([f"{np.median([np.median(np.array(r['lines'][k])[:,0]) for r in R]):.3f} Hz" for k in keys], fontsize=7)
ax.set_xlim(-10, 10); ax.axvline(0, color=GREY, lw=0.8); ax.set_xlabel("difference from median (mHz)", fontsize=6.8)
style(ax); title(ax, f"C  {len(R)} recordings: each line within\na few mHz of the others")
fig.text(0.01, 0.98, "Primus AWP: stored at 62.5 Hz with periodic repeated samples, in a fixed pattern that does not drift", fontsize=9, fontweight="bold", va="top")
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.80, wspace=0.35); fig.savefig("figs/awp-repeat-pattern.png", dpi=150, facecolor="white")
