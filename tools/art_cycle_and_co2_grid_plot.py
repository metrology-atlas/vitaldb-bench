# SPDX-License-Identifier: MIT
"""Figures for the two small tests (arterial sawtooth; CO2 update grid). Reads saw2.jsonl and c<case>.npz.

Usage: python tools/art_cycle_and_co2_grid_plot.py   (in OUTDIR; writes figs/art-sawtooth-test.png and figs/co2-update-grid.png)
"""
import json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
GREY, BLUE, ORANGE, GREEN, PURPLE, INK, MUTED, GRID = "#9b9a94", "#2a78d6", "#e65100", "#2e7d32", "#7b3fa0", "#1f2430", "#5a6170", "#e4e3df"
R = {d["case"]: d for d in map(json.loads, open("saw2.jsonl"))}
def style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
def title(ax, s): ax.set_title(s, fontsize=8.5, loc="left", fontweight="bold")
# ---------------- Figure 1: arterial sawtooth ----------------
EX = 497; d = np.load(f"c{EX}.npz"); tr, res, tb, psi = d["tr"], d["res"], d["tb"], d["psi"]
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.7), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
ax = axs[0]; w = (tr > 1200) & (tr < 1800); t, r = tr[w], res[w]
g = np.arange(t.size) // 30; tm = np.array([t[g == k].mean() for k in np.unique(g)]); rm = np.array([r[g == k].mean() for k in np.unique(g)])
ax.plot((t - 1200) / 60, r, ".", color=GREY, ms=1.5, alpha=0.5, label="single beats")
ax.plot((tm - 1200) / 60, rm, "-o", color=ORANGE, ms=3, lw=1.4, label="30-beat averages (measured)")
ph = (np.interp(t, tb, psi) / (2 * np.pi)) % 1; pred = (ph - 0.5) * 1000 / 240
ax.plot((t - 1200) / 60, pred - 8, color=BLUE, lw=1.4, label="what a 4.17 ms sawtooth would look like (shifted down)")
ax.set_ylim(-14, 9); ax.set_xlabel("minutes", fontsize=7.5); ax.set_ylabel("ECG to arterial upstroke delay,\nchange from 3-min median (ms)", fontsize=7.5)
ax.legend(fontsize=6.8, frameon=False, loc="upper right", ncol=1); style(ax)
title(ax, f"A  One recording, 10 min: no sawtooth\n(its 240 Hz cycle turns every {R[EX]['cycle_period_s']:.0f} s)")
ax = axs[1]; a = [v for v in R.values() if v.get("beats", 0) > 1000]
x = [v["control_ptp_ms"] for v in a]; y = [v["profile_ptp_ms"] for v in a]
ax.scatter(x, y, s=16, color=ORANGE, zorder=3); ax.plot([0, 5], [0, 5], color=GREY, lw=0.8, ls=(0, (4, 3)))
ax.axhline(4.17, color=BLUE, lw=1.2); ax.text(0.1, 4.3, "expected if arterial timing\nfollowed the 240 Hz cycle", color=BLUE, fontsize=6.8)
ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_xlabel("control: same, wrong alignment (ms)", fontsize=7.5)
ax.set_ylabel("swing of delay over the 240 Hz cycle (ms)", fontsize=7.5); style(ax); title(ax, f"B  {len(a)} recordings:\nswing no bigger than control")
ax = axs[2]; ex, ob = [], []
for v in a:
    z = np.load(f"c{v['case']}.npz"); t2, dl = z["tr"], z["dl"]
    if t2.max() < 2400: continue
    ob.append(np.median(dl[t2 > t2.max() - 600]) - np.median(dl[t2 < 600])); ex.append(v["cycle_rate_ppm"] * 1e-6 * (t2.max() - 600) * 1000)
ax.scatter(ex, ob, s=16, color=ORANGE, zorder=3); ax.plot([-300, 0], [-300, 0], color=BLUE, lw=1.2)
ax.text(-290, -230, "expected if arterial\ntiming drifted with\nthe 240 Hz clock", color=BLUE, fontsize=6.8)
ax.axhline(0, color=GREY, lw=0.8); ax.set_xlim(-300, 20); ax.set_ylim(-300, 100)
ax.set_xlabel("drift expected over the recording (ms)", fontsize=7.5); ax.set_ylabel("observed change in delay,\nfirst vs last 10 min (ms)", fontsize=7.5)
style(ax); title(ax, f"C  {len(ob)} recordings:\nno clock drift")
fig.text(0.01, 0.98, "Arterial pressure keeps time with the ECG: neither the 4.17 ms sawtooth nor the 25-58 ppm drift appears", fontsize=9, fontweight="bold", va="top")
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.80, wspace=0.32); fig.savefig("figs/art-sawtooth-test.png", dpi=150, facecolor="white")
# ---------------- Figure 2: CO2 update grid ----------------
d = np.load(f"c{EX}.npz"); seg, tbc, pc = d["co2seg"], d["tbc"], d["pc"]
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.7), gridspec_kw=dict(width_ratios=[1.2, 1.4, 1]))
ax = axs[0]; ch = np.nonzero(np.diff(seg) != 0)[0] + 1; dd = np.diff(seg); steep = np.convolve((dd != 0).astype(float), np.ones(32), "valid"); i0 = int(np.argmax(steep)); s = seg[i0:i0 + 32]; tt = np.arange(s.size) * 16
ax.step(tt, s, where="post", color=PURPLE, lw=1.4); ax.plot(tt, s, "o", color=PURPLE, ms=3)
for k in np.arange(0, tt[-1], 40): ax.axvline(k + 8, color=GRID, lw=0.8, zorder=0)
ax.set_xlabel("ms (dots = stored samples, 16 ms apart; lines = a 40 ms grid)", fontsize=7.5); ax.set_ylabel("CO2 (stored code)", fontsize=7.5)
style(ax); title(ax, "A  On an upstroke, CO2 changes every\n2 or 3 stored samples (~40 ms)")
ax = axs[1]; c = (np.diff(seg) != 0).astype(float); act = np.convolve(c, np.ones(25), "same") > 0; c = c - c[act].mean(); c[~act] = 0
k = np.arange(c.size); B = 62; nb = c.size // B; z = (c * np.exp(-2j * np.pi * 25 * k / 62.5))[:nb * B].reshape(nb, B).mean(1)
tt2 = (np.arange(nb) + 0.5) * B / 62.5; off = ((-np.angle(z) / (2 * np.pi)) % 1) * 40; strong = np.abs(z) > 0.05
ax.plot(tt2[strong], off[strong], "o", color=PURPLE, ms=2.5); ax.set_ylim(0, 40); ax.set_yticks([0, 10, 20, 30, 40])
ax.set_xlabel("seconds (one dot per second of breathing)", fontsize=7.5); ax.set_ylabel("where each 40 ms update starts,\nrelative to an exact 25 Hz grid (ms)", fontsize=7.5)
style(ax); title(ax, f"B  The update grid slides (8 ms steps = 16 ms sampling):\none full 40 ms wrap every {R[EX]['co2_slip_period_s']:.0f} s")
ax = axs[2]; hz = [float(np.median(v["co2_update_hz"])) for v in R.values() if "co2_update_hz" in v]
ax.hist(hz, bins=np.arange(24.960, 24.9905, 0.002), color=PURPLE); ax.axvline(25, color=GREY, lw=1, ls=(0, (4, 3)))
ax.text(24.9985, ax.get_ylim()[1] * 0.85, "exactly\n25 Hz", fontsize=6.8, color=MUTED, ha="right")
ax.set_xlim(24.958, 25.003); ax.set_xlabel("CO2 update rate on the file's clock (Hz)", fontsize=7.5); ax.set_ylabel("recordings", fontsize=7.5)
style(ax); title(ax, f"C  {len(hz)} recordings:\nmedian {np.median(hz):.3f} Hz")
fig.text(0.01, 0.98, "Primus CO2 is updated about 25 times a second, but not locked to the stored 62.5 Hz samples", fontsize=9, fontweight="bold", va="top")
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.80, wspace=0.3); fig.savefig("figs/co2-update-grid.png", dpi=150, facecolor="white")
