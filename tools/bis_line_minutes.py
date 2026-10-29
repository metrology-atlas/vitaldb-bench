# SPDX-License-Identifier: MIT
"""Figure: the BIS 60 Hz line does not follow the mains grid. Minute-by-minute line frequency (ppm about each
recording's median): two SNUADC channels against each other (control), then BIS EEG against SNUADC. Pooled over the
pilot recordings; no case ids. Data are recomputed here (per-minute arrays saved locally as bis_line_minutes.npz).

Usage: python tools/bis_line_minutes.py ROOT CASE [CASE ...]   (saves bis_line_minutes.npz)
"""
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
GREY, BLUE, INK, MUTED, GRID = "#9b9a94", "#2a78d6", "#1f2430", "#5a6170", "#e4e3df"
root = Path(sys.argv[1]); CASES = [int(c) for c in sys.argv[2:]]

def minute_freq(tr, f_line, h):
    fs = float(tr.srate); rt, rs = tr.record_times, tr.record_sizes; starts = np.r_[0, np.cumsum(rs)[:-1]]
    x = np.nan_to_num(tr.physical().astype(float)); n = int(10 * fs); nb = x.size // n
    z = (x[: nb * n] * np.exp(-2j * np.pi * f_line * np.arange(nb * n) / fs)).reshape(nb, n).mean(1)
    ph = np.unwrap(np.angle(z)); tc = (np.arange(nb) * n + n / 2) / fs
    rec = np.searchsorted(starts, np.arange(nb) * n + n / 2, side="right") - 1
    ts = rt[rec] + (np.arange(nb) * n + n / 2 - starts[rec]) / fs
    T, F = [], []
    for i in range(0, nb - 6, 6):
        if np.ptp(ts[i:i + 6]) > 70: continue
        p = np.polyfit(tc[i:i + 6], ph[i:i + 6], 1); T.append(ts[i + 3]); F.append(p[0] / (2 * np.pi * h * 60) * 1e6)
    return np.array(T), np.array(F)

def pair(tA, fA, tB, fB):
    j = np.searchsorted(tA, tB).clip(1, tA.size - 1); j = np.where(np.abs(tA[j - 1] - tB) < np.abs(tA[j] - tB), j - 1, j)
    ok = np.abs(tA[j] - tB) < 30; a, b = fA[j[ok]] - np.median(fA[j[ok]]), fB[ok] - np.median(fB[ok])
    keep = (np.abs(a) < 300) & (np.abs(b) < 300); return a[keep], b[keep]

ctl, bis = [[], []], [[], []]
for c in CASES:
    v = read_vital(root / "vital_files" / f"{c:04d}.vital", track_names={"SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/CVP", "BIS/EEG1_WAV", "BIS/EEG2_WAV"})
    have = [k for k in ("SNUADC/ECG_V5", "SNUADC/ECG_II", "SNUADC/CVP") if v.tracks.get(k) is not None and v.tracks[k].record_times.size]
    bk = [k for k in ("BIS/EEG2_WAV", "BIS/EEG1_WAV") if v.tracks.get(k) is not None and v.tracks[k].record_times.size]
    if len(have) < 2 or not bk: continue
    S1 = minute_freq(v.tracks[have[0]], 120.0, 2); S2 = minute_freq(v.tracks[have[1]], 120.0, 2); B = minute_freq(v.tracks[bk[0]], 60.0, 1)
    a, b = pair(*S1, *S2); ctl[0] += list(a); ctl[1] += list(b)
    a, b = pair(*S1, *B); bis[0] += list(a); bis[1] += list(b)
ctl, bis = np.array(ctl), np.array(bis)
np.savez("bis_line_minutes.npz", ctl=ctl, bis=bis)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
for ax, (x, y), title, sub in [(axs[0], ctl, "a  Control: two SNUADC channels (120 Hz)", "second SNUADC channel"),
                               (axs[1], bis, "b  BIS EEG 60 Hz line against SNUADC (120 Hz)", "BIS EEG line")]:
    r = np.corrcoef(x, y)[0, 1]
    ax.plot(x, y, "o", ms=3, color=BLUE, alpha=0.35, mec="none")
    ax.plot([-150, 150], [-150, 150], color=GREY, lw=1, ls=(0, (4, 3)))
    ax.set_title(title, fontsize=9, loc="left", fontweight="bold", pad=18)
    ax.text(0, 1.01, f"{x.size:,} minutes; r = {r:+.2f}; spread (SD) {np.std(x):.0f} and {np.std(y):.0f} ppm", transform=ax.transAxes, fontsize=7.5, color=MUTED)
    ax.set_xlabel("grid frequency seen by SNUADC, minute by minute (ppm about the recording's median)", fontsize=7.5)
    ax.set_ylabel(f"{sub}, same minute (ppm)", fontsize=8)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_xlim(-80, 80); axs[0].set_ylim(-300, 300)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.84, wspace=0.25)
fig.savefig("figs/bis-line-not-mains.png", dpi=150, facecolor="white")
print(ctl.shape, bis.shape, np.corrcoef(*ctl)[0, 1], np.corrcoef(*bis)[0, 1])
