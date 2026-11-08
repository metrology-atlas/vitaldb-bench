# SPDX-License-Identifier: MIT
"""Figure: the recording PCs' clock rate against the mains grid (aggregates only; no case ids).

    python tools/mains_hum_plot.py per_recording.csv out.png [--setup-min-n 30]

per_recording.csv from tools/mains_hum.py agg (its rate_src and snu_rate_setup columns). g = r - line = f_PC - f_grid
(positive = the PC runs fast). (a) g by where the SNUADC rate r comes from: the recording's own whole teeth (two
corrections in one stretch); its setup's rate, borrowed (the median of the setup's whole-teeth rates, the recording
left out); or neither, where r is a line fitted through a one-step staircase, which carries no rate
(examples/sim_stamp_clock_onecorr.py). (b) median and IQR by recording setup, own and borrowed rates together, setups
with at least --setup-min-n recordings.
"""
import argparse
import collections

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GREY, BLUE, ORANGE, INK, MUTED, GRID = "#9b9a94", "#2a78d6", "#eb6834", "#1f2430", "#5a6170", "#e4e3df"


def main(table, out, setup_min_n):
    m = pd.read_csv(table)
    m = m[m.snu_line.notna() & m.snu_rate.notna() & (m.snu_events >= 1)].copy()
    m["g"] = m.snu_rate - m.snu_line
    src = m.rate_src.fillna("fit")
    m["g_used"] = np.where(src == "teeth", m.snu_rate, np.where(src == "setup", m.snu_rate_setup, np.nan)) - m.snu_line
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    bins = np.arange(-120, 101, 4)
    for sel, col, lab, gcol in ((src == "teeth", BLUE, "own whole teeth", "g_used"),
                                (src == "setup", ORANGE, "setup's rate, borrowed", "g_used"),
                                (src == "fit", GREY, "neither (line fit, not a rate)", "g")):
        g = m[sel][gcol].values
        a.hist(np.clip(g, -120, 100), bins=bins, histtype="stepfilled", color=col, alpha=0.25, lw=0)
        a.hist(np.clip(g, -120, 100), bins=bins, histtype="step", color=col, lw=1.2)
        med = np.median(g); a.axvline(med, color=col, lw=1.2, ymax=0.78)
        a.plot([], [], color=col, lw=6, alpha=0.4, label=f"{lab}: n = {g.size:,}, median {med:+.1f} ppm")
    a.axvline(0, color=GREY, lw=1, ls=(0, (4, 3)), ymax=0.78)
    a.set_ylim(0, a.get_ylim()[1] * 1.3)
    a.legend(frameon=False, fontsize=7.5, loc="upper right")
    a.set_xlabel("PC clock rate against the mains grid (ppm; positive = PC runs fast)", fontsize=8)
    a.set_ylabel("recordings", fontsize=8)
    a.set_title("a  By how the SNUADC rate was found", fontsize=9, loc="left", fontweight="bold", pad=18)
    a.text(0, 1.01, "recordings with mains hum and a measured SNUADC rate; values beyond the axis at its edge",
           transform=a.transAxes, fontsize=7.5, color=MUTED)
    by = collections.defaultdict(list)
    used = m.g_used.notna()
    for k, v in zip(m.key[used], m.g_used[used]):
        by[k].append(v)
    keys = sorted([k for k, v in by.items() if len(v) >= setup_min_n], key=lambda k: np.median(by[k]))
    for i, k in enumerate(keys):
        v = np.array(by[k]); q1, med, q3 = np.percentile(v, [25, 50, 75])
        b.plot([q1, q3], [i, i], color=BLUE, lw=2, solid_capstyle="round")
        b.plot(med, i, "o", color=BLUE, ms=6, mec="white", mew=1.5)
        port = dict(p.split(":") for p in k.split()).get("SNUADC", "?")
        b.text(-82, i, f"setup {chr(65 + i)} (SNUADC {port}, n = {len(v)})", fontsize=7.5, va="center", ha="right", color=INK)
    b.axvline(0, color=GREY, lw=1, ls=(0, (4, 3))); b.set_yticks([]); b.set_xlim(-80, 50)
    b.set_xlabel("PC clock rate against the grid (ppm)\nmedian and interquartile range", fontsize=8)
    b.set_title(f"b  By setup ({setup_min_n}+ recordings)", fontsize=9, loc="left", fontweight="bold", pad=18)
    for ax in (a, b):
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    b.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.86, wspace=0.62)
    fig.savefig(out, dpi=150, facecolor="white")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("table"); ap.add_argument("out"); ap.add_argument("--setup-min-n", type=int, default=30)
    x = ap.parse_args()
    main(x.table, x.out, x.setup_min_n)
