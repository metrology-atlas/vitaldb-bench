# SPDX-License-Identifier: MIT
"""Figure: the SNUADC rate from the mains hum against the assumed rate, checked by the BIS bursts (aggregates only).

    python tools/hum_snuadc_plot.py check.csv out.png

check.csv from tools/hum_snuadc_rate.py --check-out: recordings without SNUADC stamp corrections whose BIS rate comes
from its own corrections. (a) the drift left over each recording (ms per hour, the error of the SNUADC rate the stamp
model used) with the port's assumed rate and with the hum rate. (b) what the hum says the rate should change by,
against the drift the bursts show with the assumed rate: on the line, the hum predicts the drift exactly.
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GREY, BLUE, ORANGE, INK, MUTED, GRID = "#9b9a94", "#2a78d6", "#eb6834", "#1f2430", "#5a6170", "#e4e3df"


def main(check, out):
    c = pd.read_csv(check)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.arange(-300, 301, 15)
    for col, colour, lab in (("err_assumed_ppm", GREY, "port's assumed rate"), ("err_hum_ppm", BLUE, "hum rate")):
        e = 3.6 * c[col].values
        a.hist(np.clip(e, -300, 300), bins=bins, histtype="stepfilled", color=colour, alpha=0.3, lw=0)
        a.hist(np.clip(e, -300, 300), bins=bins, histtype="step", color=colour, lw=1.3)
        a.plot([], [], color=colour, lw=6, alpha=0.5, label=f"{lab}: median |drift| {np.median(np.abs(e)):.0f} ms/h")
    a.axvline(0, color=INK, lw=0.9)
    a.legend(frameon=False, fontsize=7.5, loc="upper left")
    a.set_xlabel("drift left over the recording (ms per hour)", fontsize=8); a.set_ylabel("recordings", fontsize=8)
    a.set_title("a  Drift of the working reference against the BIS", fontsize=9, loc="left", fontweight="bold", pad=18)
    a.text(0, 1.01, f"{len(c)} recordings without SNUADC corrections; values beyond the axis at its edge",
           transform=a.transAxes, fontsize=7.5, color=MUTED)
    x, y = c.hum_minus_assumed_ppm.values, c.err_assumed_ppm.values
    b.plot([-80, 80], [-80, 80], color=GREY, lw=1, ls=(0, (4, 3)))
    for md, colour in (("step", BLUE), ("none", ORANGE)):
        s = c["mode"] == md
        b.plot(x[s], y[s], "o", ms=3.5, color=colour, alpha=0.6, mew=0, label=f"SNUADC mode {md} (n = {int(s.sum())})")
    b.set_xlim(-80, 80); b.set_ylim(-80, 80)
    b.axhline(0, color=INK, lw=0.6); b.axvline(0, color=INK, lw=0.6)
    b.legend(frameon=False, fontsize=7.5, loc="upper left")
    b.set_xlabel("hum rate minus the assumed rate (ppm)", fontsize=8)
    b.set_ylabel("drift with the assumed rate, from the bursts (ppm)", fontsize=8)
    b.set_title("b  The hum predicts the drift", fontsize=9, loc="left", fontweight="bold", pad=18)
    for ax in (a, b):
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.86, wspace=0.28)
    fig.savefig(out, dpi=150, facecolor="white")


if __name__ == "__main__":
    main(*sys.argv[1:3])
