# SPDX-License-Identifier: MIT
"""Figure of the clock-network reconciliation (tools/clock_reconcile.py output).

    python tools/clock_reconcile_figs.py RUN_DIR --out FIG.png

A: misclosure (ppm) of each BIS-SNUADC-PC loop through the SNUADC resets.  B: the loops through the Solar8000 route
(tools/clock_edges.py --solar-min-h), when the run has them: SNUADC-PC, resets against the Solar8000, and
BIS-SNUADC-PC without the SNUADC stamps.  C: the two Primus-PC routes.  D: misclosure divided by its standard error,
every loop type, against a standard normal (what the loops would show if every edge's standard error were right).
E: each method's pooled variance factor (1 = errors as stated).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, GREEN, RED, ORANGE = "#2a78d6", "#2f9e44", "#c2255c", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LOOPS = {"bis_steps + bursts + resets": ("BIS – SNUADC – PC, through the resets", BLUE),
         "resets + solar": ("SNUADC – PC, resets against Solar8000", GREEN),
         "bis_steps + bursts + solar": ("BIS – SNUADC – PC, through Solar8000", RED),
         "breath + pip_numeric": ("Primus – PC, two routes", ORANGE)}
NAMES = {"resets": "SNUADC resets", "bis_steps": "BIS correction steps", "bursts": "ECG–EEG bursts",
         "breath": "Breath matching", "pip_numeric": "PIP numerics", "arithmetic": "Stamp arithmetic",
         "solar": "Solar8000 route"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    C = pd.read_csv(run / "cycles.csv")
    S = json.loads((run / "summary.json").read_text())
    have = [k for k in LOOPS if (C.loop == k).any()]
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.3)
    panels = [("A", ["bis_steps + bursts + resets"], 50, 2), ("B", ["resets + solar", "bis_steps + bursts + solar"], 120, 5),
              ("C", ["breath + pip_numeric"], 400, 20)]
    for k, (tag, loops, lim, step) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        loops = [lp for lp in loops if lp in have]
        for lp in loops:
            g = C[C.loop == lp].misclosure
            name, col = LOOPS[lp]
            kw = dict(color=col, edgecolor="white", lw=0.6) if len(loops) == 1 else dict(histtype="step", lw=1.8, color=col)
            ax.hist(g.clip(-lim, lim), bins=np.arange(-lim, lim + step, step), label=f"{name}: {len(g):,}, {g.median():+.1f}",
                    **kw)
        ax.axvline(0, color=INK, lw=0.8)
        if len(loops) == 1:
            g = C[C.loop == loops[0]].misclosure
            ax.set_title(f"{tag}. {LOOPS[loops[0]][0]}\n{len(g):,} loops, median {g.median():+.1f} ppm", fontsize=9,
                         loc="left", color=INK)
        else:
            ax.set_title(f"{tag}. Through the Solar8000 route\n(loops, median ppm)", fontsize=9, loc="left", color=INK)
            if loops:
                ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
                ax.legend(fontsize=7, frameon=True, facecolor="white", edgecolor="none", framealpha=1, labelcolor=INK,
                          loc="upper left")
        ax.set_xlabel("misclosure (ppm)", fontsize=8, color=MUTED)
        if k == 0:
            ax.set_ylabel("loops", fontsize=8, color=MUTED)
        style(ax)
    ax = fig.add_subplot(gs[1, :2])
    bins = np.arange(-8, 8.25, 0.5)
    x = np.linspace(-8, 8, 400)
    for lp in have:
        name, col = LOOPS[lp]
        z = C[C.loop == lp].z.clip(-8, 8)
        ax.hist(z, bins=bins, density=True, histtype="step", lw=2, color=col, label=name)
    ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color=MUTED, lw=1, ls="--", label="if errors were as stated")
    ax.set_title("D. Misclosure ÷ its standard error", fontsize=9, loc="left", color=INK)
    ax.set_xlabel("z", fontsize=8, color=MUTED)
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper left")
    style(ax)
    ax = fig.add_subplot(gs[1, 2])
    M = {m: v["variance_factor"] for m, v in S["methods"].items() if v["variance_factor"] and m != "arithmetic"}
    order = sorted(M, key=M.get)
    ax.barh([NAMES.get(m, m) for m in order], [M[m] for m in order], color=BLUE, height=0.6)
    ax.axvline(1, color=INK, lw=0.8, ls="--")
    for i, m in enumerate(order):
        ax.text(M[m] + 0.1, i, f"{M[m]:.1f}", va="center", fontsize=8, color=INK)
    ax.set_title("E. Variance factor per method\n(1 = standard errors right)", fontsize=9, loc="left", color=INK)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    fig.savefig(a.out, dpi=150, facecolor="white", bbox_inches="tight")


if __name__ == "__main__":
    main()
