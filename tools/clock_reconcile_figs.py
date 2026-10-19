# SPDX-License-Identifier: MIT
"""Figure of the clock-network reconciliation (tools/clock_reconcile.py output).

    python tools/clock_reconcile_figs.py RUN_DIR --out FIG.png

A: misclosure (ppm) of each BIS-SNUADC-PC loop.  B: the same for the two Primus-PC routes.
C: misclosure divided by its standard error, both loop types, against a standard normal (what the loops would
show if every edge's standard error were right).  D: each method's pooled variance factor (1 = errors as stated).
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

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
LOOPS = {"bis_steps + bursts + resets": ("BIS – SNUADC – PC", BLUE), "breath + pip_numeric": ("Primus – PC, two routes", ORANGE)}
NAMES = {"resets": "SNUADC resets", "bis_steps": "BIS correction steps", "bursts": "ECG–EEG bursts",
         "breath": "Breath matching", "pip_numeric": "PIP numerics", "arithmetic": "Stamp arithmetic"}


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
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.3), gridspec_kw={"width_ratios": [1, 1, 1.1, 1]})
    for k, (loop, lim, step) in enumerate([("bis_steps + bursts + resets", 50, 2), ("breath + pip_numeric", 400, 20)]):
        g = C[C.loop == loop].misclosure
        name, col = LOOPS[loop]
        ax[k].hist(g.clip(-lim, lim), bins=np.arange(-lim, lim + step, step), color=col, edgecolor="white", lw=0.6)
        ax[k].axvline(0, color=INK, lw=0.8)
        ax[k].set_title(f"{'AB'[k]}. {name}\n{len(g):,} loops, median {g.median():+.1f} ppm", fontsize=9, loc="left", color=INK)
        ax[k].set_xlabel("misclosure (ppm)", fontsize=8, color=MUTED)
        style(ax[k])
    ax[0].set_ylabel("loops", fontsize=8, color=MUTED)
    bins = np.arange(-8, 8.25, 0.5)
    x = np.linspace(-8, 8, 400)
    for loop, (name, col) in LOOPS.items():
        z = C[C.loop == loop].z.clip(-8, 8)
        ax[2].hist(z, bins=bins, density=True, histtype="step", lw=2, color=col, label=name)
    ax[2].plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color=MUTED, lw=1, ls="--", label="if errors were as stated")
    ax[2].set_title("C. Misclosure ÷ its standard error", fontsize=9, loc="left", color=INK)
    ax[2].set_xlabel("z", fontsize=8, color=MUTED)
    ax[2].set_ylim(0, 0.62)
    ax[2].legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper left")
    style(ax[2])
    M = {m: v["variance_factor"] for m, v in S["methods"].items() if v["variance_factor"] and m != "arithmetic"}
    order = sorted(M, key=M.get)
    ax[3].barh([NAMES.get(m, m) for m in order], [M[m] for m in order], color=BLUE, height=0.6)
    ax[3].axvline(1, color=INK, lw=0.8, ls="--")
    for i, m in enumerate(order):
        ax[3].text(M[m] + 0.1, i, f"{M[m]:.1f}", va="center", fontsize=8, color=INK)
    ax[3].set_title("D. Variance factor per method\n(1 = standard errors right)", fontsize=9, loc="left", color=INK)
    style(ax[3])
    ax[3].grid(axis="x", color=GRID, lw=0.6)
    ax[3].grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
