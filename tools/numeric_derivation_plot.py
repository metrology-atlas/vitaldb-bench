# SPDX-License-Identifier: MIT
"""Plot the numeric-derivation pilot summary (tools/numeric_derivation_agg.py output; aggregates only).

    python tools/numeric_derivation_plot.py summary.json out.png
"""

from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"Solar8000": "#1f5fa8", "Primus": "#b5541c"}
DEV = {"HR": "Solar8000", "ART_SBP": "Solar8000", "ART_MBP": "Solar8000", "ART_DBP": "Solar8000", "CVP": "Solar8000",
       "ETCO2": "Primus", "RR_CO2": "Primus", "PIP": "Primus", "PEEP": "Primus"}
LABEL = {"median|t": "median over {w} s", "median|n": "median of {w} beats/breaths", "60/mean_int|n": "60/mean of {w} intervals",
         "60/mean_int|t": "60/mean interval over {w} s", "mean_rate|n": "mean rate of {w}", "ewma|n": "EWMA alpha 1/{w}",
         "ewma_rate|n": "EWMA rate 1/{w}", "wave_mean|t": "waveform mean over {w} s", "mean|n": "mean of {w}",
         "mean|t": "mean over {w} s"}


def main():
    S = json.load(open(sys.argv[1]))
    T = S["tracks"]
    names = [n for n in ["HR", "ART_SBP", "ART_MBP", "ART_DBP", "CVP", "RR_CO2", "ETCO2", "PIP", "PEEP"] if n in T]
    fig = plt.figure(figsize=(17, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.45, 1.0], wspace=0.3)
    # A: delay
    ax = fig.add_subplot(gs[0])
    for i, n in enumerate(names[::-1]):
        t = T[n]
        c = COL[DEV[n]]
        lo, me, hi = t["delay_mid_s_iqr"]
        ax.plot([lo, hi], [i, i], color=c, lw=7, alpha=0.35, solid_capstyle="butt")
        ax.plot(me, i, "o", color=c, ms=8)
        a, m, b = t["delay_min_s_iqr"]
        ax.plot(m, i, "|", color="k", ms=14, mew=2)
        ax.text(15.6, i, f"{me:.1f}", va="center", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}  (n={T[n]['n_cases']})" for n in names[::-1]])
    ax.axvline(0, color="0.6", lw=1)
    ax.set_xlim(-3, 15)
    ax.set_xlabel("delay from data to recorded value (s)")
    ax.set_title("A  Delay: median and IQR across cases\n(dot = middle of the poll interval; | = earliest)", loc="left", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    # B: winning algorithm
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    ax.set_title("B  Best-fitting algorithm (cases where it won)", loc="left", fontsize=11)
    y = 0.95
    for n in names:
        t = T[n]
        parts = []
        for fam, k in t["best_family"][:3]:
            w = t["window_by_family"][fam]
            lab = LABEL.get(fam, fam + " {w}").format(w=int(np.median(w)))
            parts.append(f"{lab} ({k})")
        extra = ""
        if "round_minus_floor" in t:
            r = t["round_minus_floor"]
            extra = "; truncated (floor) in %d/%d" % (r["floor_better_cases"], r["of"]) if r["floor_better_cases"] > r["of"] / 2 \
                else "; rounded in %d/%d" % (r["of"] - r["floor_better_cases"], r["of"])
        ax.text(0.0, y, n, fontsize=10.5, fontweight="bold", color=COL[DEV[n]], transform=ax.transAxes, va="top")
        ax.text(0.2, y, parts[0] + (extra if len(parts) < 2 else ""), fontsize=9.2, transform=ax.transAxes, va="top")
        if len(parts) > 1:
            ax.text(0.2, y - 0.04, "or " + "; ".join(parts[1:2]) + extra, fontsize=8.6, color="0.35",
                    transform=ax.transAxes, va="top")
        y -= 0.105
    # C: repeats
    ax = fig.add_subplot(gs[2])
    xs = ["<0.5", "0.5-1", "1-2", "2-4", ">4"]
    for n, ls, mk in (("HR", "-", "o"), ("ART_MBP", "-", "s"), ("CVP", "--", "^"), ("ETCO2", ":", "o"), ("PIP", "-.", "s")):
        if n not in T:
            continue
        r = T[n]["repeat_fraction_by_pred_change_q"]
        v = [r[k][0] for k in sorted(r, key=float)]
        ax.plot(range(5), v, ls, marker=mk, color=COL[DEV[n]], label=n, lw=1.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels(xs)
    ax.set_xlabel("predicted change since previous record (reporting steps)")
    ax.set_ylabel("fraction of records repeating the previous value")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("C  Repeats vs predicted change\n(a held value would repeat even when change is large)", loc="left", fontsize=11)
    fig.suptitle(f"How monitor numerics are derived from waveforms: pilot, {S['cases']} cases "
                 "(Solar8000 blue, Primus orange)", x=0.01, ha="left", fontsize=13)
    fig.savefig(sys.argv[2], dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
