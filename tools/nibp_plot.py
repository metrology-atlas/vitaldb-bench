# SPDX-License-Identifier: MIT
"""Figure: intervals between NIBP readings (with and without an arterial line) and ART vs NIBP agreement."""
import csv
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

allr = [json.loads(l) for l in open(sys.argv[1])]
art_r = [json.loads(l) for l in open(sys.argv[2])]
art = {int(r["case"]) for r in csv.DictReader(open(sys.argv[3])) if r["label"] == "art_line:access"}


def intervals(rs):
    out = []
    for r in rs:
        if r.get("nibp", 0) < 2:
            continue
        t = np.array(r["t"])
        t = t[np.r_[np.diff(t) > 10, True]]          # one measurement updating in two steps
        out += list(np.diff(t) / 60)
    return np.array(out)


Ia = intervals([r for r in allr if r["case"] in art])
In = intervals([r for r in allr if r["case"] not in art])
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
bins = np.logspace(np.log10(0.5), np.log10(240), 60)
ax[0].hist(In, bins=bins, color="C0", alpha=.7, density=True, label=f"no arterial line ({len(In)} intervals)")
ax[0].hist(Ia, bins=bins, color="C3", alpha=.6, density=True, label=f"arterial line ({len(Ia)} intervals)")
ax[0].set_xscale("log"); ax[0].set_xticks([1, 2.5, 5, 10, 30, 60, 120]); ax[0].set_xticklabels(["1", "2.5", "5", "10", "30", "60", "120"])
ax[0].set_xlabel("minutes between NIBP readings (log scale)"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
ax[0].set_title(f"NIBP intervals ({len(allr)} cases, every 5th; medians {np.median(In):.1f} / {np.median(Ia):.1f} min)", fontsize=10)
P = []
for r in art_r:
    if not r.get("art"):
        continue
    for a, v in zip(r["art"], r["v"]):
        if a is not None:
            P.append((a[0], v[2], a[1] - a[2], v[0] - v[1]))
P = np.array(P)
ax[1].scatter(P[:, 1], P[:, 0], s=6, alpha=.5)
ax[1].plot([30, 150], [30, 150], "k--", lw=.8)
ax[1].set_xlim(30, 150); ax[1].set_ylim(30, 150)
ax[1].set_xlabel("NIBP MAP (mmHg)"); ax[1].set_ylabel("ART MAP over the 35 s before the NIBP result (mmHg)")
ax[1].set_title(f"MAP agreement ({len(P)} readings, 206 ART cases)", fontsize=10)
ppr = P[:, 2] / np.maximum(P[:, 3], 5)
ax[2].hist(ppr, bins=np.arange(0, 3.01, 0.05), color="C2")
ax[2].axvline(1, color="k", lw=.8)
ax[2].set_xlabel("pulse pressure ratio, ART / NIBP"); ax[2].set_ylabel("readings")
ax[2].set_title("Pulse pressure ratio (median %.2f; < 0.7: %.0f%%, > 1.5: %.0f%%)" % (np.median(ppr), np.mean(ppr < .7) * 100, np.mean(ppr > 1.5) * 100), fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[4], dpi=80)
