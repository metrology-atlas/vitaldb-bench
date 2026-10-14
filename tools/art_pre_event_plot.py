# SPDX-License-Identifier: MIT
"""Event-locked examples: ART per minute (SBP, MAP, DBP) and cuff readings from 60 min before to 30 min after
single flushes/zeroings. Args: artmin.jsonl nibp.jsonl events.jsonl out.png case:t[:title] ..."""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = {json.loads(l)["case"]: json.loads(l) for l in open(sys.argv[1])}
N = {json.loads(l)["case"]: json.loads(l) for l in open(sys.argv[2])}
E = [json.loads(l) for l in open(sys.argv[3])]
picks = [p.split(":", 2) for p in sys.argv[5:]]
fig, axs = plt.subplots(int(np.ceil(len(picks) / 2)), 2, figsize=(16, 3.4 * np.ceil(len(picks) / 2)), squeeze=False)
for ax, p in zip(axs.flat, picks):
    c, t_ev = int(p[0]), float(p[1])
    ev = min((e for e in E if e["case"] == c and e["kind"] != "random"), key=lambda e: abs(e["t"] - t_ev))
    s = np.array(M[c]["minutes"], float)
    m = (s[:, 0] >= t_ev - 3600) & (s[:, 0] <= t_ev + 1800)
    tt = (s[m, 0] + 30 - t_ev) / 60
    ax.plot(tt, s[m, 2], ".", ms=3, color="C3", label="ART SBP")
    ax.plot(tt, s[m, 1], ".", ms=3, color="k", label="ART MAP")
    ax.plot(tt, s[m, 3], ".", ms=3, color="C0", label="ART DBP")
    r = N.get(c, {})
    if r.get("t"):
        T = np.array(r["t"]) + r["first"]
        V = np.array(r["v"], float)
        k = (T >= t_ev - 3600) & (T <= t_ev + 1800)
        k &= np.r_[np.diff(T) > 10, True]
        for j, col in ((0, "C3"), (2, "k"), (1, "C0")):
            ax.plot((T[k] - t_ev) / 60, V[k, j], "s", mfc="none", color=col, ms=6)
    for x in M[c]["accesses"]:
        if t_ev - 3600 <= x[0] <= t_ev + 1800:
            ax.axvspan((x[0] - t_ev) / 60, (x[2] - t_ev) / 60, color="C1" if "low" not in x[3] else "C4", alpha=.25)
    ax.axvline(0, color="C4" if ev["kind"] == "zero" else "C1", lw=1)
    ax.set_xlim(-60, 30); ax.set_ylabel("mmHg"); ax.set_xlabel("minutes from the event")
    txt = f"case {c}: {ev['kind']}; pre PP slope {ev['pp_slope_pct_h']:+.0f}%/h"
    if "d_dbp_20s" in ev:
        txt += f"; 20 s jump DBP {ev['d_dbp_20s']:+.1f}, PP {ev['d_pp_20s']:+.1f}"
    if "cuff_dmap_last" in ev and "cuff_dmap_after" in ev:
        txt += f"; cuff |dMAP| {abs(ev['cuff_dmap_last']):.0f}→{abs(ev['cuff_dmap_after']):.0f}"
    ax.set_title(txt, fontsize=9)
axs.flat[0].legend(fontsize=7, loc="upper left")
fig.suptitle("Single events: ART per minute (dots), cuff readings (squares); purple = zero access, orange = flush", fontsize=11)
fig.tight_layout(); fig.savefig(sys.argv[4], dpi=72)
