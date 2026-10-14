# SPDX-License-Identifier: MIT
"""Figure for art_zero_jumps with 20 s windows: diastolic and pulse pressure (SBP - DBP) change across accesses,
and pulse duration (width of the pulse at 50% height) against pulse pressure change."""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rs = [json.loads(l) for l in open(sys.argv[1])]
G = {"zero (open to air)": [], "flush, no air": [], "occlusion (cuff)": [], "random control": []}
for r in rs:
    for q in r.get("accesses", []):
        if not q["delta"]:
            continue
        if q.get("air"):
            G["zero (open to air)"].append(q["delta"])
        elif "high" in q["kinds"]:
            G["flush, no air"].append(q["delta"])
    G["occlusion (cuff)"] += [q["delta"] for q in r.get("occlusions", []) if q["delta"]]
    G["random control"] += r.get("controls", [])
cols = {"zero (open to air)": "C3", "flush, no air": "C1", "occlusion (cuff)": "C2", "random control": "0.4"}
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for v in G.values():
    for x in v:
        x["pp"] = x["sbp"] - x["dbp"]
for i, (m, xl, xmax) in enumerate((("dbp", "|diastolic change| (mmHg)", 30), ("pp", "|pulse pressure change| (SBP − DBP, mmHg)", 30))):
    for k, v in G.items():
        d = np.sort(np.abs([x[m] for x in v if x.get(m) is not None]))
        ax[i].plot(d, 1 - np.arange(d.size) / d.size, color=cols[k], label=f"{k} (n={d.size})")
    ax[i].set_xlim(0, xmax); ax[i].set_ylim(0, 1); ax[i].set_xlabel(xl); ax[i].set_ylabel("share of events exceeding")
    ax[i].grid(alpha=.3); ax[i].legend(fontsize=8)
ax[0].set_title("Diastolic pressure, 20 s before vs 20 s after", fontsize=10)
ax[1].set_title("Pulse pressure (SBP − DBP), 20 s before vs after", fontsize=10)
F = [x for x in G["flush, no air"] + G["zero (open to air)"] if x.get("pw50_ms") is not None]
dpw = np.array([x["pw50_ms"] for x in F]); dpp = np.array([x["sbp"] - x["dbp"] for x in F])
ax[2].scatter(dpp, dpw, s=6, alpha=.4, color="C1")
ax[2].set_xlim(-40, 40); ax[2].set_ylim(-120, 120); ax[2].axhline(0, color="k", lw=.5); ax[2].axvline(0, color="k", lw=.5)
ax[2].set_xlabel("pulse pressure change (mmHg)"); ax[2].set_ylabel("pulse duration change, width at 50% height (ms)")
ax[2].set_title(f"Flush accesses: longer pulse with smaller pulse pressure (r = {np.corrcoef(dpp, dpw)[0, 1]:.2f})", fontsize=10)
fig.suptitle("Arterial pressure across line accesses, 20 s windows (206 VitalDB cases)", fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=80)
