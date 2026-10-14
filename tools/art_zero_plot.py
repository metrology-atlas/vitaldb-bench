# SPDX-License-Identifier: MIT
"""Figure for art_zero_jumps: pressure changes across line accesses against controls; open-to-air readings."""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rs = [json.loads(l) for l in open(sys.argv[1])]
G = {"zero (open to air)": [], "flush/draw, no air": [], "occlusion (cuff)": [], "random control": []}
air, pairs = [], []
for r in rs:
    for q in r.get("accesses", []):
        key = "zero (open to air)" if q.get("air") else ("flush/draw, no air" if "high" in q["kinds"] else None)
        if key and q["delta"]:
            G[key].append(q["delta"])
        for e in q.get("air", []):
            air.append(e["first"])
            if q["delta"] and e["n"] >= 3:
                pairs.append((e["first"], q["delta"]["map"]))
    G["occlusion (cuff)"] += [q["delta"] for q in r.get("occlusions", []) if q["delta"]]
    G["random control"] += r.get("controls", [])
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
cols = {"zero (open to air)": "C3", "flush/draw, no air": "C1", "occlusion (cuff)": "C2", "random control": "0.4"}
for k, v in G.items():
    d = np.sort(np.abs([x["dbp"] for x in v]))
    ax[0].plot(d, 1 - np.arange(d.size) / d.size, color=cols[k], label=f"{k} (n={d.size})")
ax[0].set_xlim(0, 30); ax[0].set_ylim(0, 1); ax[0].set_xlabel("|diastolic change| after minus before (mmHg)")
ax[0].set_ylabel("share of events exceeding"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_title("Diastolic pressure change across the event (60 s windows)", fontsize=10)
ax[1].hist(air, bins=np.arange(-15, 15.5, 1), color="C3")
ax[1].set_xlabel("pressure read while open to air (mmHg; ideal 0)"); ax[1].set_ylabel("open-to-air plateaus")
ax[1].set_title(f"Open-to-air readings ({len(air)} plateaus)", fontsize=10)
P = np.array(pairs)
ax[2].scatter(P[:, 0], P[:, 1], s=8, alpha=.5, color="C3")
ax[2].plot([-15, 15], [15, -15], "k--", lw=.8, label="expected if zeroing removed the offset")
ax[2].set_xlim(-15, 15); ax[2].set_ylim(-40, 40); ax[2].legend(fontsize=8)
ax[2].set_xlabel("open-to-air reading (mmHg)"); ax[2].set_ylabel("MAP change after minus before (mmHg)")
ax[2].set_title(f"Air reading vs MAP change (r = {np.corrcoef(P[:, 0], P[:, 1])[0, 1]:.2f})", fontsize=10)
fig.suptitle("Arterial pressure across line accesses (206 VitalDB cases)", fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=80)
