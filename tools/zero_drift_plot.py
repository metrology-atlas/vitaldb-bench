# SPDX-License-Identifier: MIT
"""Per-case line-minus-cuff MAP (NIBP-level-bias corrected) over time with zero and flush accesses marked.
Args: artmin.jsonl nibp_art.jsonl out.png case1,case2,..."""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = {json.loads(l)["case"]: json.loads(l) for l in open(sys.argv[1])}
N = {json.loads(l)["case"]: json.loads(l) for l in open(sys.argv[2])}
cases = [int(x) for x in sys.argv[4].split(",")]
rows = {}
allp = []
for c, r in N.items():
    if not r.get("art"):
        continue
    T = np.array(r["t"]) + r["first"]
    V = np.array(r["v"], float)
    keep = []
    for i, x in enumerate(r["art"]):
        if x is None:
            continue
        if keep and T[i] - T[keep[-1]] <= 10:
            keep[-1] = i
        else:
            keep.append(i)
    rows[c] = np.array([(T[i], r["art"][i][0] - V[i, 2], V[i, 2]) for i in keep]) if keep else np.empty((0, 3))
    allp += list(rows[c])
allp = np.array(allp)
b = np.polyfit(allp[:, 2], allp[:, 1], 1)
fig, axs = plt.subplots(int(np.ceil(len(cases) / 2)), 2, figsize=(16, 2.9 * np.ceil(len(cases) / 2)), squeeze=False)
for ax, c in zip(axs.flat, cases):
    d = rows[c]
    t0 = M[c]["minutes"][0][0]
    adj = d[:, 1] - np.polyval(b, d[:, 2])
    ax.plot((d[:, 0] - t0) / 3600, adj, "o-", ms=4, lw=.6, color="k")
    nz = nf = 0
    for x in M[c]["accesses"]:
        z = "low" in x[3]
        nz += z
        nf += (not z) and "high" in x[3]
        ax.axvline((x[0] - t0) / 3600, color="C4" if z else "C1", lw=1.2 if z else .6, alpha=.8)
    ax.axhline(0, color="0.6", lw=.6)
    ax.set_title(f"case {c}: {len(d)} cuff readings with clean ART; {nz} zero, {nf} flush accesses", fontsize=9)
    ax.set_ylabel("line − cuff MAP (mmHg)"); ax.set_xlabel("hours from ART start")
fig.suptitle("Line minus cuff MAP (bias-corrected) between zeroings (purple) and flushes (orange)", fontsize=11)
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=72)
