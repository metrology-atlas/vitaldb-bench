# SPDX-License-Identifier: MIT
"""Figure for the setup-identification study, from setup_assign.py's outputs (aggregates only).

    python tools/setup_assign_figure.py --dir DIR --out setup-identification.png

Left: calibration of the posterior in the hidden-port validation. Complete recordings had one core port hidden
in turn; for each posterior bin, the share of recordings whose true key came out on top, with the bin's size (bins of fewer than 10 left out).
Right: the recordings without a complete PC key, by the device that is missing, coloured by how well a key
was assigned (no candidate, posterior below 0.5, 0.5-0.9, at least 0.9 or a single candidate).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="setup_assign.py --out directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    summ = json.loads((d / "summary.json").read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1, 1.25]})

    colours = {"SNUADC": "#1f77b4", "BIS": "#d62728", "Solar8000": "#2ca02c", "Primus": "#9467bd"}
    for dev, v in summ["validation"].items():
        xs, ys, ns = [], [], []
        for b, (hit, n) in sorted(v["calibration"].items()):
            lo, hi = map(float, b.split("-"))
            if n >= 10:   # smaller bins are noise
                xs.append((lo + hi) / 2); ys.append(hit / n); ns.append(n)
        ax1.plot(xs, ys, "-", color=colours.get(dev, "k"), lw=1, alpha=.6)
        hollow = dev in ("Solar8000", "Primus")   # these two overlap near 1; draw them open so both show
        ax1.scatter(xs, ys, s=[8 + 60 * (n / max(ns)) ** .5 for n in ns], marker="s" if dev == "Primus" else "o",
                    facecolors="none" if hollow else colours.get(dev, "k"), edgecolors=colours.get(dev, "k"),
                    label=f"{dev} hidden (top-1 {100 * v['top1']:.1f}%)", zorder=3)
    ax1.plot([0, 1], [0, 1], "--", color="grey", lw=.8)
    ax1.set(xlim=(0.15, 1.02), ylim=(0, 1.02), xlabel="posterior of the top key", ylabel="share where it is the true key",
            title="Validation: one port hidden in turn")
    ax1.legend(fontsize=8, loc="lower right", frameon=False)

    rows = list(csv.DictReader(open(d / "assigned.csv")))
    def cls(r):
        if not r["key"]:
            return "no candidate"
        if int(r["candidates"] or 0) == 1:
            return "single candidate"
        p = float(r["posterior"])
        return "posterior ≥ 0.9" if p >= .9 else "posterior 0.5–0.9" if p >= .5 else "posterior < 0.5"
    order = ["single candidate", "posterior ≥ 0.9", "posterior 0.5–0.9", "posterior < 0.5", "no candidate"]
    shade = ["#08519c", "#4292c6", "#fdae6b", "#e6550d", "#bdbdbd"]
    by = defaultdict(Counter)
    for r in rows:
        by[r["missing"]][cls(r)] += 1
    groups = sorted(by, key=lambda g: -sum(by[g].values()))
    left = [0] * len(groups)
    for c, col in zip(order, shade):
        vals = [by[g][c] for g in groups]
        ax2.barh(groups, vals, left=left, color=col, label=c)
        left = [l + v for l, v in zip(left, vals)]
    for i, g in enumerate(groups):
        ax2.text(left[i] + 4, i, f"{left[i]}", va="center", fontsize=8)
    ax2.invert_yaxis()
    ax2.set(xlabel="recordings", title=f"{len(rows)} recordings without a complete PC key, by missing device")
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    tot = Counter(cls(r) for r in rows)
    print({k: tot[k] for k in order})


if __name__ == "__main__":
    main()
