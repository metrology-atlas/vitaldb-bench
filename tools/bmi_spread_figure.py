# SPDX-License-Identifier: MIT
"""Figure for the BMI study: how wide the true BMI is, record by record, with vitaldb-resolution's derived model.

    python tools/bmi_spread_figure.py --vr <vitaldb-resolution checkout> --release <VitalDB release> --out bmi-spread.png

BMI is derived from the posteriors of height and weight, so its spread carries their rounding and measurement
variability. Left: the 95% interval width of the true BMI over every clinical row. Right: that width against the
recorded BMI (each dot one record). Records are coloured by how height and weight were written: a height on
a 5 cm mark (read as written in, not measured), whole cm or kg, or both with decimals. Only the figure and printed
aggregates leave this machine.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vr", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.vr)
    import vitaldb_resolution as vr

    reg = vr.load()
    clinical = pd.read_csv(Path(a.release) / "clinical_data.csv")
    width = []
    for i in range(0, len(clinical), 400):
        for d in reg.distributions("bmi", clinical.iloc[i:i + 400]):
            try:
                lo, hi = d.interval(0.95) if getattr(d, "status", None) == "ok" else (np.nan, np.nan)
            except Exception:  # noqa: BLE001
                lo, hi = np.nan, np.nan
            width.append(hi - lo)
    c = clinical.assign(w=width)
    c = c[np.isfinite(c.w) & (c.w > 0) & np.isfinite(c.bmi)]
    # how the inputs were written: a height on a 5 cm mark, whole units, or both with decimals
    h5 = c.height % 5 == 0
    whole = (c.height % 1 == 0) | (c.weight % 1 == 0)
    groups = [("height and weight with decimals", ~whole, "#4292c6"),
              ("height or weight in whole cm / kg", whole & ~h5, "#fd8d3c"),
              ("height on a 5 cm mark", h5, "#d62728")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.logspace(np.log10(c.w.min()), np.log10(c.w.max()), 60)
    ax1.hist([c.w[m] for _, m, _ in groups], bins=bins, stacked=True, color=[g[2] for g in groups],
             label=[f"{g[0]} ({int(g[1].sum()):,})" for g in groups])
    ax1.set_xscale("log")
    for q, ls in [(50, "-"), (90, "--")]:
        v = np.percentile(c.w, q)
        ax1.axvline(v, color="k", ls=ls, lw=.8)
        ax1.text(v, ax1.get_ylim()[1] * .97, f" p{q} {v:.2g}", fontsize=8, va="top")
    ax1.set(xlabel="95% interval width of the true BMI (kg/m²)", ylabel="records", title=f"{len(c):,} records")
    ax1.legend(fontsize=8, frameon=False, loc="center right")
    for lab, m, col in groups:
        ax2.scatter(c.bmi[m], c.w[m], s=3, alpha=.3, color=col, lw=0)
    ax2.set_yscale("log")
    ax2.set(xlabel="recorded BMI (kg/m²)", ylabel="95% width (kg/m²)", title="Width against recorded BMI")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    for lab, m, _ in groups:
        x = c.w[m]
        print(f"{lab}: n={len(x)} width95 median {x.median():.3g} p10 {x.quantile(.1):.3g} p90 {x.quantile(.9):.3g}")
    print(f"all: median {c.w.median():.3g} p90 {c.w.quantile(.9):.3g}")

if __name__ == "__main__":
    main()
