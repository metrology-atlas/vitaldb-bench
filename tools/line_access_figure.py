# SPDX-License-Identifier: MIT
"""Figure for the arterial-line-accesses study. This machine only; the figure shows one case by its id.

    python tools/line_access_figure.py --root R --art-labels art_line_labels.csv --draws lab_draw_labels.csv \
        --lab lab_data.csv --case 1 --access 0 --out arterial-line-accesses.png

Left: the arterial pressure around one detected access (the --access-th access of --case), shaded.
Right: time from the start of the latest access to each blood-gas stamp, against
the same measure at random times in the same recordings (the control); only times with an access in the 15 min
before them count, and the share that has one is in the legend.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WIN = 900.0


def latest_access_lag(starts: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Seconds from the latest access start at or before t (within WIN), else NaN."""
    i = np.searchsorted(starts, t, side="right") - 1
    lag = np.where(i >= 0, t - starts[np.clip(i, 0, None)], np.nan)
    lag[lag > WIN] = np.nan
    return lag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--art-labels", required=True)
    ap.add_argument("--draws", required=True)
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--access", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import vitaldb

    lab_lbl = pd.read_csv(a.art_labels)
    acc = lab_lbl[lab_lbl.label == "art_line:access"]
    draws = pd.read_csv(a.draws)
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    ev = acc[acc.case == a.case].iloc[a.access]
    vf = vitaldb.VitalFile(str(a.root / "vital_files" / f"{a.case:04d}.vital"), ["SNUADC/ART"])
    t0 = ev.start_s - 60
    x = vf.to_numpy(["SNUADC/ART"], 1 / 100)[:, 0]   # 100 Hz is enough to show the shape
    ts = np.arange(len(x)) / 100 + vf.dtstart
    m = (ts >= t0) & (ts <= ev.end_s + 120)
    ax[0].plot(ts[m] - ev.start_s, x[m], lw=.5, color="#08519c")
    ax[0].axvspan(0, ev.end_s - ev.start_s, color="#fdae6b", alpha=.45, label=f"detected access ({ev.kinds})")
    ax[0].set(xlabel="seconds from the start of the access", ylabel="arterial pressure (mmHg)",
              title=f"case {a.case}: one arterial line access", ylim=(-20, max(360, np.nanmax(x[m]) + 10)))
    ax[0].title.set_fontsize(9)
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")

    # control: random times within each case's scanned span, as many as that case has lab groups
    starts = {c: np.sort(g.start_s.to_numpy()) for c, g in acc.groupby("case")}
    span = lab_lbl.groupby("case").agg(lo=("start_s", "min"), hi=("end_s", "max"))
    groups = draws.drop_duplicates("draw_id")
    real, ctrl = {}, []
    for panel in ("blood_gas", "central"):
        g = groups[groups.panel == panel]
        real[panel] = g.lag_s.to_numpy()
    n_all = {p: int((groups.panel == p).sum()) for p in real}
    for c, g in groups.groupby("case"):
        if c not in starts or c not in span.index:
            continue
        lo, hi = span.loc[c, ["lo", "hi"]]
        t = rng.uniform(lo + WIN, hi, 20 * len(g))
        ctrl.append(latest_access_lag(starts[c], t))
    ctrl = np.concatenate(ctrl)
    bins = np.arange(0, WIN + 1, 30)
    ax[1].hist(ctrl[np.isfinite(ctrl)], bins=bins, density=True, color="#d9d9d9",
               label=f"random times ({100 * np.isfinite(ctrl).mean():.0f}% have an access)")
    for panel, col, name in (("blood_gas", "#08519c", "blood gas"),):
        ax[1].hist(real[panel], bins=bins, density=True, histtype="step", lw=2, color=col,
                   label=f"{name} stamps ({len(real[panel]):,} matched)")
    ax[1].set(xlabel="lab stamp minus the start of the latest access (s)", ylabel="share (per s)", xlim=(0, WIN),
              title="Blood-gas stamps follow accesses 3–12 min later; random times do not")
    ax[1].title.set_fontsize(9)
    ax[1].legend(frameon=False, fontsize=8)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print({p: {"matched": len(v), "lag_median_s": float(np.median(v))} for p, v in real.items()},
          "control share with access %.3f" % np.isfinite(ctrl).mean())


if __name__ == "__main__":
    main()
