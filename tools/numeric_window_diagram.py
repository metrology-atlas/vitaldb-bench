# SPDX-License-Identifier: MIT
"""Schematic of the timing model used in tools/numeric_derivation_pilot.py (no data; illustrative numbers).

    python tools/numeric_window_diagram.py out.png [--T 6 --d 2.6 --poll 2 --rr 0.8]

One recorded value at stamp t: computed at u = t - d - phi (phi uniform over one poll), from the beats whose
R-R intervals completed in the T s before u. Shows the window's start and end and the delay measured from
the window's end, the anchor of every timing quantity (the availability rule: the value cannot exist before
its window ends). The window's middle is shown only as a derived, descriptive offset, never a time stamp.
For values of the last completed breath (Primus ETCO2, PIP) the computation band is one 0.48 s record, not
the poll interval.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bracket(ax, x0, x1, y, text, col, h=0.12, fs=10):
    ax.plot([x0, x0, x1, x1], [y + h, y, y, y + h], color=col, lw=1.6)
    ax.text((x0 + x1) / 2, y - 0.08, text, ha="center", va="top", fontsize=fs, color=col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--T", type=float, default=6.0)
    ap.add_argument("--d", type=float, default=2.6)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--rr", type=float, default=0.8)
    a = ap.parse_args()
    T, d, P, RR = a.T, a.d, a.poll, a.rr
    phi = P / 2                        # the example position within the poll
    t = 0.0
    u = t - d - phi
    close, open_ = u, u - T          # close = the window's end
    centre = u - T / 2               # descriptive only

    fig, ax = plt.subplots(figsize=(15, 7.2))
    x0 = open_ - 2.2
    # ECG-like schematic with R peaks
    r = np.arange(x0 + 0.3, t + 1.2, RR)
    tt = np.linspace(x0, t + 1.2, 6000)
    ecg = np.zeros_like(tt)
    for rp in r:
        ecg += 1.0 * np.exp(-((tt - rp) / 0.018) ** 2) - 0.15 * np.exp(-((tt - rp - 0.04) / 0.02) ** 2)
        ecg += 0.18 * np.exp(-((tt - rp - 0.28) / 0.06) ** 2)
    yE = 4.2
    ax.plot(tt, yE + 0.55 * ecg, color="0.25", lw=1)
    ax.text(x0, yE + 0.75, "ECG (SNUADC, recorder time)", fontsize=10, color="0.25")

    # R-R intervals placed at their second R (when they can first be measured); included if that R is in the window
    yI = 3.45
    comp = r[1:]
    inc = (comp > open_) & (comp <= close)
    for k, (a0, a1) in enumerate(zip(r[:-1], r[1:])):
        if a1 < x0 + 0.4 or a1 > t + 1.1:
            continue
        col = "#1f5fa8" if inc[k] else "0.75"
        ax.annotate("", xy=(a1, yI), xytext=(a0, yI), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.plot(a1, yI, "o", color=col, ms=4)
    first = np.flatnonzero(inc)[0]
    ax.text(x0, yI + 0.22, "R-R intervals, each placed at its closing R (blue = in the window)", fontsize=10, color="#1f5fa8")
    ax.annotate("first included interval starts\n~one R-R before the window opens", xy=(r[first], yI - 0.02),
                xytext=(r[first] - 1.9, yI - 0.75), fontsize=9, arrowprops=dict(arrowstyle="->", color="0.4"))

    # window
    yW = 2.55
    ax.add_patch(plt.Rectangle((open_, yW - 0.28), T, 0.56, color="#1f5fa8", alpha=0.15, ec="#1f5fa8"))
    ax.text(centre, yW, f"window: median of R-R intervals completed in the last T = {T:g} s",
            ha="center", va="center", fontsize=10.5, color="#1f5fa8")
    for x, lab in ((open_, "window start\n(oldest data)"), (centre, "middle\n(derived only)"), (close, "window end\n(newest data)")):
        ax.plot([x, x], [0.2, yE + 0.7], color="#1f5fa8", lw=1, ls=":")
        ax.text(x, yW + 0.42, lab, ha="center", fontsize=9, color="#1f5fa8")

    # computation and recording
    yC = 1.75
    ax.add_patch(plt.Rectangle((t - d - P, yC - 0.16), P, 0.32, color="#e08a1e", alpha=0.25))
    ax.text(t - d - P / 2 - 0.9, yC + 0.22, f"u = window end = computation: anywhere in this {P:g} s band (uniform phi; polled numerics);\nthe whole window slides with it. Last-breath values: one 0.48 s record", ha="center", fontsize=9.5,
            color="#b35f00")
    ax.plot(u, yC, "v", color="#b35f00", ms=10)
    ax.text(u, yC - 0.3, "u (example: middle)", ha="center", fontsize=9, color="#b35f00")
    ax.annotate("", xy=(u, yC + 0.0), xytext=(close, yC + 0.0), arrowprops=dict(arrowstyle="<->", color="k"))
    # close == u in this layout: the window ends at the computation instant; d sits between u and the stamp
    for x in (t - P, t):
        ax.plot([x, x], [0.2, yE + 0.7], color="#c0392b", lw=1.2, ls="--")
    ax.text(t, yE + 0.85, "stamp t\n(recorder poll:\nvalue first known)", ha="center", fontsize=9.5, color="#c0392b")
    ax.text(t - P, yE + 0.85, "previous poll\nt - poll", ha="center", fontsize=9, color="#c0392b")
    ax.annotate("", xy=(t - phi, yC - 0.62), xytext=(u, yC - 0.62), arrowprops=dict(arrowstyle="<->", color="#6a3d9a", lw=1.6))
    ax.text((u + t - phi) / 2, yC - 0.55, f"d = {d:g} s (processing, transmission)", ha="center", fontsize=9.5,
            color="#6a3d9a", va="bottom")
    ax.annotate("", xy=(t, yC - 0.62), xytext=(t - phi, yC - 0.62), arrowprops=dict(arrowstyle="<->", color="#b35f00", lw=1.6))
    ax.text(t - phi / 2, yC - 0.55, "phi", ha="center", fontsize=9.5, color="#b35f00", va="bottom")

    # lag brackets (to the stamp)
    yb = 0.75
    bracket(ax, close, t, yb, f"delay from the window end: d + phi = {d + phi:.1f} s  (the anchor; earliest {d:g})", "k")
    bracket(ax, open_, t, yb - 0.55, f"window length T = {T:g} s, so the oldest data is d + phi + T = {d + phi + T:.1f} s old (+ ~one R-R)", "k")
    bracket(ax, centre, t, yb - 1.1, f"middle (derived, not a stamp): d + phi + T/2 = {d + phi + T / 2:.1f} s", "0.45")

    ax.set_xlim(x0, t + 1.4)
    ax.set_ylim(-0.75, yE + 1.45)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("time relative to the recorded stamp t (s)")
    ax.set_title(f"Timing model for one recorded numeric value (HR, median-case numbers: T = {T:g} s, d = {d:g} s, "
                 f"{P:g} s polls; schematic ECG at {60 / RR:.0f} bpm)", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)


if __name__ == "__main__":
    main()
