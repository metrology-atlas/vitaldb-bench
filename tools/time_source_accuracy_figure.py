# SPDX-License-Identifier: MIT
"""Accuracy against resolution for every kind of VitalDB time stamp (log-log; aggregates only).

    python tools/time_source_accuracy_figure.py out.png [out.svg]

Reference: UTC is the latent reference (timing graph). x is measured against the working reference, the recorder
clock via SNUADC stamp-model time; the recorder's own offset from UTC (link A, assumed) is shown as its own entry
and adds to every x when a time is needed against UTC. Labels name the timing-map link (A-S) each entry rests on.
  accuracy (x)   = how far a stamp lies after the END of the interval its value describes (the availability
                   rule: never before it), or a sample stamp's clock error, against the reference clock (s):
                   clock offsets left after (or without) synchronisation, device delays, documentation lag.
  resolution (y) = the width of the interval the time stands for (s): one sample period, a poll interval,
                   an averaging window, a rounding step, a day.
Each entry is an ellipse spanning its [lo, hi] range on both axes (log space). Solid outline = measured in
this audit; dashed = estimated. Arrows join a raw stamp to the same stamp after synchronisation.
Edit ENTRIES as models improve; keep the 'source' strings so each number can be traced.
"""

from __future__ import annotations

import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

GROUP = {"wave": ("#2a78d6", "Waveform samples"), "num": ("#eb6834", "Numeric values (interval they describe)"),
         "rec": ("#1baf7a", "Records, EMR and lab times")}

# key: (group, label, x_lo, x_hi, y_lo, y_hi, measured, label_dx, label_dy (decades), source)
ENTRIES = {
    "snu_raw": ("wave", "SNUADC raw stamps (link E)", 0.01, 0.48, 0.002, 0.002, True, 0.0, -0.75,
                "sawtooth up to -480 ms (step mode corrections)"),
    "snu_model": ("wave", "SNUADC stamp-model time\n(the reference)", 3e-4, 3e-3, 0.002, 0.01, False, -0.2, -0.85,
                  "stamp-model residual where corrections are seen (recordings with none can drift about +-0.1 s/h on the port's rate); 100 Hz files 10 ms"),
    "pleth": ("wave", "PLETH samples\n(oximeter output, link F)", 0.37, 0.40, 0.002, 0.033, True, 0.95, 0.25,
              "one global hardware delay 0.385 s (0.365-0.405 s with the adopted PAT's range); 30 Hz frames"),
    "bis_raw": ("wave", "BIS EEG raw stamps (link K)", 0.53, 1.05, 0.0078, 0.0078, True, 0.75, 0.45,
                "offset p5-p95 -1053..-533 ms on the recorded stamps (191-case pilot)"),
    "bis_sync": ("wave", "BIS EEG after sync", 0.017, 0.042, 0.0078, 0.0078, True, -0.2, 0.55,
                 "17-42 ms (recording's own constant to the mode's; whole release, 2,710 recordings); latency -0.80 s in link K"),
    "primus_raw": ("wave", "Primus AWP/CO2 raw stamps (link P)", 0.6, 5.1, 0.016, 0.016, True, 0.4, -0.6,
                   "offset p10-p90 0.6-5.1 s, median 1.8 s (1,061 recordings) incl. AWP-CVP physiology; +104 ppm vs the PC"),
    "primus_sync": ("wave", "Primus AWP/CO2 after\nbreath matching", 0.055, 0.187, 0.016, 0.016, True, -0.1, 0.55,
                    "leave-one-out error 55 ms median, p90 187 ms, a line per stretch between AWP pauses (1,067 recordings with CVP)"),
    "num_stamp": ("num", "Numeric record stamp itself", 1e-3, 0.02, 1e-4, 1e-3, False, -0.1, -0.75,
                  "1 ms grid (0.1 ms in ~17% of files); receipt latency not measured"),
    "solar_win": ("num", "Solar8000 ART / CVP\nvalue windows (link L)", 0.3, 4.7, 3.0, 8.0, True, -1.3, 0.35,
                  "delay from window end: ART 0.3-2.5 s, CVP 3.1-4.7 s; windows 3-6 s + 2 s poll"),
    "solar_lab": ("num", "Solar8000 ART / CVP after\nper-case delay label", 0.055, 0.40, 3.0, 8.0, True, 0.3, -0.35,
                  "numeric_delay label SE p10-p90: ART 0.06-0.35 s, CVP 0.11-0.40 s (fixed within a case)"),
    "hr_early": ("num", "Solar8000 HR, early group (link L)", 1.05, 2.3, 6.0, 8.0, True, 0.3, -0.35,
                 "delay p10-p90 1.05-2.3 s, 1,775 cases; best window 6-8 s"),
    "hr_late": ("num", "Solar8000 HR, late group (link L)", 3.6, 5.25, 6.0, 8.0, True, 0.3, -0.35,
                "delay p10-p90 3.6-5.25 s, 3,652 cases; one extra 2 s cycle; follows the PC"),
    "hr_lab": ("num", "Solar8000 HR after\nper-case delay label", 0.12, 0.61, 6.0, 8.0, True, 0.3, -0.35,
               "numeric_delay label SE p10-p90 0.12-0.61 s (6,222 cases; 795 group-uncertain)"),
    "pleth_hr": ("num", "PLETH_HR", 2.8, 4.6, 8.0, 12.0, True, -0.95, 0.1,
                 "delay 2.4-4.2 s + 0.4 s PLETH; ~12 beats"),
    "spo2": ("num", "SpO2", 4.0, 4.8, 10.0, 20.0, True, -0.75, 0.3, "onset +4 s (+0.4 s PLETH); window 10-20 s estimated, not measured"),
    "primus_num": ("num", "Primus ETCO2 / PIP / RR (link P)", 0.75, 7.6, 7.0, 30.0, True, 0.75, 0.2,
                   "from window end: ETCO2 0.75-3.4, PIP 3.65-6.5, RR 1.6-7.6 s; 7 s polls, RR over 30 s"),
    "primus_lab": ("num", "Primus ETCO2 / PIP after\nper-case delay label", 0.07, 0.57, 7.0, 30.0, True, 0.3, -0.35,
                   "numeric_delay label SE p10-p90 0.07-0.57 s (RR_CO2 weaker: SE ~1 s)"),
    "bis_num": ("num", "BIS index (link S)", 3.0, 4.0, 16.0, 61.0, True, 0.0, -0.65,
                "3-4 s after its EEG window ends; window 16 s (48%) or 32-61 s (40%); half the weight older than 30 s"),
    "nibp": ("num", "NIBP", 20.0, 40.0, 30.0, 40.0, True, 0.3, 0.35, "result 20-40 s after the cuff measurement"),
    "ev1000": ("num", "EV1000 ART_MBP, raw stamps (link N)", 0.9, 3.25, 20.0, 20.0, True, 0.3, -0.35,
               "from the end of an assumed 20 s window: median 1.6 s (p25 0.9, p95 3.2; 579 cases)"),
    "ev1000_sync": ("num", "EV1000 after MAP alignment", 0.25, 2.4, 20.0, 20.0, True, 0.3, -0.35,
                    "residual SD 0.25-2.4 s (p5-p95); three-way loop closes, median +0.13 s"),
    "vigileo": ("num", "Vigileo SV, CO, SVV", 2.0, 25.0, 20.0, 20.0, True, 0.3, -0.35,
                "SVV vs arterial PPV: aligned in 25 of 343 cases, |offset| 4.7 s median, p95 26 s"),
    "vigilance": ("num", "Vigilance HR_AVG, continuous CO (link O)", 1.0, 175.0, 600.0, 2185.0, False, 0.0, 0.7,
                  "running median over 10-36 min; 0-175 s after its window ends (52-case pilot, 16 at zero)"),
    "cardioq_raw": ("wave", "CardioQ ABP/flow raw stamps (link M)", 0.4, 4.4, 0.0056, 0.0056, True, 0.3, -0.35,
                    "lag vs SNUADC ART p5-p95 0.4-4.4 s; re-anchors ~40 times per case"),
    "cardioq_sync": ("wave", "CardioQ after beat matching", 0.1, 0.3, 0.0056, 0.0056, True, 0.3, -0.35,
                     "per stretch; matched-beat MAD 0.24 s, R-R correlation 0.96 (24 cases)"),
    "invos_fms": ("num", "Invos rSO2, FMS infuser (no shared signal)", 1.0, 100.0, 1.0, 10.0, False, 0.3, -0.35,
                  "not synchronised; FMS flow start -> CVP rise 32 s median (plausible); Invos uninformative"),
    "pumps": ("num", "Infusion pumps, rate and volume (link Q)", 0.1, 2.0, 1.0, 10.0, False, -0.2, 0.55,
              "1 s records; propofol start -> BIS<60 132 s median, none negative (497 cases)"),
    "caseend": ("rec", "caseend", 0.01, 0.5, 1.0, 1.0, True, -0.9, 0.0, "recording end rounded to 1 s (98.7% within 1 s)"),
    "lab_bg": ("rec", "Lab timestamp, blood gas\n(after the draw, link R)", 190.0, 730.0, 1.0, 1.0, True, 0.15, -0.55,
               "3-12 min after the access (modes ~4, ~9 min); the value is known later"),
    "lab_central": ("rec", "Lab timestamp, central lab\n(specimen logged, link R)", 480.0, 1440.0, 1.0, 1.0, True, 0.15, 0.45,
                    "~8-24 min after the access (median ~14; weak); result after an unrecorded turnaround"),
    "lab_draw": ("rec", "Lab draw time (from\narterial access)", 10.0, 100.0, 28.0, 117.0, False, -1.15, 0.0,
                 "access 28-117 s (p10-p90); matched 0-15 min (gas) or 0-30 min (central) before"),
    "emr1": ("rec", "EMR events, minute entries", 30.0, 300.0, 60.0, 60.0, False, -1.55, 0.0,
             "minute grid; documentation lag not yet measured"),
    "emr5": ("rec", "EMR events, 5-min entries (link G)", 150.0, 600.0, 300.0, 300.0, True, 0.15, 0.4,
             "rounded up: 90-290 s later than minute entries; ceiling kernel best in 11/11"),
    "utc": ("rec", "Recorder PC clock vs UTC\n(link A, assumed)", 0.003, 0.9, 0.001, 0.001, False, 0.3, -0.35,
            "assumed NTP sync every 8-24 h, drift 0.1-10 ppm between syncs; not observable in the release"),
    "admdis": ("rec", "Admission / discharge", 3600.0, 86400.0, 86400.0, 86400.0, False, -0.4, 0.45,
               "whole days"),
}
ARROWS = [("snu_raw", "snu_model"), ("hr_late", "hr_lab"), ("solar_win", "solar_lab"), ("primus_num", "primus_lab"), ("bis_raw", "bis_sync"), ("primus_raw", "primus_sync"),
          ("cardioq_raw", "cardioq_sync"), ("ev1000", "ev1000_sync")]


def centre(e):
    _, _, x0, x1, y0, y1 = e[:6]
    return 10 ** ((np.log10(x0) + np.log10(x1)) / 2), 10 ** ((np.log10(y0) + np.log10(y1)) / 2)


def main():
    paper = "--paper" in sys.argv          # print layout for a landscape page: no title, one-line key, print-size type
    if paper:
        sys.argv.remove("--paper")
    S = (dict(fig=(9.7, 5.9), ax=[0.075, 0.105, 0.50, 0.875], kx=[0.60, 0.0, 0.40, 0.985], badge=6.0, lw=1.0, ms=8,
              lab=7.5, tick=7, head=7.2, entry=6.2, step=0.0205, gap=0.008, leg=6.5, detail=False)
         if paper else
         dict(fig=(18.5, 15.5), ax=[0.06, 0.08, 0.60, 0.84], kx=[0.68, 0.03, 0.31, 0.90], badge=8.5, lw=1.8, ms=13,
              lab=11, tick=None, head=10.5, entry=8.6, step=0.0262, gap=0.012, leg=9.5, detail=True))
    fig = plt.figure(figsize=S["fig"])
    ax = fig.add_axes(S["ax"])
    kx = fig.add_axes(S["kx"])
    kx.axis("off")
    XL, YL = (1e-4, 3e5), (5e-5, 3e5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    lax = ax.inset_axes([0, 0, 1, 1], zorder=2)
    lax.set_xlim(np.log10(XL[0]), np.log10(XL[1]))
    lax.set_ylim(np.log10(YL[0]), np.log10(YL[1]))
    lax.axis("off")
    lax.patch.set_alpha(0)
    order = sorted(ENTRIES, key=lambda k: (list(GROUP).index(ENTRIES[k][0]), centre(ENTRIES[k])[1]))
    num = {k: i + 1 for i, k in enumerate(order)}
    for k in order:
        e = ENTRIES[k]
        g, label, x0, x1, y0, y1, meas = e[:7]
        col = GROUP[g][0]
        cx, cy = np.log10(centre(e))
        w = max(np.log10(x1) - np.log10(x0), 0.2)
        h = max(np.log10(y1) - np.log10(y0), 0.2)
        lax.add_patch(Ellipse((cx, cy), w, h, facecolor=col, alpha=0.18, edgecolor="none"))
        lax.add_patch(Ellipse((cx, cy), w, h, facecolor="none", edgecolor=col, lw=S["lw"],
                              ls="-" if meas else (0, (4, 3))))
    for a, b in ARROWS:
        (x0, y0), (x1, y1) = np.log10(centre(ENTRIES[a])), np.log10(centre(ENTRIES[b]))
        lax.annotate("", xy=(x1 + 0.15, y1), xytext=(x0 - 0.15, y0),
                     arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=1.2 if not paper else 0.8, ls="--", mutation_scale=S["ms"]))
    # number badges, nudged apart where centres collide
    placed = []
    cands = [(0, 0)] + [(r * np.cos(a), r * np.sin(a)) for r in (0.35, 0.6, 0.85) for a in np.radians(np.arange(0, 360, 45))]
    for k in order:
        cx, cy = np.log10(centre(ENTRIES[k]))
        for ox, oy in cands:
            bx, by = cx + ox, cy + oy
            if all(np.hypot(bx - px, by - py) > 0.3 for px, py in placed):
                break
        placed.append((bx, by))
        col = GROUP[ENTRIES[k][0]][0]
        if (bx, by) != (cx, cy):
            lax.plot([cx, bx], [cy, by], color=col, lw=0.9)
            lax.plot(cx, cy, "o", color=col, ms=3)
        lax.text(bx, by, str(num[k]), ha="center", va="center", fontsize=S["badge"], fontweight="bold", color="#0b0b0b",
                 bbox=dict(boxstyle="circle,pad=0.18", facecolor="white", edgecolor=col, lw=1.2 if not paper else 0.8), zorder=5)
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=40))          # a major tick at every power of 10
        axis.set_major_formatter(LogFormatterSciNotation(base=10))
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=400))
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="major", length=6 if not paper else 3.5)
    if S["tick"]:
        ax.tick_params(labelsize=S["tick"])
    ax.tick_params(which="minor", length=3)
    ax.grid(which="major", axis="both", color="0.80", lw=0.9)
    ax.grid(which="minor", axis="both", color="0.93", lw=0.5)
    ax.set_axisbelow(True)
    if paper:
        ax.set_xlabel("Accuracy: delay from the end of the interval a value describes to its stamp,\n"
                      "or a stamp's clock error, on the working reference (s)", fontsize=S["lab"])
    else:
        ax.set_xlabel("Accuracy: delay from the end of the interval a value describes to its stamp (or a stamp's clock "
                      "error), on the working reference; add link A for UTC (s)", fontsize=S["lab"])
    ax.set_ylabel("Resolution: width of the interval a time stands for (s)", fontsize=S["lab"])
    if not paper:
        fig.text(0.06, 0.955, "VitalDB time stamps: accuracy and resolution", fontsize=14, fontweight="bold")
        fig.text(0.06, 0.93, "Current best estimates from the audit; ellipses span the measured or estimated range on each axis. "
             "To be updated as temporal models and synchronisation improve.", fontsize=9.5, color="#52514e")
    # key
    y = 0.99
    for g, (col, gl) in GROUP.items():
        if paper:
            gl = {"wave": "Waveforms", "num": "Numerics", "rec": "Records"}[g]
        kx.text(0.0, y, gl, fontsize=S["head"], fontweight="bold", color="#0b0b0b", transform=kx.transAxes, va="top")
        kx.add_patch(plt.Rectangle((-0.04, y - S["step"] * 0.84), 0.025, S["step"] * 0.69, color=col,
                                   transform=kx.transAxes, clip_on=False))
        y -= S["step"] * (1.03 if not paper else 1.15)
        for k in [k for k in order if ENTRIES[k][0] == g]:
            e = ENTRIES[k]
            lab = e[1].replace("\n", " ")
            kx.text(0.0, y, f"{num[k]:>2}", fontsize=S["entry"], fontweight="bold", transform=kx.transAxes, va="top")
            if paper:   # names only: the line style already says measured or estimated
                lab = re.sub(r"\s*\([^)]*\)", "", lab).strip()
                lab = {"caseend": "Case end"}.get(lab, lab)
            else:
                lab += "" if e[6] or "assumed" in lab else "  (estimated)"
            kx.text(0.06, y, lab, fontsize=S["entry"], transform=kx.transAxes, va="top")
            if S["detail"]:
                kx.text(0.06, y - 0.0118, e[9], fontsize=7.3, color="#52514e", transform=kx.transAxes, va="top")
            y -= S["step"]
        y -= S["gap"]
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="0.3", lw=1.8, label="measured in this audit"),
               Line2D([0], [0], color="0.3", lw=1.8, ls=(0, (4, 3)), label="estimated"),
               Line2D([0], [0], color="#52514e", lw=1.2, ls="--", marker=">", label="raw -> after synchronisation or per-case label")]
    ax.legend(handles=handles, loc="upper left", fontsize=S["leg"], framealpha=0.95)
    fig.savefig(sys.argv[1], dpi=120 if not paper else 300)
    if len(sys.argv) > 2:
        fig.savefig(sys.argv[2])


if __name__ == "__main__":
    main()
