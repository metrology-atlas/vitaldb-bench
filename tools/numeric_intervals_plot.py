# SPDX-License-Identifier: MIT
"""Plot tools/numeric_intervals.py output (aggregates only).

    python tools/numeric_intervals_plot.py all.json out_prefix

out_prefix_dist.png: per numeric, intervals between records (grey) and between changes of value (colour),
log time axis. out_prefix_jitter.png: record intervals near the nominal period, 5 ms bins.
"""

from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"Solar8000": "#1f5fa8", "Primus": "#b5541c", "BIS": "#2c8a4b"}
WAVE = {  # source waveform recorded alongside, for the panel subtitle
    "Solar8000/HR": "ECG", "Solar8000/PLETH_HR": "PLETH", "Solar8000/PLETH_SPO2": "(PLETH, value not computable)",
    "Solar8000/ART_SBP": "ART", "Solar8000/ART_MBP": "ART", "Solar8000/ART_DBP": "ART",
    "Solar8000/FEM_SBP": "FEM", "Solar8000/FEM_MBP": "FEM", "Solar8000/FEM_DBP": "FEM", "Solar8000/CVP": "CVP",
    "Solar8000/ST_II": "ECG", "Solar8000/RR": "(ECG impedance, not recorded)", "Solar8000/RR_CO2": "Primus CO2?",
    "Solar8000/ETCO2": "Primus CO2?", "Solar8000/VENT_PIP": "Primus AWP", "Solar8000/VENT_RR": "Primus AWP",
    "Solar8000/NIBP_SBP": "none", "Primus/ETCO2": "CO2", "Primus/RR_CO2": "CO2", "Primus/PIP_MBAR": "AWP",
    "Primus/PEEP_MBAR": "AWP", "Primus/MAWP_MBAR": "AWP", "Primus/TV": "(flow, not recorded)",
    "Primus/MV": "(flow, not recorded)", "BIS/BIS": "EEG", "BIS/SEF": "EEG", "BIS/EMG": "EEG", "BIS/SR": "EEG",
    "BIS/SQI": "EEG"}


def main():
    J = json.load(open(sys.argv[1]))
    pre = sys.argv[2]
    E = np.asarray(J["log_edges"])
    tracks = [(n, a) for n, a in J["tracks"].items() if a["cases"]]
    nc = 5
    nr = int(np.ceil(len(tracks) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.0 * nc, 2.7 * nr), sharex=True)
    for ax, (n, a) in zip(axs.flat, tracks):
        dev = n.split("/")[0]
        for key, kw in (("rec", dict(color="0.55", alpha=0.45, step="post", label="between records")),
                        ("chg", dict(color=COL[dev], alpha=None, label="between value changes"))):
            h = np.asarray(a[key], float)
            if h.sum() == 0:
                continue
            f = h / h.sum()
            if key == "rec":
                ax.fill_between(E[:-1], f, **kw)
            else:
                ax.step(E[:-1], f, where="post", color=kw["color"], lw=1.4, label=kw["label"])
        cm = np.asarray(a["case_median_chg"]) if a["case_median_chg"] else None
        sub = f"{a['cases']} cases; from {WAVE.get(n, '?')}"
        if cm is not None and cm.size:
            sub += f"\nchange median {np.median(cm):.1f} s (cases p10-p90 {np.percentile(cm, 10):.1f}-{np.percentile(cm, 90):.1f})"
        ax.set_title(n, fontsize=10.5, fontweight="bold", color=COL[dev], loc="left")
        ax.text(0.02, 0.97, sub, transform=ax.transAxes, va="top", fontsize=7.5, color="0.25")
        ax.set_xscale("log")
        ax.set_xlim(0.5, 600)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, which="both")
    for ax in axs.flat[len(tracks):]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("interval (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel("fraction of intervals")
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", fontsize=10, ncol=2)
    fig.suptitle("VitalDB numerics: time between records and between changes of value (1,000 random cases; gaps > 60 s excluded)",
                 fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(pre + "_dist.png", dpi=110)

    fe = np.arange(0, 15.0001, 0.005)
    fc = (fe[:-1] + fe[1:]) / 2
    sel = [("Solar8000/HR", 1.9, 2.15), ("Solar8000/ST_II", -0.01, 2.15), ("BIS/BIS", 0.95, 1.1),
           ("Primus/ETCO2", 5.5, 8.5)]
    fig, axs = plt.subplots(1, len(sel), figsize=(4.4 * len(sel), 3.2))
    for ax, (n, lo, hi) in zip(axs, sel):
        f = np.asarray(J["tracks"][n]["fine"], float)
        f = f / f.sum()
        m = (fc >= lo) & (fc <= hi)
        ax.bar(fc[m], f[m], width=0.005, color=COL[n.split("/")[0]])
        ax.set_title(n, fontsize=10.5, loc="left", fontweight="bold")
        ax.set_xlabel("record interval (s), 5 ms bins")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("fraction of intervals")
    fig.suptitle("Record-interval jitter near the nominal period", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(pre + "_jitter.png", dpi=110)


if __name__ == "__main__":
    main()
