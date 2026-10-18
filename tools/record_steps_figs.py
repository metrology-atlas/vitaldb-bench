# SPDX-License-Identifier: MIT
"""Figures for record steps (tools/record_steps_collect.py tables). Aggregates only.

    python tools/record_steps_figs.py DIR

steps_magnitude.png  every non-zero step between records, forward (gap) and backward (overlap), by |step|, log-log;
                     marks at one sample period and one message
steps_remainder.png  forward gaps longer than one message: the remainder of the gap modulo the message duration, as a
                     fraction of a message (whole messages lost -> 0 or 1; resynchronisation -> anywhere)
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TR = ["SNUADC__ECG_II", "SNUADC__PLETH", "SNUADC__ART", "SNUADC__CVP", "Primus__CO2", "Primus__AWP", "BIS__EEG1_WAV", "CardioQ__FLOW"]


def main():
    d = Path(sys.argv[1])
    D = pd.concat([pd.read_parquet(p, columns=["case", "track", "fs", "step", "msg"]) for p in sorted(glob.glob(str(d / "steps_*.parquet")))])
    D = D[D.track.isin(TR)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.logspace(-6, 5, 221)
    fig, axs = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
    for a_, tr in zip(axs.flat, TR):
        x = D[D.track == tr]
        T = 1 / x.fs.iloc[0]
        msg = x.msg.iloc[0] * T
        f, b = x.step[x.step > 0], -x.step[x.step < 0]
        a_.hist(f, bins=edges, histtype="step", color="#08519c", lw=1.1, label=f"forward (gap): {len(f):,}")
        a_.hist(b, bins=edges, histtype="step", color="#e6550d", lw=1.1, label=f"backward (overlap): {len(b):,}")
        for v, lab, c in ((T, "1 sample", "#999"), (msg, "1 message", "#333")):
            a_.axvline(v, color=c, ls=":", lw=.8)
            a_.text(v, 1.02, lab, transform=a_.get_xaxis_transform(), fontsize=6.5, ha="center", color=c)
        a_.set_xscale("log"); a_.set_yscale("log")
        a_.set_title(f"{tr.replace('__', '/')}: {x.case.nunique():,} cases; message {x.msg.iloc[0]} samples = {msg:.3g} s",
                     fontsize=8, loc="left", pad=12)
        a_.set_ylabel("steps per bin", fontsize=7.5)
        a_.legend(frameon=False, fontsize=7, loc="upper right")
        a_.tick_params(labelsize=7.5)
        a_.spines[["top", "right"]].set_visible(False)
    for a_ in axs[-1]:
        a_.set_xlabel("|step| between records (s), log; 20 bins per decade", fontsize=8)
    fig.tight_layout(); fig.savefig(d / "steps_magnitude.png", dpi=140); plt.close(fig)

    fig, axs = plt.subplots(4, 2, figsize=(11, 10))
    for a_, tr in zip(axs.flat, TR):
        x = D[D.track == tr]
        T = 1 / x.fs.iloc[0]
        msg = x.msg.iloc[0] * T
        g = x.step[x.step > msg]
        r = np.mod(g / msg, 1.0)
        a_.hist(r, bins=np.linspace(0, 1, 101), color="#08519c")
        a_.set_title(f"{tr.replace('__', '/')}: {len(g):,} gaps longer than one message ({msg:.3g} s)", fontsize=8, loc="left")
        a_.set_ylabel("gaps", fontsize=7.5)
        a_.tick_params(labelsize=7.5)
        a_.spines[["top", "right"]].set_visible(False)
    for a_ in axs[-1]:
        a_.set_xlabel("gap modulo the message duration, as a fraction of a message", fontsize=8)
    fig.tight_layout(); fig.savefig(d / "steps_remainder.png", dpi=140); plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
