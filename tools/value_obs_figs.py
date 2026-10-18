# SPDX-License-Identifier: MIT
"""Figures for the value-space and record-structure observations (19 Oct). This machine only.

    python tools/value_obs_figs.py --survey C:/vitalDB/work/signal_survey --root R --out DIR
Writes primus-waveform-gains.png, cardioq-flow-segmented.png, snuadc-ten-bit-codes.png, interleaved-records.png,
duplicated-records.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GREY = "#1f5fa6", "#e08a1e", "#7f7f7f"


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def primus(s, out):
    d = pd.read_csv(s / "numeric_vs_wave" / "by_case.csv")
    d = d[(d.window == 7.0) & (d.lag == 0.0)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.linspace(0.6, 1.1, 101)
    for ax, num, wav, cfgs in [
        (axs[0], "Primus/ETCO2", "Primus/CO2@0.075", [(0.1, "CO2 stored at 0.1"), (0.075, "CO2 stored at 0.075")]),
        (axs[1], "Primus/PIP_MBAR", "Primus/AWP@0.1", [(0.1, "AWP stored at 0.1 / −20"), (0.1064, "AWP stored at 0.1064 / −22")])]:
        for (g, lab), c in zip(cfgs, (ORANGE, BLUE)):
            x = d[(d.track == num) & np.isclose(d.wave_gain, g)].med_ratio.dropna()
            ax.hist(x.clip(bins[0], bins[-1]), bins=bins, color=c, alpha=0.85, label=f"{lab} ({len(x):,})")
        y = d[(d.track == f"{num} vs {wav}")].med_ratio.dropna()
        ax.hist(y.clip(bins[0], bins[-1]), bins=bins, histtype="step", color="k", lw=1.2,
                label=f"all, rescaled to {'0.075' if 'CO2' in wav else '0.1 / −20'} ({len(y):,})")
        ax.axvline(1, color=GREY, ls=":", lw=1)
        ax.set_xlabel(f"{num.split('/')[1]} numeric ÷ waveform peak (median per recording)")
        ax.set_ylabel("recordings"); ax.set_yscale("log"); ax.legend(fontsize=8, frameon=False, loc="upper left"); clean(ax)
    axs[0].set_title("ETCO2 against the CO2 waveform", loc="left", fontsize=10)
    axs[1].set_title("PIP against the AWP waveform", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "primus-waveform-gains.png", dpi=130); plt.close(fig)


def cardioq(s, out):
    z = np.load(s / "agg_events" / "hist_CardioQ__FLOW.npz"); c, n = z["codes"], z["counts"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axs[0]
    ax.vlines(c, 0.8, n, color=BLUE, lw=0.6)
    for x, lab in [(256, "step 2"), (512, "step 4")]:
        ax.axvline(x, color=ORANGE, ls="--", lw=1); ax.text(x + 8, n.max() * 0.5, lab, color=ORANGE, fontsize=8)
    ax.text(20, n.max() * 0.5, "step 1", color=ORANGE, fontsize=8)
    ax.set_yscale("log"); ax.set_xlim(-10, 1030); ax.set_xlabel("stored FLOW value"); ax.set_ylabel("samples")
    ax.set_title("Every stored value (one line each): 509 values in three scales", loc="left", fontsize=10); clean(ax)
    ax = axs[1]
    m = (c >= 100) & (c <= 140)
    ax.bar(c[m], n[m], width=0.8, color=[BLUE if v % 2 == 0 else ORANGE for v in c[m]])
    ax.set_yscale("log"); ax.set_xlabel("stored value (100–140)"); ax.set_ylabel("samples")
    ax.set_title("Even values (blue) dominate below 256", loc="left", fontsize=10); clean(ax)
    fig.tight_layout(); fig.savefig(out / "cardioq-flow-segmented.png", dpi=130); plt.close(fig)


def snuadc(s, out):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for tr, col, lab in [("SNUADC__ECG_II", BLUE, "ECG II"), ("SNUADC__ART", ORANGE, "ART")]:
        z = np.load(s / "agg_events" / f"hist_{tr}.npz")
        ax.step(z["codes"], z["counts"], where="mid", color=col, lw=0.9, label=lab)
    ax.axvline(1023.5, color="k", ls=":", lw=1); ax.text(1035, 1e9, "1023: top of a 10-bit range", fontsize=8)
    ax.axvspan(1024, 1300, color=GREY, alpha=0.12); ax.text(1060, 3e2, "no samples above 1023\n(a 12-bit range would\nreach 4095)", fontsize=8)
    ax.set_yscale("log"); ax.set_xlim(-20, 1300); ax.set_xlabel("stored code (one bin per code)"); ax.set_ylabel("samples")
    ax.legend(frameon=False, fontsize=8, loc="upper left"); clean(ax)
    ax.set_title("SNUADC codes over the whole release: 0–1023 and nothing above", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "snuadc-ten-bit-codes.png", dpi=130); plt.close(fig)


def records(root, case, track):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: PLC0415
    tr = read_vital(Path(root) / "vital_files" / f"{case:04d}.vital").tracks[track]
    rt = np.asarray(tr.record_times, float); sz = np.asarray(tr.record_sizes, int)
    x = np.asarray(tr.values, float) * float(tr.gain) + float(tr.offset)
    return rt, sz, x, float(tr.srate)


def interleaved(root, out, case=5327, k0=20000, nrec=8):
    rt, sz, x, fs = records(root, case, "SNUADC/ECG_II")
    st = np.r_[0, np.cumsum(sz)[:-1]]
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.2), sharex=False)
    ax = axs[0]
    seq = np.concatenate([x[st[k]:st[k] + sz[k]] for k in range(k0, k0 + nrec)])
    ax.plot(np.arange(seq.size) / fs, seq, color=BLUE, lw=0.8)
    for j in range(1, nrec):
        ax.axvline(j * sz[k0] / fs, color=GREY, lw=0.5, ls=":")
    ax.set_title("Eight consecutive records in file order: the signal jumps at every boundary", loc="left", fontsize=10)
    ax.set_ylabel("ECG II (mV)"); clean(ax)
    ax = axs[1]
    for par, col, lab in [(0, BLUE, "records k, k+2, ..."), (1, ORANGE, "records k+1, k+3, ...")]:
        ks = range(k0 + par, k0 + nrec + 8, 2)
        y = np.concatenate([x[st[k]:st[k] + sz[k]] for k in ks])
        ax.plot(np.arange(y.size) / fs, y, color=col, lw=0.8, label=lab)
    ax.set_title("Every other record joined up: two continuous, unrelated signals", loc="left", fontsize=10)
    ax.set_xlabel("s (samples ÷ 500 Hz)"); ax.set_ylabel("ECG II (mV)"); ax.legend(frameon=False, fontsize=8); clean(ax)
    fig.tight_layout(); fig.savefig(out / "interleaved-records.png", dpi=130); plt.close(fig)


def duplicated(root, out, case=5595, k0=30000, nrec=40):
    rt, sz, x, fs = records(root, case, "Primus/AWP")
    st = np.r_[0, np.cumsum(sz)[:-1]]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axs[0]
    iv = np.diff(rt[k0:k0 + nrec + 1])
    ax.plot(np.arange(iv.size), iv, "o", color=BLUE, ms=4)
    ax.set_xlabel("record"); ax.set_ylabel("stamp interval (s)"); ax.set_yticks([0, 0.24, 0.48])
    ax.set_title("Stamps alternate 0 and 0.48 s", loc="left", fontsize=10); clean(ax)
    ax = axs[1]
    for k in range(k0, k0 + 12):
        t = rt[k] - rt[k0] + np.arange(sz[k]) / fs
        same = k > 0 and np.array_equal(x[st[k]:st[k] + sz[k]], x[st[k - 1]:st[k - 1] + sz[k - 1]])
        ax.plot(t, x[st[k]:st[k] + sz[k]] + (0 if not same else 0.6), color=ORANGE if same else BLUE, lw=1)
    ax.set_xlabel("s from the first record (by stamp)"); ax.set_ylabel("AWP (stored scale)")
    ax.set_title("Each record written twice (orange, drawn 0.6 higher): identical copies", loc="left", fontsize=10)
    clean(ax)
    fig.tight_layout(); fig.savefig(out / "duplicated-records.png", dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--survey", required=True); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    s, out = Path(a.survey), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    primus(s, out); cardioq(s, out); snuadc(s, out); interleaved(a.root, out); duplicated(a.root, out)


if __name__ == "__main__":
    main()
