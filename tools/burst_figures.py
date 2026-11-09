# SPDX-License-Identifier: MIT
"""Example-trace figures of the shared-burst studies, from the whole-release burst check (metrology-atlas studies
shared-bursts, figure shared-bursts-examples, and snuadc-stamp-sawtooth, figure stamp-model-validation).

    python tools/burst_figures.py examples --stamps $W/burst_validation_stamps --drift $W/results/sync_full/full.drift.csv \
        --out $W/burst_figures
    python tools/burst_figures.py validation --stamps $W/burst_validation_stamps --model $W/burst_validation_m3 \
        --modes $W/results/sync_full/snuadc_mode_label.json --levels $W/stamp_error_levels/summary.json --out $W/burst_figures

Inputs are the per-recording burst lags of tools/sa_corrected.py on every recording, on the recorded stamps (variant
``stamps``) and on the adopted stamp model (``model3``): for each 10-minute window (stepped 5 min) the EEG-minus-ECG
lag of the shared broadband bursts, its peak correlation and the best peak more than 1 s away. Good windows as in
tools/burst_validation_agg.py: peak >= 0.3, margin >= 0.15, lag between -2,500 and +500 ms (ECG pairs); the
EEG1-EEG2 control is taken within +/-500 ms. Summary panels are over recordings with at least 5 good windows.
Each subcommand writes its PNG and a JSON of the numbers it shows.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXAMPLES = [750, 5534, 5311, 5780, 1936, 5872, 2778, 359]
VALIDATION = [4644, 359, 2580, 5780, 750, 1703]
LAG = (-2500.0, 500.0)


def load(folder: str, variant: str) -> dict:
    out = {}
    for f in sorted(glob.glob(str(Path(folder) / "part_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            m = r.get(variant)
            if m and "burst" in m:
                out[r["case"]] = m
    return out


def good(rows, lag=LAG) -> np.ndarray:
    a = np.array(rows or [], float).reshape(-1, 5)
    return a[(a[:, 2] >= 0.3) & (a[:, 2] - a[:, 3] >= 0.15) & (a[:, 1] > lag[0]) & (a[:, 1] < lag[1])]


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def examples(a) -> None:
    B = load(a.stamps, "stamps")
    d = pd.read_csv(a.drift)
    drift = {(int(r.case), r.device_type): r.drift_ppm for r in d.itertuples()}
    plt = plt_()
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 4)
    for i, c in enumerate(EXAMPLES):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        bb = B[c]["burst"]
        e1, e2 = good(bb.get("ecg~eeg1")), good(bb.get("ecg~eeg2"))
        ee = good(bb.get("eeg1~eeg2"), (-500, 500))
        ax.scatter(ee[:, 0] / 3600, ee[:, 1] - 750, s=4, color="0.6", label="eeg1~eeg2 (−750 ms)")
        ax.scatter(e1[:, 0] / 3600, e1[:, 1], s=14, color="C0", label="ecg~eeg1")
        ax.scatter(e2[:, 0] / 3600, e2[:, 1], s=14, marker="x", color="C1", label="ecg~eeg2")
        ax.set_ylim(-1300, -200)
        s_, b_ = drift.get((c, "SNUADC")), drift.get((c, "BIS"))
        ax.set_title(f"case {c}  SNUADC {s_:.1f} ppm, BIS {b_:.1f} ppm" if s_ is not None and b_ is not None else f"case {c}", fontsize=8)
        ax.set_xlabel("hours")
        ax.set_ylabel("EEG lag vs ECG (ms)")
        if i == 0:
            ax.legend(fontsize=6)
    lag, ctrl, same = [], [], []
    n_rec = 0
    for c, m in B.items():
        bb = m["burst"]
        e1 = good(bb.get("ecg~eeg1"))
        if len(e1) < 5:
            continue
        n_rec += 1
        lag += list(e1[:, 1])
        ctrl += list(good(bb.get("eeg1~eeg2"), (-500, 500))[:, 1])
        e2 = good(bb.get("ecg~eeg2"))
        k2 = {t: l for t, l, *_ in e2}
        same += [l - k2[t] for t, l, *_ in e1 if t in k2]
    S = {"recordings_5_good": n_rec, "windows": len(lag), "lag_ms_p5_25_50_75_95": [round(float(x), 1) for x in np.percentile(lag, [5, 25, 50, 75, 95])],
         "eeg1_eeg2_control_ms_median_abs": round(float(np.median(np.abs(ctrl))), 1), "control_windows": len(ctrl),
         "same_window_eeg1_minus_eeg2_ms_median_abs_p90": [round(float(np.median(np.abs(same))), 1), round(float(np.percentile(np.abs(same), 90)), 1)],
         "same_windows": len(same), "cases": EXAMPLES}
    for j, (v, rng, title, xl) in enumerate(((lag, (-2100, 0), f"all good windows, ECG~EEG1 ({len(lag):,} windows, {n_rec:,} recordings)", "lag (ms)"),
                                             (ctrl, (-60, 60), "control: EEG1~EEG2 (same device)", "lag (ms)"),
                                             (same, (-100, 100), "same window: (ECG~EEG1) − (ECG~EEG2)", "ms"))):
        ax = fig.add_subplot(gs[2, j])
        ax.hist(np.clip(v, *rng), bins=60, range=rng, color="C0")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xl + "; values outside the axis in the end bins")
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    ax.text(0, 0.5, "Negative lag: the same burst is stamped\nearlier in BIS EEG than in SNUADC ECG.\n"
            "Windows 10 min, step 5 min; good = r>=0.3\nand margin>=0.15 over lags >1 s away.\n"
            "Whole release, recorded stamps.", fontsize=9, va="center")
    fig.suptitle("Shared broadband bursts (electrocautery/EMG) in SNUADC ECG and BIS EEG: stamp offset between devices")
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "shared-bursts-examples.png", dpi=80)
    (out / "shared-bursts-examples.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


def validation(a) -> None:
    Bs, Bm = load(a.stamps, "stamps"), load(a.model, a.variant)
    modes = {int(k): v for k, v in json.loads(Path(a.modes).read_text()).items()}
    L = json.loads(Path(a.levels).read_text())
    plt = plt_()
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 4)
    for i, c in enumerate(VALIDATION):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        gs_, gm = good(Bs[c]["burst"].get("ecg~eeg1")), good(Bm[c]["burst"].get("ecg~eeg1"))
        ax.scatter(gs_[:, 0] / 3600, gs_[:, 1], s=12, color="0.7", label="recorded stamps")
        ax.scatter(gm[:, 0] / 3600, gm[:, 1], s=12, color="C0", label="stamp model")
        cl = Bm[c].get("clocks", {})
        def rate(k):
            x = cl.get(k, {})
            r = x.get("rate_ppm")
            return f"{r:.0f} ppm ({x.get('source')})" if r is not None else f"0 ppm ({x.get('source')})"
        ax.set_title(f"case {c}: SNUADC {rate('ecg')}, BIS {rate('eeg1')}", fontsize=8)
        ax.set_ylim(-1300, -300)
        ax.set_xlabel("hours")
        ax.set_ylabel("EEG minus ECG (ms)")
        if i == 0:
            ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[0:2, 3])
    names = list(L["levels"])
    med = [L["levels"][k]["median_ms"] for k in names]
    p90 = [L["levels"][k]["p90_ms"] for k in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, med, 0.4, label="median")
    ax.bar(x + 0.2, p90, 0.4, label="90th percentile")
    for xx, m_, p_ in zip(x, med, p90):
        ax.text(xx - 0.2, m_, f"{m_:.0f}", ha="center", va="bottom", fontsize=7)
        ax.text(xx + 0.2, p_, f"{p_:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["recorded\nstamps", "stamps −\nconstant", "model −\nconstant", "model −\nmode const.", "model −\nown const."], fontsize=7)
    ax.set_ylabel("|EEG−ECG timing error| per 10-min window (ms)")
    ax.set_title(f"Timing error by correction level\n({L['windows']:,} windows, {L['recordings']:,} recordings)", fontsize=8)
    ax.legend(fontsize=7)
    const, sd_s, sd_m = {"step": [], "small": []}, [], []
    for c, m in Bm.items():
        gm = good(m["burst"].get("ecg~eeg1"))
        if len(gm) < 5:
            continue
        md = modes.get(c)
        if md in const:
            const[md].append(float(np.median(gm[:, 1])))
        sd_m.append(1.4826 * float(np.median(np.abs(gm[:, 1] - np.median(gm[:, 1])))))
        if c in Bs:
            g_ = good(Bs[c]["burst"].get("ecg~eeg1"))
            if len(g_) >= 5:
                sd_s.append(1.4826 * float(np.median(np.abs(g_[:, 1] - np.median(g_[:, 1])))))
    ax = fig.add_subplot(gs[2, 0:2])
    for md, col in (("step", "C0"), ("small", "C1")):
        v = const[md]
        ax.hist(v, bins=np.arange(-1100, -400, 10), alpha=0.7, color=col, label=f"SNUADC {md} mode ({len(v):,} recordings, median {np.median(v):.0f} ms)")
    ax.set_xlabel("constant EEG−ECG offset left after the model, per recording (ms)")
    ax.set_ylabel("recordings")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[2, 2:4])
    ax.hist(np.clip(sd_s, 0, 300), bins=np.arange(0, 305, 5), color="0.7", alpha=0.8, label=f"recorded stamps (median {np.median(sd_s):.0f} ms)")
    ax.hist(np.clip(sd_m, 0, 300), bins=np.arange(0, 305, 5), color="C0", alpha=0.7, label=f"stamp model (median {np.median(sd_m):.0f} ms)")
    ax.set_xlabel("within-recording SD of the EEG−ECG offset (ms, 1.4826 × MAD; over 300 in the last bin)")
    ax.set_ylabel("recordings")
    ax.legend(fontsize=7)
    fig.suptitle("Stamp sawtooth model (physiosync.stamp_clock) validated against shared ECG/EEG bursts (whole release)")
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "stamp-model-validation.png", dpi=80)
    S = {"cases": VALIDATION, "constants_ms": {k: {"n": len(v), "median": round(float(np.median(v)), 1)} for k, v in const.items()},
         "within_sd_ms_median": {"stamps": round(float(np.median(sd_s)), 1), "model": round(float(np.median(sd_m)), 1), "n_stamps": len(sd_s), "n_model": len(sd_m)},
         "levels_from": a.levels}
    (out / "stamp-model-validation.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("examples")
    p.add_argument("--stamps", required=True)
    p.add_argument("--drift", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("validation")
    p.add_argument("--stamps", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--variant", default="model3", help="the stamp-model variant in --model (model4 from 9 Nov 2026)")
    p.add_argument("--modes", required=True)
    p.add_argument("--levels", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"examples": examples, "validation": validation}[a.cmd](a)


if __name__ == "__main__":
    main()
