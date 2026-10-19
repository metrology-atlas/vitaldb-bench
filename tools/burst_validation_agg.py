# SPDX-License-Identifier: MIT
"""Summarise the full-release burst validation of the stamp model (tools/sa_corrected.py --variants model).

    python tools/burst_validation_agg.py --inp C:/vitalDB/work/burst_validation --modes .../snuadc_mode_label.json \
        --ports .../snuadc_mode.json [--out DIR]

Good burst windows as in the validation (vitaldb_bench.clock_model): peak r >= 0.3, peak minus the best peak more than
1 s away >= 0.15, EEG-minus-ECG lag between -2,500 and +500 ms, pair ECG II ~ EEG1. Per recording with at least 5 good
windows: the constant (median lag), the within-recording spread (1.4826 x MAD), the slope over the recording (Theil-Sen, ms/h),
and lag jumps (successive good windows differing by more than 100 ms). Errors per window: lag minus the recording's
own constant, and lag minus its SNUADC mode's constant (leave-one-out median of the other recordings' constants).
Writes per_recording.csv, windows.parquet, summary.json and burst_validation.png.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

GOOD_PEAK, GOOD_MARGIN, LAG_RANGE_MS = 0.3, 0.15, (-2500.0, 500.0)
BLUE, ORANGE, GREY = "#1f5fa6", "#e08a1e", "#7f7f7f"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True); p.add_argument("--modes", required=True); p.add_argument("--ports", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--variant", default="model", help="which sa_corrected variant to summarise (model, model2)")
    a = p.parse_args()
    inp = Path(a.inp); out = Path(a.out or inp); out.mkdir(exist_ok=True)
    modes = {int(k): v for k, v in json.loads(Path(a.modes).read_text()).items()}
    ports = {int(k): v for k, v in json.loads(Path(a.ports).read_text()).items()}
    win, rec, n_cases, n_err = [], [], 0, 0
    for f in glob.glob(str(inp / "part_*.jsonl")):
        for line in open(f):
            r = json.loads(line); n_cases += 1
            m = r.get(a.variant)
            if not m or "burst" not in m:
                n_err += 1; continue
            rows = np.array(m["burst"].get("ecg~eeg1") or [], float).reshape(-1, 5)
            cl = m.get("clocks", {})
            for t, lag, pk, sec, act in rows:
                win.append((r["case"], t, lag, pk, sec))
            rec.append(dict(case=r["case"], ecg_source=cl.get("ecg", {}).get("source"), ecg_rate=cl.get("ecg", {}).get("rate_ppm"),
                            ecg_teeth=cl.get("ecg", {}).get("teeth"), eeg_source=cl.get("eeg1", {}).get("source"),
                            eeg_rate=cl.get("eeg1", {}).get("rate_ppm"), eeg_teeth=cl.get("eeg1", {}).get("teeth"),
                            gaps=cl.get("ecg", {}).get("gaps")))
    W = pd.DataFrame(win, columns=["case", "t", "lag_ms", "peak", "second"])
    R = pd.DataFrame(rec).set_index("case")
    g = W[(W.peak >= GOOD_PEAK) & (W.peak - W.second >= GOOD_MARGIN) & (W.lag_ms > LAG_RANGE_MS[0]) & (W.lag_ms < LAG_RANGE_MS[1])].copy()
    cnt = g.groupby("case").size()
    keep = cnt[cnt >= 5].index
    g = g[g.case.isin(keep)].sort_values(["case", "t"])
    def per(x):
        y = x.lag_ms.to_numpy(); t = x.t.to_numpy() / 3600
        mad = 1.4826 * np.median(np.abs(y - np.median(y)))
        from scipy.stats import theilslopes  # robust to the occasional wrong burst window
        slope = theilslopes(y, t)[0] if np.ptp(t) > 0.25 else np.nan
        jumps = int((np.abs(np.diff(y)) > 100).sum())
        return pd.Series(dict(n=len(y), const_ms=float(np.median(y)), sd_ms=float(mad), slope_ms_h=float(slope), jumps=jumps,
                              hours=float(np.ptp(t))))
    P = g.groupby("case").apply(per).join(R)
    P["mode"] = [modes.get(c, "unknown") for c in P.index]
    P["snuadc_port"] = [ports.get(c, {}).get("port") for c in P.index]
    P["bis_port"] = [ports.get(c, {}).get("bis_port") for c in P.index]
    # leave-one-out mode constant
    loo = {}
    for md, grp in P.groupby("mode"):
        v = grp.const_ms
        for c in grp.index:
            loo[c] = float(np.median(v.drop(c))) if len(v) > 1 else np.nan
    P["mode_const_loo"] = pd.Series(loo)
    g = g.join(P[["const_ms", "mode_const_loo", "mode"]], on="case")
    g["err_own"] = g.lag_ms - g.const_ms
    g["err_mode"] = g.lag_ms - g.mode_const_loo
    P.to_csv(out / "per_recording.csv"); g.to_parquet(out / "windows.parquet")
    q = lambda s, qs=(.5, .9, .99): [round(float(s.abs().quantile(x)), 1) for x in qs]  # noqa: E731
    summ = dict(recordings_read=n_cases, without_ecg_or_eeg=n_err, windows=int(len(W)), good_windows=int(len(g)),
                recordings_with_5_good=int(len(P)),
                const_ms_by_mode={md: dict(n=int(len(x)), median=round(float(x.const_ms.median()), 1),
                                           iqr=[round(float(v), 1) for v in x.const_ms.quantile([.25, .75])])
                                  for md, x in P.groupby("mode")},
                within_sd_ms=dict(median=round(float(P.sd_ms.median()), 1), p90=round(float(P.sd_ms.quantile(.9)), 1)),
                slope_ms_h=dict(median_abs=round(float(P.slope_ms_h.abs().median()), 1), p90_abs=round(float(P.slope_ms_h.abs().quantile(.9)), 1)),
                jumps_per_recording_mean=round(float(P.jumps.mean()), 2),
                error_own_const_ms_abs_q50_90_99=q(g.err_own), error_mode_const_ms_abs_q50_90_99=q(g.err_mode.dropna()),
                error_mode_const_by_mode={md: q(x.err_mode.dropna()) for md, x in g.groupby("mode")})
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    ax = axs[0]
    for md, c in (("step", BLUE), ("small", ORANGE), ("none", GREY)):
        x = P[P["mode"] == md].const_ms
        if len(x):
            ax.hist(x.clip(-1500, 0), bins=np.arange(-1500, 1, 10), histtype="step", lw=1.3, color=c, label=f"{md} ({len(x):,})")
    ax.set_xlabel("EEG minus ECG after the stamp model (ms), median per recording"); ax.set_ylabel("recordings")
    ax.legend(frameon=False, fontsize=8, title="SNUADC correction mode", title_fontsize=8)
    ax.set_title("The constant left after the model, by SNUADC mode", loc="left", fontsize=10)
    ax = axs[1]
    for col, c, lab in (("err_mode", BLUE, "minus the mode's constant (leave-one-out)"), ("err_own", ORANGE, "minus the recording's own constant")):
        s = np.sort(g[col].dropna().abs().to_numpy())
        ax.plot(s, np.arange(1, s.size + 1) / s.size, color=c, label=lab)
    ax.set_xscale("log"); ax.set_xlim(1, 2000); ax.set_xlabel("|timing error| per 10-min window (ms)"); ax.set_ylabel("share of windows")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Remaining ECG–EEG timing error", loc="left", fontsize=10)
    ax = axs[2]
    both = (P.ecg_source == "teeth") & (P.eeg_source == "teeth")
    nocorr = P["mode"] == "none"
    bins = np.arange(-600, 601, 10)
    for sel, c, lab in ((both, BLUE, "both clocks fitted from whole teeth"),
                        (~both & ~nocorr, ORANGE, "one clock given a typical rate"),
                        (nocorr, GREY, "SNUADC without corrections (drift assumed 0)")):
        ax.hist(P.slope_ms_h[sel].clip(-600, 600), bins=bins, histtype="step", lw=1.3, color=c, label=f"{lab} ({sel.sum():,})")
    ax.set_xlabel("slope of the EEG–ECG offset over the recording (ms/h)"); ax.set_ylabel("recordings")
    ax.legend(frameon=False, fontsize=7.5); ax.set_title("Drift left after the model", loc="left", fontsize=10)
    for x in axs:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "burst_validation.png", dpi=110)


if __name__ == "__main__":
    main()
