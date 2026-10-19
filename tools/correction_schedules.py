# SPDX-License-Identifier: MIT
"""Device clock-correction schedules from the step tables, for every recording. This machine only.

    python tools/correction_schedules.py --survey C:/vitalDB/work/signal_survey --drift C:/vitalDB/work/results/sync_full/full.drift.csv

Inputs: steps_agg/chunk_*.parquet (tools/step_classify_agg.py) and tables/segs_*.parquet (signal_events_collect.py:
each track's segments, whose first start is the track's start). Writes continuity/: bis_drift_per_recording.csv,
snuadc_reset_events.csv, snuadc_drift_per_recording.csv, primus_corrections.csv, primus_awp_pauses.csv,
summary.json and correction_schedules.png.

- BIS: corrections are forward steps of 10 ms to 1 s not shared with another device; drift = step / interval for
  corrections 8192 +- 20 s after the previous one.
- SNUADC: a reset is a run of backward steps summing to -0.45..-0.52 s (step_classify_agg class 'reset');
  drift = size / interval between resets, and for the first reset size / time since the track started.
  Recordings with resets under 600 s apart (a few with repeated -0.5 s steps throughout) are left out.
- Primus: corrections are AWP steps of 10 ms to 0.2 s in either direction; pauses are AWP forward steps of
  19-23.5 s not shared with another device.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BLUE, ORANGE, GREY = "#1f5fa6", "#e08a1e", "#7f7f7f"


def load(s):
    d = pd.concat([pd.read_parquet(f, columns=["case", "track", "k", "t", "step", "cls", "other_devices"])
                   for f in glob.glob(str(s / "steps_agg" / "chunk_*.parquet"))])
    d = d[d.case.notna() & d.t.notna()]
    d["case"] = d.case.astype(int)
    segs = pd.concat([pd.read_parquet(f) for f in glob.glob(str(s / "tables" / "segs_*.parquet"))])
    t0 = {tr: segs[segs.track == tr].groupby("case").t.min() for tr in ("BIS__EEG1_WAV", "SNUADC__ECG_II", "Primus__AWP")}
    return d, t0


def bis(d, t0):
    b = d[(d.track == "BIS/EEG1_WAV") & (d.step > 0.01) & (d.step < 1.0) & (d.other_devices == 0)].sort_values(["case", "t"]).copy()
    b["prev"] = b.groupby("case").t.shift()
    b["dt"] = (b.t - b.prev).fillna(b.t - b.case.map(t0["BIS__EEG1_WAV"]))
    b["first"] = b.prev.isna()
    reg = b[(b.dt - 8192).abs() < 20].copy()
    reg["ppm"] = reg.step / reg.dt * 1e6
    pc = reg.groupby("case").agg(ppm=("ppm", "median"), n=("ppm", "size"), step=("step", "median"))
    return b, pc


def snuadc(d, t0):
    x = d[(d.track == "SNUADC/ECG_II") & (d.cls == "reset")].sort_values(["case", "k"])
    run = (x.case.ne(x.case.shift()) | x.k.diff().ne(1)).cumsum()
    ev = x.groupby(run).agg(case=("case", "first"), t=("t", "first"), size=("step", "sum"), nrec=("step", "size"))
    ev = ev.reset_index(drop=True).sort_values(["case", "t"])
    ev["dt"] = ev.groupby("case").t.diff()
    bad = ev[ev.dt < 600].case.unique()
    ev = ev[~ev.case.isin(bad)].copy()
    ev["dt"] = ev.groupby("case").t.diff()
    ev["since_start"] = ev.t - ev.case.map(t0["SNUADC__ECG_II"])
    later = ev[ev.dt.notna()].copy(); later["ppm"] = later["size"] / later.dt * 1e6
    two = later.groupby("case").agg(ppm=("ppm", "median"), n=("ppm", "size"), period_s=("dt", "median"))
    first = ev[ev.dt.isna()].copy(); first["ppm_first"] = first["size"] / first.since_start * 1e6
    pc = first.set_index("case")[["ppm_first", "since_start", "size"]].join(two, how="left")
    pc["drift_ppm"] = pc.ppm.fillna(pc.ppm_first)
    pc["basis"] = np.where(pc.ppm.notna(), "intervals between resets", "first reset from the start")
    return ev, pc, bad


def primus(d, t0):
    a = d[d.track == "Primus/AWP"]
    c = a[(a.step.abs() > 0.01) & (a.step.abs() < 0.2) & (a.cls != "duplicate")].sort_values(["case", "t"]).copy()
    c["dt"] = c.groupby("case").t.diff()
    g = a[(a.step > 19) & (a.step < 23.5) & (a.other_devices == 0)].sort_values(["case", "t"]).copy()
    g["dt"] = g.groupby("case").t.diff()
    near, sign = [], []
    cc = {k: v.t.to_numpy() for k, v in c.groupby("case")}
    for cs, t in zip(g.case, g.t):
        arr = cc.get(cs)
        if arr is None or arr.size == 0:
            near.append(np.nan); sign.append(np.nan); continue
        j = np.argmin(np.abs(arr - t)); near.append(abs(arr[j] - t)); sign.append(np.sign(arr[j] - t))
    g["to_correction_s"] = near; g["correction_after"] = sign
    co = d[(d.track == "Primus/CO2") & (d.step > 19) & (d.step < 23.5)][["case", "t"]]
    m = g.merge(co, on="case", suffixes=("", "_co")); m = m[(m.t - m.t_co).abs() < 2]
    g["co2_gap_too"] = g.set_index(["case", "t"]).index.isin(m.set_index(["case", "t"]).index)
    return c, g


def figure(b, bpc, ev, spc, c, g, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    ax = axs[0, 0]
    later = b[~b["first"]].dt
    ax.hist(later.clip(0, 20000), bins=np.arange(0, 20001, 100), color=BLUE)
    ax.hist(b[b["first"]].dt.clip(0, 20000), bins=np.arange(0, 20001, 100), histtype="step", color=ORANGE, lw=1.2)
    ax.set_yscale("log"); ax.set_xlabel("s since the previous correction (blue) or since the BIS started (orange)")
    ax.set_title("BIS: corrections every 8,192 s; the first at any time", loc="left", fontsize=10)
    ax = axs[1, 0]
    ax.hist(bpc.ppm.clip(-10, 90), bins=np.arange(-10, 91, 1), color=BLUE)
    ax.set_xlabel("BIS drift (ppm), one value per recording"); ax.set_ylabel("recordings")
    ax.set_title(f"BIS drift in {len(bpc):,} recordings: median {bpc.ppm.median():.1f} ppm", loc="left", fontsize=10)
    ax = axs[0, 1]
    two = spc[spc.ppm.notna()]
    pred = (two["size"] / two.ppm) * 1e6 / 3600
    ax.scatter(pred, two.since_start / 3600, s=4, color=BLUE, alpha=0.5)
    lim = [0, 8]; ax.plot(lim, lim, color=GREY, lw=1, ls=":"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("0.49 s ÷ drift from later resets (h)"); ax.set_ylabel("first reset after SNUADC start (h)")
    ax.set_title("SNUADC: first reset against the time 0.49 s of drift takes", loc="left", fontsize=10)
    ax = axs[1, 1]
    one = spc[spc.ppm.isna()]
    ax.hist(two.ppm.clip(-120, 0), bins=np.arange(-120, 1, 1), color=BLUE, label=f"from intervals between resets ({len(two):,})")
    ax.hist(one.ppm_first.clip(-120, 0), bins=np.arange(-120, 1, 1), histtype="step", color=ORANGE, lw=1.2,
            label=f"single reset, timed from the start ({len(one):,}; approximate)")
    ax.set_xlabel("SNUADC step-mode drift (ppm, against the recorder PC)"); ax.set_ylabel("recordings"); ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"SNUADC drift from reset intervals: median {two.ppm.median():.1f} ppm", loc="left", fontsize=10)
    ax = axs[0, 2]
    ax.hist(c.dt.dropna().clip(0, 20000), bins=np.arange(0, 20001, 100), color=BLUE, label="corrections (10–200 ms steps)")
    ax.hist(g.dt.dropna().clip(0, 20000), bins=np.arange(0, 20001, 100), histtype="step", color=ORANGE, lw=1.2, label="AWP pauses (19–23.5 s)")
    ax.set_yscale("log"); ax.set_xlabel("s since the previous event of the same kind"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Primus: corrections and AWP pauses both every 7,230 s", loc="left", fontsize=10)
    ax = axs[1, 2]
    x = (g.to_correction_s * g.correction_after).dropna()
    within = (x.abs() < 60).mean()
    ax.hist(x[x.abs() < 60], bins=np.arange(-60, 61, 1), color=BLUE)
    ax.set_xlabel("nearest correction minus pause (s), within 60 s")
    ax.set_ylabel("AWP pauses")
    ax.set_title(f"A correction follows {within:.0%} of AWP pauses by about 21 s", loc="left", fontsize=10)
    for a_ in axs.flat:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "correction_schedules.png", dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--survey", required=True); p.add_argument("--drift", default=None)
    a = p.parse_args()
    s = Path(a.survey); out = s / "continuity"; out.mkdir(exist_ok=True)
    d, t0 = load(s)
    b, bpc = bis(d, t0); ev, spc, bad = snuadc(d, t0); c, g = primus(d, t0)
    bpc.to_csv(out / "bis_drift_per_recording.csv"); ev.to_csv(out / "snuadc_reset_events.csv", index=False)
    spc.to_csv(out / "snuadc_drift_per_recording.csv"); c.to_csv(out / "primus_corrections.csv", index=False)
    g.to_csv(out / "primus_awp_pauses.csv", index=False)
    two = spc[spc.ppm.notna()]
    summ = {
        "bis": {"recordings_with_corrections": int(b.case.nunique()), "recordings_with_drift": int(len(bpc)),
                "later_at_8192_share": float(((b[~b["first"]].dt - 8192).abs() < 20).mean()),
                "first_at_8192_share": float(((b[b["first"]].dt - 8192).abs() < 20).mean()),
                "drift_ppm_q": bpc.ppm.quantile([.05, .25, .5, .75, .95]).round(1).tolist(),
                "step_s_median": float(bpc.step.median())},
        "snuadc": {"recordings_with_reset": int(len(spc)), "excluded": [int(x) for x in bad],
                   "reset_events": int(len(ev)), "two_or_more": int(len(two)),
                   "interval_h_q": (ev.dt.dropna() / 3600).quantile([.05, .25, .5, .75, .95]).round(2).tolist(),
                   "drift_ppm_q_intervals": two.ppm.quantile([.05, .25, .5, .75, .95]).round(1).tolist(),
                   "drift_ppm_q_all": spc.drift_ppm.quantile([.05, .25, .5, .75, .95]).round(1).tolist(),
                   "first_over_predicted_q": (two.since_start / (two["size"] / two.ppm * 1e6))
                   .quantile([.05, .25, .5, .75, .95]).round(2).tolist(),
                   "records_per_reset": {str(k): int(v) for k, v in ev.nrec.value_counts().head(4).items()}},
        "primus": {"corrections": int(len(c)), "recordings_with_corrections": int(c.case.nunique()),
                   "correction_ms_top": {str(k): int(v) for k, v in (c.step * 1000).round(0).value_counts().head(8).items()},
                   "correction_interval_top10s": {str(k): int(v) for k, v in (c.dt / 10).round().mul(10).value_counts().head(4).items()},
                   "pauses": int(len(g)), "recordings_with_pauses": int(g.case.nunique()),
                   "pause_interval_top10s": {str(k): int(v) for k, v in (g.dt / 10).round().mul(10).value_counts().head(4).items()},
                   "pause_within_60s_of_correction": float((g.to_correction_s < 60).mean()),
                   "pause_correction_offset_s_median": float((g.to_correction_s * g.correction_after)[g.to_correction_s < 60].median()),
                   "co2_gap_too_share": float(g.co2_gap_too.mean())},
    }
    if a.drift:
        o = pd.read_csv(a.drift)
        for dev, pc, col in (("BIS", bpc, "ppm"), ("SNUADC", two, "ppm")):
            j = pc[[col]].join(o[o.device_type == dev].set_index("case").drift_ppm.rename("earlier"), how="inner")
            dd = j[col] - j.earlier
            summ[dev.lower()]["vs_earlier"] = {"n": int(len(j)), "median_diff_ppm": round(float(dd.median()), 3),
                                               "iqr": dd.quantile([.25, .75]).round(3).tolist()}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    figure(b, bpc, ev, spc, c, g, out)


if __name__ == "__main__":
    main()
