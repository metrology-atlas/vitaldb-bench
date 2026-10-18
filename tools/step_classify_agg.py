# SPDX-License-Identifier: MIT
"""Aggregate tools/step_classify.py output: one table, step classes per signal, and figures. This machine only.

    python tools/step_classify_agg.py collect --inp DIR --out AGG [--time-budget S]   # resumable, chunks of 500 cases
    python tools/step_classify_agg.py summary --out AGG                                 # classes.csv, by_case.csv
    python tools/step_classify_agg.py figure --out AGG                                  # step_classes.png

Classes (first match wins):
  backward  duplicate     record k+1 identical to record k (same stamp: a step of -1 message)
            reset         SNUADC step-mode reset: consecutive backward steps summing to -0.45..-0.52 s (one step, or the
                          five records written out of order around it)
            interleave    record k+1 continues record k-1 better than record k, in a track with 50 or more such steps
            backward      any other backward step
  forward   shared        a step over 1 s with a forward step over 1 s on another device within 1 s (recorder pause)
            whole         at least one message and a whole number of messages (within half a sample)
            sub_message   shorter than one message
            fractional    longer than one message, not a whole number
Backward steps were capped at 3,000 per track in the scan, so backward counts are lower bounds; recordings are exact.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["duplicate", "reset", "interleave", "backward", "shared", "whole", "sub_message", "fractional"]


def classify(d: pd.DataFrame) -> pd.Series:
    """Step classes; uses the whole case (reset clusters, interleave counts), so pass whole cases."""
    tol = 0.5 / (d["msg"] * d["fs"])
    back = d.step < 0
    ident = d.identical.fillna(False).astype(bool)
    jk = d.joins_k_1.fillna(False).astype(bool)
    snu = d.device.eq("SNUADC")
    # SNUADC step-mode resets: a run of consecutive backward steps (one -0.49 s step, or the five records written
    # out of order around it, intervals 0.12 / 0.08 s) that together step the stamps back by 0.45-0.52 s
    key = d.case.astype(str) + "|" + d.track
    sb = d.loc[back & snu, ["k", "step"]].assign(key=key[back & snu])
    sb = sb.sort_values(["key", "k"])
    run = (sb.key.ne(sb.key.shift()) | sb.k.diff().ne(1)).cumsum()
    tot = sb.groupby(run).step.transform("sum")
    reset = pd.Series(False, index=d.index)
    reset.loc[sb.index[tot.between(-0.52, -0.45).to_numpy()]] = True
    njk = (back & jk & ~ident & ~reset).groupby(key).transform("sum")
    c = np.select(
        [back & ident,
         reset,
         back & jk & (njk >= 50),
         back,
         (d.step > 1) & (d.other_devices > 0),
         (d.steps_msg >= 1 - tol) & (d.remainder.abs().fillna(0) <= tol),
         d.steps_msg < 1],
        ORDER[:7], "fractional")
    return pd.Series(c, index=d.index)


def collect(a):
    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(inp.glob("*.parquet"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    files = [f for f in files if f.stem.isdigit()]
    t0 = time.time()
    for i in range(0, len(files), 500):
        dst = out / f"chunk_{i // 500:02d}.parquet"
        if dst.exists() and not a.force:
            continue
        d = pd.concat([pd.read_parquet(f) for f in files[i:i + 500]], ignore_index=True)
        d["cls"] = classify(d)
        d.drop(columns=["devices_present"]).to_parquet(dst.with_suffix(".tmp"))
        dst.with_suffix(".tmp").rename(dst)
        print(dst.name, len(d), flush=True)
        if a.time_budget and time.time() - t0 > a.time_budget:
            print("time budget reached; rerun to continue")
            return


def load(out):
    return pd.concat([pd.read_parquet(f) for f in sorted(Path(out).glob("chunk_*.parquet"))], ignore_index=True)


def summary(a):
    d = load(a.out)
    d["track"] = d.track.str.replace("#2", "", regex=False)
    n = d.groupby(["track", "cls"]).size().unstack(fill_value=0).reindex(columns=ORDER, fill_value=0)
    rec = d.groupby(["track", "cls"]).case.nunique().unstack(fill_value=0).reindex(columns=ORDER, fill_value=0)
    lost = d[d.step > 0].groupby(["track", "cls"]).step.sum().unstack(fill_value=0).reindex(columns=ORDER[4:], fill_value=0)
    jz = d[d.step > 0].groupby(["track", "cls"]).join_z.median().unstack().reindex(columns=ORDER[4:])
    lowz = d[d.step > 0].assign(low=lambda x: x.join_z < 3).groupby(["track", "cls"]).low.mean().unstack().reindex(columns=ORDER[4:])
    tab = pd.concat({"steps": n, "recordings": rec, "seconds": lost.round(0), "join_z_median": jz.round(2),
                     "share_join_z_lt3": lowz.round(3)}, axis=1)
    tab.to_csv(Path(a.out) / "classes.csv")
    bc = d.groupby(["case", "track", "cls"]).size().unstack(fill_value=0).reindex(columns=ORDER, fill_value=0)
    bc.to_csv(Path(a.out) / "by_case.csv")
    pd.set_option("display.width", 250, "display.max_columns", 50)
    main = [t for t in n.index if n.loc[t].sum() > 500]
    print(n.loc[main].to_string()); print(); print(rec.loc[main].to_string()); print()
    print(lost.loc[main].round(0).to_string()); print(); print(jz.loc[main].round(2).to_string()); print()
    print(lowz.loc[main].round(3).to_string())


def figure(a):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = load(a.out)
    d["track"] = d.track.str.replace("#2", "", regex=False)
    nul = json.loads((Path(a.out) / "join_null.json").read_text()) if (Path(a.out) / "join_null.json").exists() else {}
    tracks = ["SNUADC/ECG_II", "SNUADC/PLETH", "SNUADC/ART", "Primus/CO2", "Primus/AWP", "BIS/EEG1_WAV"]
    cls = ["shared", "whole", "sub_message", "fractional"]
    col = {"shared": "#7f7f7f", "whole": "#1f5fa6", "sub_message": "#e08a1e", "fractional": "#2a9d5c"}
    fig, axs = plt.subplots(len(tracks), 2, figsize=(13, 2.5 * len(tracks)), gridspec_kw={"width_ratios": [2.2, 1]})
    bins = np.logspace(-4, 4.5, 171)
    for r, t in enumerate(tracks):
        x = d[(d.track == t) & (d.step > 0)]
        ax = axs[r, 0]
        for c in cls:
            y = x[x.cls == c]
            if len(y):
                ax.hist(y.step, bins=bins, histtype="step", lw=1.3, color=col[c], label=f"{c} ({len(y):,})")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.spines[["top", "right"]].set_visible(False)
        m = x.msg.median() if len(x) else np.nan
        ax.axvline(m, color="k", ls=":", lw=0.8)
        ax.set_title(f"{t}: forward steps by class (dotted: one message, {m:.3g} s)", loc="left", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax = axs[r, 1]
        sh = [(x[x.cls == c].join_z.dropna() < 3).mean() if (x.cls == c).any() else np.nan for c in cls]
        ax.barh(range(len(cls)), sh, color=[col[c] for c in cls])
        ax.set_yticks(range(len(cls)), cls, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        if t in nul:
            ax.axvline(nul[t]["ord_lt3"], color="k", lw=1, label="no step (control)")
            ax.axvline(nul[t]["lost1_lt3"], color="k", ls="--", lw=1, label="one message removed (control)")
        if r == 0:
            ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.set_title("share of steps the waveform runs across (join_z < 3)", loc="left", fontsize=9)
    axs[-1, 0].set_xlabel("step (s)"); axs[-1, 1].set_xlabel("share with join_z < 3")
    fig.tight_layout()
    fig.savefig(Path(a.out) / "step_classes.png", dpi=110)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect"); c.add_argument("--inp", required=True); c.add_argument("--out", required=True)
    c.add_argument("--time-budget", type=float, default=0)
    c.add_argument("--force", action="store_true", help="rebuild chunks that exist")
    for k in ("summary", "figure"):
        s = sp.add_parser(k); s.add_argument("--out", required=True)
    a = p.parse_args()
    {"collect": collect, "summary": summary, "figure": figure}[a.cmd](a)


if __name__ == "__main__":
    main()
