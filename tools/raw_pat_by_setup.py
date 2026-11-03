# SPDX-License-Identifier: MIT
"""Raw ECG-to-PPG foot delay per recording, by recording setup, on the whole release (metrology-atlas study
ppg-offset-reference, figure raw-pat-by-setup).

    python tools/raw_pat_by_setup.py --cases $W/pat_census/aggf/cases.csv \
        --modes $W/results/sync_full/snuadc_mode_label.json --out $W/raw_pat_by_setup [--min-n 20]

Input: the per-recording medians of tools/pat_offset_census.py (``agg``; stored delays, no sawtooth fit and no
offset correction), R peak to PLETH foot (intersecting tangent) and foot to steepest upstroke. A setup is the
SNUADC serial port together with the recording's SNUADC correction mode (snuadc_mode_label.json: step, small,
mixed, or none, drawn as no-corr; as the benchmark figure this replaces). Setups with fewer than --min-n recordings are pooled as "other".

Recordings with a foot delay under 400 ms are counted, not drawn: there the beats were paired one beat out (as in
the benchmark figure). Writes summary.json and raw-pat-by-setup.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

YLIM = (540.0, 800.0)
ONE_BEAT_OUT_MS = 400.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.cases)
    modes = json.loads(Path(a.modes).read_text())
    d["mode"] = d["case"].astype(str).map(modes).fillna("unknown").replace({"none": "no-corr"})
    d = d[d["ppg_foot_ms"].notna() & d["snuadc_port"].notna()].copy()
    low = d["ppg_foot_ms"] < ONE_BEAT_OUT_MS
    n_low = int(low.sum())
    d = d[~low]
    d["setup"] = d["snuadc_port"].astype(str) + " " + d["mode"]
    n = d["setup"].value_counts()
    big = n[n >= a.min_n].index
    d["group"] = np.where(d["setup"].isin(big), d["setup"], "other")
    order = list(n[big].index) + (["other"] if (d["group"] == "other").any() else [])
    med = d.groupby("group")["ppg_foot_ms"].median()
    by_mode = d.groupby("mode")["ppg_foot_ms"].agg(["size", "median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    by_mode.columns = ["n", "median", "q25", "q75"]
    f2u = d["ppg_foot_to_up_ms"].dropna()
    outside = int(((d["ppg_foot_ms"] < YLIM[0]) | (d["ppg_foot_ms"] > YLIM[1])).sum())
    S = {"recordings": int(len(d)), "one_beat_out_not_drawn": n_low, "outside_axis": outside,
         "setups": int(len(n)), "setups_drawn": int(len(big)), "other_recordings": int((d["group"] == "other").sum()),
         "foot_ms_median_iqr": [round(float(x), 1) for x in d["ppg_foot_ms"].quantile([0.5, 0.25, 0.75])],
         "setup_medians_ms_range": [round(float(med.drop("other", errors="ignore").min()), 1),
                                    round(float(med.drop("other", errors="ignore").max()), 1)],
         "by_mode": {k: {kk: round(float(vv), 1) for kk, vv in v.items()} for k, v in by_mode.iterrows()},
         "foot_to_upstroke_ms_median_p10_p90": [round(float(x), 1) for x in f2u.quantile([0.5, 0.1, 0.9])],
         "by_setup": {g: {"n": int((d["group"] == g).sum()), "median_ms": round(float(med[g]), 1)} for g in order}}
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: v for k, v in S.items() if k != "by_setup"}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 4)
    ax = fig.add_subplot(gs[0, :3])
    rng = np.random.default_rng(0)
    data = [d.loc[d["group"] == g, "ppg_foot_ms"].to_numpy() for g in order]
    ax.boxplot(data, positions=range(len(order)), widths=0.6, showfliers=False,
               medianprops=dict(color="#1a2a4a", lw=2), boxprops=dict(color="0.5"), whiskerprops=dict(color="0.5"),
               capprops=dict(color="0.5"))
    for i, v in enumerate(data):
        vv = np.clip(v, *YLIM)
        ax.scatter(i + rng.uniform(-0.22, 0.22, v.size), vv, s=4, alpha=0.35, color="#3b6fb6", lw=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{g} (n={len(v)})" for g, v in zip(order, data)], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(*YLIM)
    ax.set_ylabel("median raw PAT per recording, R peak to PPG foot (ms)\nno offset or sawtooth correction applied")
    ax.set_title(f"Raw ECG→PPG foot delay by recording setup (whole release, {len(d):,} recordings; "
                 f"setups with {a.min_n} or more)", fontsize=10)
    ax.text(0.01, 0.02, f"{n_low} recordings under {ONE_BEAT_OUT_MS:.0f} ms not drawn (beats paired one beat out); "
            f"{outside} outside the axis drawn at its edge", transform=ax.transAxes, fontsize=7, color="0.4")
    ax.grid(axis="y", color="0.9")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax2 = fig.add_subplot(gs[0, 3])
    ax2.hist(np.clip(f2u, 0, 125), bins=np.arange(0, 127.5, 2.5), color="#3b6fb6")
    ax2.set_xlabel("upstroke minus foot, per recording (ms)")
    ax2.set_ylabel("recordings")
    ax2.set_title("Foot vs steepest upstroke", fontsize=10)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "raw-pat-by-setup.png", dpi=110)


if __name__ == "__main__":
    main()
