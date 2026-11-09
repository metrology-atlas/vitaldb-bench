# SPDX-License-Identifier: MIT
"""EEG-ECG timing error at each level of correction, from the full-release burst check (metrology-atlas study
snuadc-stamp-sawtooth, figure "validation").

    python tools/stamp_error_levels.py --stamps $W/burst_validation_stamps --model $W/burst_validation_m3 \
        --modes $W/results/sync_full/snuadc_mode_label.json --out $W/stamp_error_levels

Inputs are the per-recording burst lags of tools/sa_corrected.py on every recording: on the recorded stamps
(variant ``stamps``) and on the adopted stamp model (``model3``). Good burst windows as in
tools/burst_validation_agg.py (peak r >= 0.3, peak minus the best peak more than 1 s away >= 0.15, lag between
-2,500 and +500 ms, ECG II against EEG1). The levels are compared on the same windows: those good in both variants,
in recordings with at least 5 such windows.

Levels (absolute EEG-minus-ECG error per window, since the true offset of a shared burst is zero):

1. recorded stamps;
2. recorded stamps less one constant (the median over all windows);
3. the model less one constant (likewise);
4. the model less the SNUADC mode's constant (median of the other recordings' constants in that mode, leave one out);
5. the model less the recording's own constant (its median lag).

Writes summary.json and stamp-model-error-by-level.png (median and 90th percentile per level).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

GOOD_PEAK, GOOD_MARGIN, LAG_RANGE_MS = 0.3, 0.15, (-2500.0, 500.0)


def good_windows(folder: str, variant: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(Path(folder) / "part_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            m = r.get(variant)
            if not m or "burst" not in m:
                continue
            for t, lag, pk, sec, _ in np.array(m["burst"].get("ecg~eeg1") or [], float).reshape(-1, 5):
                rows.append((r["case"], t, lag, pk, sec))
    W = pd.DataFrame(rows, columns=["case", "t", "lag_ms", "peak", "second"]).drop_duplicates(["case", "t"])
    return W[(W.peak >= GOOD_PEAK) & (W.peak - W.second >= GOOD_MARGIN) & (W.lag_ms > LAG_RANGE_MS[0]) & (W.lag_ms < LAG_RANGE_MS[1])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamps", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="model3", help="the stamp-model variant in --model (default model3; model4 from 9 Nov 2026)")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = {int(k): v for k, v in json.loads(Path(a.modes).read_text()).items()}
    s = good_windows(a.stamps, "stamps").rename(columns={"lag_ms": "lag_stamps"})[["case", "t", "lag_stamps"]]
    m = good_windows(a.model, a.variant).rename(columns={"lag_ms": "lag_model"})[["case", "t", "lag_model"]]
    w = s.merge(m, on=["case", "t"])
    n = w.groupby("case").size()
    w = w[w.case.isin(n[n >= 5].index)].copy()
    own = w.groupby("case").lag_model.median()
    w["own"] = w.case.map(own)
    w["mode"] = w.case.map(lambda c: modes.get(int(c), "unknown"))
    loo = {}
    for md, g in own.groupby(lambda c: modes.get(int(c), "unknown")):
        for c in g.index:
            loo[c] = float(np.median(g.drop(c))) if len(g) > 1 else np.nan
    w["mode_const"] = w.case.map(loo)
    levels = {
        "Recorded stamps": w.lag_stamps.abs(),
        "Stamps minus one constant": (w.lag_stamps - w.lag_stamps.median()).abs(),
        "Model minus one constant": (w.lag_model - w.lag_model.median()).abs(),
        "Model minus the SNUADC mode's constant": (w.lag_model - w.mode_const).abs(),
        "Model minus the recording's own constant": (w.lag_model - w.own).abs(),
    }
    S = {"windows": int(len(w)), "recordings": int(w.case.nunique()),
         "windows_good_stamps": int(len(s)), "windows_good_model": int(len(m)),
         "constants_ms": {"stamps": round(float(w.lag_stamps.median()), 1), "model": round(float(w.lag_model.median()), 1)},
         "levels": {k: {"median_ms": round(float(v.median()), 1), "p90_ms": round(float(v.quantile(0.9)), 1)} for k, v in levels.items()}}
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = list(levels)
    med = [S["levels"][k]["median_ms"] for k in names]
    p90 = [S["levels"][k]["p90_ms"] for k in names]
    y = np.arange(len(names))[::-1]
    ax.barh(y + 0.18, med, 0.36, color="#1f5fa6", label="median")
    ax.barh(y - 0.18, p90, 0.36, color="#9fbbe0", label="90th percentile")
    for yy, a_, b_ in zip(y, med, p90):
        ax.text(a_, yy + 0.18, f" {a_:.0f}", va="center", fontsize=8)
        ax.text(b_, yy - 0.18, f" {b_:.0f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("EEG−ECG timing error per 10-minute window (ms, log scale)")
    ax.set_title(f"Shared bursts, {S['windows']:,} windows in {S['recordings']:,} recordings (whole release)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "stamp-model-error-by-level.png", dpi=150)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
