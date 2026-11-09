# SPDX-License-Identifier: MIT
"""Numbers for the BIS EEG timing forms in the atlas, from the whole-release burst windows. This machine; the output is
aggregates.

    python tools/bis_numbers.py --stamps $W/burst_validation_stamps --model $W/burst_validation_m4b \
        --model-variant model4 --out bis_numbers.json

Sign: displacement = BIS stamp - the same burst's time on the working reference (SNUADC stamp-model ECG), in s;
negative = the BIS stamp comes before the event on the reference (look-ahead). The lag of the burst cross-correlation
(tools/shared_artifacts.py: EEG delayed by lag -> peak at +lag) is that displacement.
Forms: as stored (raw stamps, --stamps, variant "stamps"); fixed shift (raw stamps plus a constant known in advance:
the median of the recordings' raw constants); after the stamp model with the mode's constant (default; --model,
its windows.parquet from tools/burst_validation_agg.py); after the stamp model with the recording's own burst constant
(leave-one-window-out). Also the correlation length of the within-recording error after the stamp model: an
exponential fitted to the pooled autocorrelation at 5-60 min.

Moved here on 9 Nov 2026 from the work folder (reframe_stage3/bis_numbers.py and the BIS part of corr_lengths.py),
unchanged in method; --model makes the stamp-model run a choice (model3 until 9 Nov, model4 with setup-rate
fallbacks after).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

GOOD_PEAK, GOOD_MARGIN, LAG = 0.3, 0.15, (-2500.0, 500.0)
rsd = lambda x: float(1.4826 * np.median(np.abs(x)))            # robust SD about zero (bias 0 after correction)
rsd_c = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))
q = lambda x, p=(2.5, 97.5): [round(float(v), 4) for v in np.percentile(x, p)]


def windows(path, variant):
    rows = []
    for f in glob.glob(str(Path(path) / "part_*.jsonl")):
        for l in open(f):
            r = json.loads(l); m = r.get(variant)
            if not m or "burst" not in m:
                continue
            for t, lag, pk, sec, act in np.array(m["burst"].get("ecg~eeg1") or [], float).reshape(-1, 5):
                rows.append((r["case"], t, lag / 1000, pk, sec))
    W = pd.DataFrame(rows, columns=["case", "t", "lag", "peak", "second"])
    g = W[(W.peak >= GOOD_PEAK) & (W.peak - W.second >= GOOD_MARGIN) & (W.lag * 1000 > LAG[0]) & (W.lag * 1000 < LAG[1])]
    n = g.groupby("case").size()
    return g[g.case.isin(n[n >= 5].index)].copy()


def acf(df, maxlag_s=3600, step=300):
    out = {}
    for c, g in df.groupby("case"):
        t, e = g.t.to_numpy(), g.e.to_numpy(); e = e - np.median(e)
        if e.size < 6 or np.std(e) == 0:
            continue
        e = e / (1.4826 * np.median(np.abs(e)) + 1e-9); e = np.clip(e, -4, 4)
        dt = np.abs(t[:, None] - t[None, :]); k = np.round(dt / step).astype(int)
        prod = e[:, None] * e[None, :]
        for lag in range(0, maxlag_s // step + 1):
            m = np.triu(k == lag, 1) if lag else np.eye(e.size, dtype=bool)
            if m.any():
                out.setdefault(lag, []).extend(prod[m].tolist())
    lags = sorted(out); r = np.array([np.mean(out[l]) for l in lags]); r = r / r[0]
    x = np.array(lags) * step
    keep = x > 0
    L = curve_fit(lambda d, L: np.exp(-d / L), x[keep], np.clip(r[keep], 1e-3, 1), p0=[1800])[0][0]
    return {"lag_s": x.tolist(), "corr": np.round(r, 3).tolist(), "length_s": round(float(L)),
            "pairs_at_lag1": len(out.get(1, []))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stamps", required=True, help="sa_corrected output with variant 'stamps' (raw stamps)")
    ap.add_argument("--model", required=True, help="burst_validation_agg output folder of the stamp-model run")
    ap.add_argument("--model-variant", default="model3", help="recorded in the output only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"model": str(a.model), "model_variant": a.model_variant}
    R = windows(a.stamps, "stamps")
    const = R.groupby("case").lag.median()
    med_all = float(const.median())
    R["loo_const"] = med_all          # leaving one recording out of ~2,700 moves the median by < 1 ms
    R["err_fixed"] = R.lag - R.loo_const
    R["err_within"] = R.lag - R.case.map(const)
    out["raw"] = {"recordings": int(const.size), "windows": int(len(R)),
                  "bias_per_recording_s": {"median": round(med_all, 4), "p5": round(float(const.quantile(.05)), 4),
                                           "p95": round(float(const.quantile(.95)), 4)},
                  "within_recording_rsd_s": round(rsd(R.err_within), 4),
                  "window_displacement_p2.5_97.5": q(R.lag)}
    out["fixed_shift"] = {"shift_s": round(med_all, 4), "u_headline_rsd_s": round(rsd(R.err_fixed), 4),
                          "interval95_s": q(R.err_fixed), "between_recordings_rsd_s": round(rsd_c(const), 4),
                          "within_recording_rsd_s": round(rsd(R.err_within), 4)}
    M = pd.read_parquet(Path(a.model) / "windows.parquet")
    M["lag_s"], M["err_mode_s"] = M.lag_ms / 1000, M.err_mode / 1000
    own_loo = []
    for c, g in M.groupby("case"):
        y = g.lag_s.to_numpy()
        for i in range(y.size):
            own_loo.append(y[i] - np.median(np.delete(y, i)))
    own_loo = np.array(own_loo)
    cm = M.groupby("case").err_mode_s.median()
    out["sync_mode_constant"] = {"recordings": int(M.case.nunique()), "windows": int(len(M)),
        "u_headline_rsd_s": round(rsd(M.err_mode_s.dropna()), 4), "interval95_s": q(M.err_mode_s.dropna()),
        "median_abs_s": round(float(M.err_mode_s.abs().median()), 4),
        "common_to_recording_rsd_s": round(rsd(cm.dropna()), 4), "within_recording_rsd_s": round(rsd(M.err_own / 1000), 4),
        "by_mode_rsd_s": {md: round(rsd(g.err_mode_s.dropna()), 4) for md, g in M.groupby("mode")},
        "mode_constants_s": {md: round(float(g.groupby("case").lag_s.median().median()), 4) for md, g in M.groupby("mode")}}
    out["sync_own_bursts"] = {"u_headline_rsd_s": round(rsd(own_loo), 4), "interval95_s": q(own_loo),
                              "median_abs_s": round(float(np.median(np.abs(own_loo))), 4)}
    out["sync_within_correlation"] = acf(pd.DataFrame({"case": M.case, "t": M.t, "e": M.err_own / 1000}))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "sync_within_correlation"}, indent=1))
    print("correlation length", out["sync_within_correlation"]["length_s"], "s")


if __name__ == "__main__":
    main()
