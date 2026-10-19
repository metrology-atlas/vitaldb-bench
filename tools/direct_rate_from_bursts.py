# SPDX-License-Identifier: MIT
"""Direct BIS-against-SNUADC rate per recording, from the stamp-model burst check (no waveforms read).

    python tools/direct_rate_from_bursts.py <burst_validation dir> [--variant model3] [--out DIR]
        [--snuadc-rates snuadc_reset_drift_per_recording.csv] [--bis-rates bis_drift_per_recording.csv]

The burst check (tools/sa_corrected.py) puts the ECG and EEG on the recorder PC's timeline with the stamp model and
records the EEG-minus-ECG lag of every 10-minute burst window. Each device's model rate r is its drift against the
PC, defined as how much the PC gains on the device (BIS about +25 ppm, SNUADC step mode about -63 ppm). On the raw
stamps between corrections the EEG-minus-ECG lag therefore ramps at -(r_bis - r_snuadc); the slope still left on
the model timeline adds to it. So, per recording,

    direct rate (ppm) = -(r_eeg1 - r_ecg) + residual slope of the good burst lags (ms/h / 3.6)

which is the slope the 191-case pilot measured directly on the raw stamps (-72 ppm median). It is a check on the
stamp model only where both rates come from the stamps' own whole teeth (source "teeth"); elsewhere a rate was
given (port or unit typical) and the residual slope carries the recording's own departure from it. The SNUADC rate
includes the -0.954 ppm stamp arithmetic, so the rate between the two crystals' sample counts is the direct rate
minus 0.954 ppm (reported separately).

Good windows as in burst_validation_agg: peak r >= 0.3, peak minus the best peak more than 1 s away >= 0.15, lag
between -2,500 and +500 ms, ECG II ~ EEG1; a recording needs 5 good windows spanning at least 1 hour.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

GOOD_PEAK, GOOD_MARGIN, LAG_RANGE_MS = 0.3, 0.15, (-2500.0, 500.0)
ARITH_PPM = -0.954


def load(d: str, variant: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(Path(d) / "part_*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            m = r.get(variant) or {}
            cl = m.get("clocks") or {}
            if "ecg" not in cl or "eeg1" not in cl:
                continue
            w = np.array([x for x in (m.get("burst") or {}).get("ecg~eeg1", [])
                          if x[2] >= GOOD_PEAK and x[2] - x[3] >= GOOD_MARGIN and LAG_RANGE_MS[0] < x[1] < LAG_RANGE_MS[1]])
            if len(w) < 5 or np.ptp(w[:, 0]) < 3600:
                continue
            s = theilslopes(w[:, 1], w[:, 0] / 3600)
            rows.append(dict(case=int(r["case"]), n=len(w), span_h=float(np.ptp(w[:, 0]) / 3600),
                             r_ecg=cl["ecg"]["rate_ppm"], src_ecg=cl["ecg"]["source"],
                             r_eeg=cl["eeg1"]["rate_ppm"], src_eeg=cl["eeg1"]["source"],
                             resid_ms_h=float(s[0]), resid_lo=float(s[2]), resid_hi=float(s[3])))
    P = pd.DataFrame(rows)
    P["model_ppm"] = -(P.r_eeg - P.r_ecg)
    P["direct_ppm"] = P.model_ppm + P.resid_ms_h / 3.6
    P["direct_counts_ppm"] = P.direct_ppm - ARITH_PPM
    P["direct_ci_ppm"] = (P.resid_hi - P.resid_lo) / 3.6
    return P


def q(s, ps=(10, 25, 50, 75, 90)):
    s = pd.Series(s).dropna()
    return [round(float(np.percentile(s, p)), 1) for p in ps] + [int(s.size)] if s.size else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--variant", default="model3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--snuadc-rates", default=None, help="per-recording SNUADC drift from reset intervals (case, ppm)")
    ap.add_argument("--bis-rates", default=None, help="per-recording BIS drift from correction steps (case, ppm)")
    a = ap.parse_args()
    P = load(a.dir, a.variant)
    both = (P.src_ecg == "teeth") & (P.src_eeg == "teeth")
    out = {"recordings": int(len(P)), "both_teeth": int(both.sum()),
           "direct_ppm_p10_25_50_75_90_n": {"all": q(P.direct_ppm), "both_teeth": q(P.direct_ppm[both]),
                                            "ecg_teeth_eeg_any": q(P.direct_ppm[P.src_ecg == "teeth"])},
           "model_part_ppm_both_teeth": q(P.model_ppm[both]),
           "resid_ppm_both_teeth": q(P.resid_ms_h[both] / 3.6),
           "slope_ci_width_ppm_median": round(float(P.direct_ci_ppm.median()), 1)}
    if a.snuadc_rates and a.bis_rates:
        s = pd.read_csv(a.snuadc_rates)[["case", "ppm"]].rename(columns={"ppm": "snu_pc"})
        b = pd.read_csv(a.bis_rates)[["case", "ppm"]].rename(columns={"ppm": "bis_pc"})
        C = P.merge(s, on="case").merge(b, on="case")
        C["closure_ppm"] = C.direct_ppm - (-(C.bis_pc - C.snu_pc))
        out["closure_vs_correction_steps"] = {"recordings": int(len(C)), "direct": q(C.direct_ppm),
                                              "from_pc_rates": q(-(C.bis_pc - C.snu_pc)), "closure": q(C.closure_ppm),
                                              "abs_closure_median": round(float(C.closure_ppm.abs().median()), 1)}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        P.to_csv(Path(a.out) / "direct_rate_per_recording.csv", index=False)
        (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
