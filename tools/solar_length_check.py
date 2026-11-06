# SPDX-License-Identifier: MIT
"""How the Solar8000 route's SNUADC rate depends on recording length (metrology-atlas studies solar-snuadc-clock and
loop-closure): the evidence for the length cut of the solar edge in tools/clock_edges.py.

    python tools/solar_length_check.py --work W --out W/clock_network/solar_length_check.json

The route's per-recording rate (solar_snuadc_clock/agg/counts_combined.csv, Theil-Sen slope of 30-minute block
delays) is biased toward 0 in shorter recordings. Per length band:

* recordings with whole SNUADC reset intervals (solar_snuadc_clock/reset_rates.csv): the route minus the reset rate
  (median, p25, p75), the median |z| with the route's standard error, and the route's median standard error;
* recordings without reset intervals, by SNUADC mode: the route minus the stamp model's rate (median, p25, p75).

An unbiased route sits at about 0 against the resets. Also the factor on the route's standard error that
clock_edges.py applies (solar_se_factor, from 6 h). Percentiles are numpy's (linear).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from clock_edges import solar_rates, solar_se_factor  # noqa: E402

BANDS = ((0, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 6), (6, 8), (8, 99))


def q(x) -> dict | None:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    return {"n": int(x.size), "median": round(float(np.median(x)), 1), "p25": round(float(np.percentile(x, 25)), 1),
            "p75": round(float(np.percentile(x, 75)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    c = solar_rates(a.work)
    res = pd.read_csv(a.work / "solar_snuadc_clock" / "reset_rates.csv").rename(columns={"ppm": "reset_ppm"})
    c = c.merge(res, on="case", how="left")
    S = {"recordings": int(len(c)), "with_resets": int(c.reset_ppm.notna().sum()), "against_resets": {},
         "against_model": {}}
    for lo, hi in BANDS:
        b = c[(c.hours >= lo) & (c.hours < hi)]
        r = b[b.reset_ppm.notna()]
        if len(r):
            d = r.ppm - r.reset_ppm
            S["against_resets"][f"{lo}-{hi} h"] = {**q(d), "z_abs_median": round(float(np.median(np.abs(d / r.se))), 2),
                                                   "se_median": round(float(r.se.median()), 1)}
        o = b[b.reset_ppm.isna() & b.model_ppm.notna()]
        S["against_model"][f"{lo}-{hi} h"] = {m: q(g.ppm - g.model_ppm) for m, g in o.groupby("mode")}
    for cut in (4.5, 6.0, 8.0):
        r = c[(c.hours >= cut) & c.reset_ppm.notna()]
        S[f"against_resets_from_{cut:g}_h"] = q(r.ppm - r.reset_ppm)
    S["se_factor_from_6_h"] = solar_se_factor(a.work, 6.0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
