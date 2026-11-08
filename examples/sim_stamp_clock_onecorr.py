# SPDX-License-Identifier: MIT
"""Synthetic check: does physiosync.stamp_clock recover a known SNUADC rate as well in recordings with one
correction event as in those with two or more? (Raised 9 Nov 2026 over atlas parameters pc-grid-rate-single and
pc-grid-rate-multi, which split the PC-against-grid rate by the SNUADC's number of corrections.)

    python examples/sim_stamp_clock_onecorr.py OUT.json [--n 3000] [--seed 1]

Model (stamp_clock's own): records of 100 samples at 500 Hz are stamped by sample count; the PC's time runs away
from the count at RATE (the SNUADC fast: negative), and each time the accumulated difference reaches THR the
recorder steps the stamps by -THR to catch up (a staircase). Random starting phase; recording lengths uniform
LO-HI h. stamp_clock is called as vitaldb-bench tools/clock_correction_periods.py calls it (correction_sign=-1,
defaults otherwise). Output: the fitted rate's error by number of correction events, and the rate source.
"""
import argparse, json
import numpy as np
from physiosync import stamp_clock

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--lo", type=float, default=1)
ap.add_argument("--hi", type=float, default=8); a = ap.parse_args()
rng = np.random.default_rng(a.seed)
FS, REC, THR = 500.0, 100, 0.48
res = {}
for _ in range(a.n):
    rate = rng.normal(-64e-6, 15e-6)                   # step-mode rates: 0.48 s per about 2 h, varying
    T = rng.uniform(a.lo, a.hi) * 3600
    k = np.arange(0, int(T * FS), REC, dtype=float)
    c = k / FS
    acc = rate * c - rng.uniform(0, THR)               # PC minus count: negative and growing (SNUADC fast)
    stamps = c - THR * np.floor(-acc / THR)            # staircase: a -THR step each time THR is reached
    clk = stamp_clock(stamps, k, FS, correction_sign=-1)
    ev = clk["corrections"]["events"]
    key = "0" if ev == 0 else "1" if ev == 1 else "2+"
    res.setdefault(key, []).append((clk["rate_ppm"] - rate * 1e6, clk["rate_source"]))
out = {"model": "staircase, THR 0.48 s, rate N(-64, 15) ppm, lengths %g-%g h" % (a.lo, a.hi)}
for key in ("0", "1", "2+"):
    v = res.get(key, [])
    if not v: continue
    err = np.array([x[0] for x in v])
    out[key] = {"n": len(v), "source": sorted(set(x[1] for x in v)),
                "rate_error_ppm_p10_50_90": np.round(np.percentile(err, [10, 50, 90]), 1).tolist()}
json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))
