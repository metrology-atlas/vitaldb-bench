# SPDX-License-Identifier: MIT
"""Can a recording without a whole tooth borrow its setup's SNUADC rate? (9 Nov 2026)

    python examples/setup_rate_check.py PER_RECORDING.csv OUT.json [--min-n 5]

PER_RECORDING.csv from vitaldb-bench tools/mains_hum.py agg (snu_rate, snu_events, snu_teeth, key = the setup's
device ports; all recordings, not only those with hum). For every recording whose SNUADC rate comes from whole teeth,
the leave-one-out median of the other whole-teeth rates in its setup, and the difference. Also how many recordings
without a whole tooth (with one or more corrections) have a setup with at least --min-n whole-teeth recordings, and
how many of those have hum.
"""
import argparse, collections, json
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("table"); ap.add_argument("out"); ap.add_argument("--min-n", type=int, default=5)
a = ap.parse_args()
m = pd.read_csv(a.table)
m = m[m.snu_rate.notna() & (m.snu_events >= 1)].copy()
m["teeth"] = m.snu_teeth.fillna(0) >= 1
by = collections.defaultdict(list)
for k, r in zip(m.key[m.teeth], m.snu_rate[m.teeth]):
    by[k].append(r)
d, n_ok = [], 0
for k, r in zip(m.key[m.teeth], m.snu_rate[m.teeth]):
    others = list(by[k]); others.remove(r)
    if len(others) >= a.min_n:
        d.append(r - float(np.median(others)))
d = np.array(d)
nt = m[~m.teeth]
covered = nt[nt.key.map(lambda k: len(by.get(k, [])) >= a.min_n)]
out = {"min_n": a.min_n, "whole_teeth_recordings": int(m.teeth.sum()),
       "loo_checked": int(d.size),
       "own_minus_setup_ppm": {"median": round(float(np.median(d)), 2),
                               "p5_25_75_95": np.round(np.percentile(d, [5, 25, 75, 95]), 2).tolist(),
                               "within_3ppm_share": round(float(np.mean(np.abs(d) <= 3)), 3),
                               "robust_sd": round(float(1.4826 * np.median(np.abs(d - np.median(d)))), 2)},
       "no_tooth_recordings": int(len(nt)), "no_tooth_with_setup_rate": int(len(covered)),
       "no_tooth_with_setup_rate_and_hum": int(covered.snu_line.notna().sum()),
       "setups_with_rate": int(sum(len(v) >= a.min_n for v in by.values()))}
json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))
