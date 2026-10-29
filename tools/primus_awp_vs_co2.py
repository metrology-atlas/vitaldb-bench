# SPDX-License-Identifier: MIT
"""Experiment 3: are AWP repeats locked to the CO2 update grid? CO2 updates modelled as a grid at the recording's
CO2 rate (saw2.jsonl), phase re-estimated in 60 s pieces; each steep AWP sample gets its position in the CO2
40 ms cycle (sample times from record_times, since the two tracks' records differ). Repeat rate by position, 10 bins.
Control: the same with the grid rate off by 0.05 Hz.

Usage: python tools/primus_awp_vs_co2.py ROOT CASE   (needs saw2.jsonl from art_cycle_and_co2_grid.py; appends coinc.jsonl)
"""
import sys, json
from pathlib import Path
import numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
root, case = Path(sys.argv[1]), int(sys.argv[2]); F2 = 62.5
fu = float(np.median([d for d in map(json.loads, open("saw2.jsonl")) if d["case"] == case][0]["co2_update_hz"]))
v = read_vital(root / "vital_files" / f"{case:04d}.vital", track_names={"Primus/AWP", "Primus/CO2"})
def times(t):
    rt = np.asarray(t.record_times, float); rs = np.asarray(t.record_sizes)
    return np.concatenate([r + np.arange(s) / F2 for r, s in zip(rt, rs)])
a, c = v.tracks["Primus/AWP"], v.tracks["Primus/CO2"]
xa, ta = np.asarray(a.values, float), times(a); xc, tc = np.asarray(c.values, float), times(c)
T0 = tc[0] + 600; W = (tc > T0) & (tc < T0 + 5400); xc, tc = xc[W], tc[W]; Wa = (ta > T0) & (ta < T0 + 5400); xa, ta = xa[Wa], ta[Wa]
dc = np.diff(xc); chg = (dc != 0).astype(float); act = np.convolve(chg, np.ones(25), "same") > 0; chg = chg - chg[act].mean(); chg[~act] = 0
tchg = tc[1:]
i = np.arange(1, xa.size - 2); st = np.abs(xa[i + 2] - xa[i - 1]) >= 8; rep = (np.diff(xa)[i] == 0)
ts, rs_ = ta[i + 1][st], rep[st]
res = {"case": case, "co2_hz": fu}
for tag, f in (("", fu), ("_control", fu + 0.05)):
    pos = np.full(ts.size, np.nan)
    for p0 in np.arange(T0, T0 + 5400, 60):
        m = (tchg >= p0) & (tchg < p0 + 60); 
        if chg[m].std() == 0: continue
        ph = np.angle(np.sum(chg[m] * np.exp(-2j * np.pi * f * (tchg[m] - p0)))) / (2 * np.pi)
        q = (ts >= p0) & (ts < p0 + 60); pos[q] = ((ts[q] - p0) * f - ph) % 1
    ok = np.isfinite(pos); b = np.minimum((pos[ok] * 10).astype(int), 9)
    prof = [float(rs_[ok][b == k].mean()) for k in range(10)]
    res["profile" + tag] = [round(p, 4) for p in prof]; res["swing" + tag] = round((max(prof) - min(prof)) / np.mean(prof), 3)
    res["n" + tag] = int(ok.sum())
print(json.dumps(res))
with open("coinc.jsonl", "a") as fh: fh.write(json.dumps(res) + "\n")
