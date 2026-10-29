# SPDX-License-Identifier: MIT
"""Rate check on Primus/AWP, as for CO2. Indicator: 'sandwiched holds' (a stored value equal to the previous one while
the samples either side change), i.e. a duplicated sample in a moving trace. Its spectrum has a family of lines;
for each line near the nominal list, fine-scan (0.0005 Hz) on 10-min pieces; record frequency and strength
(peak over median of a +-0.2 Hz neighbourhood). Also folds the indicator at the strongest line and at the spacing.

Usage: python tools/primus_awp_repeats.py ROOT CASE   (appends awp.jsonl in the current folder)
"""
import sys, json
from pathlib import Path
import numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
root, case = Path(sys.argv[1]), int(sys.argv[2]); F2 = 62.5
NOM = [4.23, 8.52, 12.35, 12.76, 16.58, 20.84, 24.58, 25.105, 28.93, 29.34]
v = read_vital(root / "vital_files" / f"{case:04d}.vital", track_names={"Primus/AWP"})
t = v.tracks.get("Primus/AWP"); res = {"case": case}
if t is not None and t.record_times.size:
    x = np.asarray(t.values, float); x = x[int(F2 * 600):int(F2 * 600) + int(F2 * 5400)]
    d = np.diff(x); sw = ((d[1:-1] == 0) & (d[:-2] != 0) & (d[2:] != 0)).astype(float); sw -= sw.mean()
    res["hold_share"] = round(float((sw + 0).mean() + ((d[1:-1] == 0) & (d[:-2] != 0) & (d[2:] != 0)).mean()), 3)
    P = int(F2 * 600); out = {str(f): [] for f in NOM}
    for s0 in range(0, sw.size - P + 1, P):
        seg = sw[s0:s0 + P]; k = np.arange(P)
        for f0 in NOM:
            fs_ = np.arange(f0 - 0.2, f0 + 0.2, 0.0005)
            a = np.array([abs(np.mean(seg * np.exp(-2j * np.pi * f * k / F2))) for f in fs_])
            i = int(np.argmax(np.where(np.abs(fs_ - f0) < 0.06, a, 0)))
            out[str(f0)].append([round(float(fs_[i]), 4), round(float(a[i] / np.median(a)), 1)])
    res["lines"] = out
print(json.dumps(res))
with open("awp.jsonl", "a") as fh: fh.write(json.dumps(res) + "\n")
