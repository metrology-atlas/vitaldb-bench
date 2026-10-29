# SPDX-License-Identifier: MIT
"""Experiment 1: where do AWP repeated samples fall within the .vital file's 30-sample (480 ms) records?
Steep parts only (|x[i+2]-x[i-1]| >= 8 codes), so natural repeats are rare. Also CO2 change positions by record slot.

Usage: python tools/primus_awp_records.py ROOT CASE   (appends blocks.jsonl in the current folder)
"""
import sys, json
from pathlib import Path
import numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
root, case = Path(sys.argv[1]), int(sys.argv[2])
v = read_vital(root / "vital_files" / f"{case:04d}.vital", track_names={"Primus/AWP", "Primus/CO2"})
res = {"case": case}
for k in ("Primus/AWP", "Primus/CO2"):
    t = v.tracks.get(k)
    if t is None or not t.record_times.size: continue
    x = np.asarray(t.values, float); ro = np.asarray(t.record_offsets)[:-1]; rs = np.asarray(t.record_sizes)
    slot = np.full(x.size, -1); 
    for o, s in zip(ro, rs):
        if s == 30: slot[o:o + 30] = np.arange(30)
    d = np.diff(x)
    if k.endswith("AWP"):
        i = np.arange(1, x.size - 2); st = np.abs(x[i + 2] - x[i - 1]) >= 8; ev = i[st & (d[i] == 0)] + 1; base = i[st] + 1
    else:
        i = np.arange(1, x.size - 1); act = (np.convolve((d != 0).astype(float), np.ones(25), "same") > 0)[i]
        ev = i[act & (d[i] != 0)] + 1; base = i[act] + 1
    ev, base = ev[slot[ev] >= 0], base[slot[base] >= 0]
    h = np.bincount(slot[ev], minlength=30); hb = np.bincount(slot[base], minlength=30)
    res[k.split("/")[1]] = {"events": int(ev.size), "rate_by_slot": [round(float(a / b), 4) if b else None for a, b in zip(h, hb)],
                            "record_dt_ms": sorted(set(np.round(np.diff(np.asarray(t.record_times, float)) * 1000, 1).tolist()))[:5]}
print(json.dumps(res))
with open("blocks.jsonl", "a") as fh: fh.write(json.dumps(res) + "\n")
