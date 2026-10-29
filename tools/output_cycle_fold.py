# SPDX-License-Identifier: MIT
"""240 Hz folding on many recordings: per recording, the cycle (from ART, else PLETH), and for each SNUADC channel
the swing of its relative step size around the cycle (max - min of a 12-bin profile, as a share) and the position of
its maximum. Control: the same fold at a wrong frequency (237.3 Hz). This machine only.

Usage: python tools/output_cycle_fold.py ROOT OUT.jsonl CASE [CASE ...]
"""
import sys, json
from pathlib import Path
import numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
root, out = Path(sys.argv[1]), Path(sys.argv[2]); cases = [int(c) for c in sys.argv[3:]]
CH = ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/PLETH", "SNUADC/CVP", "SNUADC/FEM"]
rel = lambda x: np.abs(np.diff(x)) / (np.convolve(np.abs(np.diff(x)), np.ones(25) / 25, "same") + 1e-9)
done = {json.loads(l)["case"] for l in out.read_text().splitlines()} if out.exists() else set()
for c in cases:
    if c in done: continue
    v = read_vital(root / "vital_files" / f"{c:04d}.vital", track_names=set(CH))
    have = {k: np.asarray(v.tracks[k].values, float) for k in CH if v.tracks.get(k) is not None and v.tracks[k].record_times.size}
    refk = "SNUADC/ART" if "SNUADC/ART" in have else "SNUADC/PLETH" if "SNUADC/PLETH" in have else None
    res = {"case": c, "ref": refk, "channels": {}}
    if refk:
        N = 500 * 600; n0 = 500 * 900
        r = rel(have[refk][n0:n0 + N]); n = np.arange(r.size)
        fs_ = np.arange(239.95, 240.05, 0.001)
        amp = [np.abs(np.mean((r - r.mean()) * np.exp(-2j * np.pi * f * n / 500))) for f in fs_]
        f = float(fs_[int(np.argmax(amp))]); ph0 = np.angle(np.mean((r - r.mean()) * np.exp(-2j * np.pi * f * n / 500)))
        res["cycle_hz"] = round(f, 4)
        for fold_f, tag in ((f, ""), (237.3, "_control")):
            phase = (fold_f * n / 500 + (ph0 / (2 * np.pi) if not tag else 0)) % 1.0; idx = (phase * 12).astype(int)
            for k, x in have.items():
                x = x[n0:n0 + N]
                if x.size < N or np.ptp(x) == 0: continue
                rr = rel(x); prof = np.array([np.mean(rr[idx == b]) for b in range(12)]); m = prof.mean()
                res["channels"].setdefault(k.split("/")[1], {})["swing" + tag] = round(float((prof.max() - prof.min()) / m), 3)
                if not tag: res["channels"][k.split("/")[1]]["peak_bin"] = int(np.argmax(prof))
    with out.open("a") as fh: fh.write(json.dumps(res) + "\n")
