# SPDX-License-Identifier: MIT
"""Check breath_match by injection: shift each case's tracked CVP lags by +-1 and +-2 breaths and see whether
breath matching finds the shift. Per case, this machine.

    python tools/vent_match_check.py --root R --cases 1,2 --units U --step-rates S --pilot vent20.jsonl --out out.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vent_sync_pilot as V
from pat import sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

ap = argparse.ArgumentParser()
for k in ("--root", "--cases", "--units", "--step-rates", "--pilot", "--out"):
    ap.add_argument(k, required=True)
a = ap.parse_args()
U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
sr = json.loads(Path(a.step_rates).read_text())
P = {json.loads(l)["case"]: json.loads(l) for l in open(a.pilot)}
done = set()
if Path(a.out).exists():
    done = {json.loads(l)["case"] for l in open(a.out)}
with open(a.out, "a") as fh:
    for c in [int(x) for x in a.cases.split(",")]:
        if c in done or c not in P or not P[c]["rows"]["cvp"]:
            continue
        v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={"SNUADC/CVP", "Primus/AWP"})
        aw = v.tracks["Primus/AWP"]
        ta = sample_times(aw)
        xa = np.nan_to_num(aw.physical().astype(float))
        on = V.breath_onsets(ta, xa, float(aw.srate))
        tc, _ = V.snu_times(v.tracks["SNUADC/CVP"], U.get(c, {}), sr)
        xc = v.tracks["SNUADC/CVP"].physical().astype(float)
        t0 = float(np.floor(min(ta[0], tc[0])))
        n = int((max(ta[-1], tc[-1]) - t0) / V.G) + 1
        ok = np.isfinite(xc)
        cvp50, awp50 = V.binned(tc[ok], xc[ok], t0, V.G, n), V.binned(ta, xa, t0, V.G, n)
        sst = np.asarray(V.stretches_of(aw)) - t0
        rows = P[c]["rows"]["cvp"]
        T = np.median(np.diff(on)[(np.diff(on) > 1.5) & (np.diff(on) < 12)])
        res = {"case": c, "T": round(float(T), 2)}
        for inj in (-2, -1, 0, 1, 2):
            rr = [[r[0], r[1] + inj * T * 1000, *r[2:]] for r in rows]
            res[str(inj)] = V.breath_match(on, awp50, cvp50, t0, rr, sst)
        fh.write(json.dumps(res) + "\n")
        fh.flush()
