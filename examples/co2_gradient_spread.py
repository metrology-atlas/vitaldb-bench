"""Arterial to end-tidal CO2 gradient: spread between and within patients (11 Nov 2026).

Atlas parameter paco2-etco2-between-patient-sd (study blood-gas-at-the-draw). Reads the parts of
examples/blood_gas_gradients.py run (one row per gas and column; kind etco2, point_draw = PaCO2 minus EtCO2 at the
draw): between patients, the SD of each patient's median gradient; within, the median over patients with two or
more gases of their own SD.

    python examples/co2_gradient_spread.py $W/examples/bg/parts --out OUT.json
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(Path(a.parts) / "*.csv")))])
    e = d[d.kind == "etco2"].dropna(subset=["point_draw"])
    g = e.groupby("case").point_draw
    n, sd = g.size(), g.std()
    out = {"patients": int(n.size), "between_patient_sd": round(float(g.median().std()), 2),
           "patients_two_or_more": int((n >= 2).sum()), "within_patient_sd_median": round(float(sd[n >= 2].median()), 2)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
