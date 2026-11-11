"""Recording duration over the release: caseend minus casestart in clinical_data.csv (11 Nov 2026).

Atlas parameter recording-duration.

    python examples/recording_duration.py $R/clinical_data.csv --out OUT.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clinical")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = pd.read_csv(a.clinical, encoding="utf-8-sig")
    d = (c.caseend - c.casestart).to_numpy(float)
    q = np.percentile(d, [5, 25, 50, 75, 95])
    out = {"recordings": int(d.size), "duration_s_p5_25_50_75_95": [round(float(x), 1) for x in q],
           "min_s": float(d.min()), "max_s": float(d.max())}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
