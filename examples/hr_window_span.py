"""Span of ECG the Solar8000 HR window covers: N R-R intervals at each recording's median R-R (11 Nov 2026).

Atlas parameter hr-window-span (N = 8, the adopted window, hr-window). Reads the parts of tools/hr_derivation.py
(median_rr_s per recording; recordings without an analysis are skipped).

    python examples/hr_window_span.py $W/examples/hr_derivation_v4/part_*.jsonl --n 8 --out $W/examples/hr_derivation_v4/window_span.json
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rr = [r["median_rr_s"] for p in a.parts for r in map(json.loads, open(p, encoding="utf-8"))
          if r.get("cands") and r.get("median_rr_s")]
    s = a.n * np.array(rr)
    q = [round(float(x), 2) for x in np.percentile(s, [10, 25, 50, 75, 90])]
    out = {"n_intervals": a.n, "recordings": len(s), "span_s_p10_25_50_75_90": q,
           "median_rr_s": round(float(np.median(rr)), 3)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
