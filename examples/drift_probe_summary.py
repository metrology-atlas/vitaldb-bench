"""Summary of tools/primus_drift_probe.py outputs: the Primus drift against the SNUADC stamp model (11 Nov 2026).

The atlas parameter primus-drift-breath is the median over recordings of the common slope (one slope per recording,
an offset per stretch between AWP pauses), with p10 and p90 and the number of recordings. Until now it was read off
the probe's output by hand.

    python examples/drift_probe_summary.py PROBE.jsonl [--out summary.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("--out")
    a = ap.parse_args()
    R = [json.loads(l) for l in open(a.probe) if l.strip()]
    out = {}
    for k in ("slope_common_ppm", "slope_single_ppm"):
        v = np.array([r[k] for r in R], float)
        out[k] = {"n": int(v.size), "median": round(float(np.median(v)), 1),
                  "p10": round(float(np.percentile(v, 10)), 1), "p90": round(float(np.percentile(v, 90)), 1)}
    print(json.dumps(out))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
