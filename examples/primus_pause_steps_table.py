"""The table of the atlas study primus-pause-steps, from tools/primus_pause_steps.py agg's outputs (11 Nov 2026).

Until now the table's rows (pauses with CO2 carrying on or stopping too: count, median step and p10-p90, share within
150 ms; controls) were read off steps.jsonl by hand. This computes them, with the recordings the measured pauses
come from.

    python examples/primus_pause_steps_table.py AGG_DIR [--out table.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def row(v):
    v = np.asarray(v, float) * 1000
    return {"pauses": int(v.size), "median_ms": round(float(np.median(v))), "p10_ms": round(float(np.percentile(v, 10))),
            "p90_ms": round(float(np.percentile(v, 90))), "within_150ms": round(float(np.mean(np.abs(v) <= 150)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agg")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = Path(a.agg)
    s = [json.loads(x) for x in open(d / "steps.jsonl")]
    summ = json.loads((d / "summary.json").read_text())
    c = summ["control_s_p5_10_25_50_75_90_95"]
    t = {"recordings": len({x["case"] for x in s}), "pauses_measured": len(s),
         "co2_carries_on": row([x["step_s"] for x in s if not x["co2_gap"]]),
         "co2_stops_too": row([x["step_s"] for x in s if x["co2_gap"]]),
         "controls": {"median_ms": round(c[3] * 1000), "p10_ms": round(c[1] * 1000), "p90_ms": round(c[5] * 1000)}}
    t["co2_carries_on"]["share_of_pauses"] = round(t["co2_carries_on"]["pauses"] / len(s), 3)
    print(json.dumps(t, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(t, indent=1))


if __name__ == "__main__":
    main()
