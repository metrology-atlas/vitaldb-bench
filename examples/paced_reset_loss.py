"""Signal lost at a SNUADC step-mode correction, from the paced-ECG breaks (11 Nov 2026).

Atlas parameter snuadc-reset-loss (study paced-ecg). Reads the output of tools/paced_breaks.py and keeps the events
where the stamps step back by a reset (stamp minus count within --tol-ms of -481 ms); the signal lost is minus the
step on the sample count. Leads of one correction are averaged first, then the corrections.

    python examples/paced_reset_loss.py $W/results/paced/breaks_rec.jsonl --out $W/results/paced/reset_loss.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("breaks")
    ap.add_argument("--reset-ms", type=float, default=-481.0)
    ap.add_argument("--tol-ms", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by = defaultdict(list)
    rows = []
    for line in open(a.breaks, encoding="utf-8"):
        r = json.loads(line)
        for e in r.get("events", []):
            if abs((e.get("stamp_minus_count_ms") or 0) - a.reset_ms) <= a.tol_ms:
                loss = -e["step_count_ms"]
                by[(r["case"], e["t_stamp"])].append(loss)
                rows.append({"case": r["case"], "lead": r["lead"], "t_stamp": e["t_stamp"], "loss_ms": round(loss, 3)})
    per = [float(np.mean(v)) for v in by.values()]
    meas = [x["loss_ms"] for x in rows]
    out = {"measurements": rows, "n_measurements": len(rows), "corrections": len(by),
           "per_correction_ms": [round(x, 2) for x in per],
           "mean_ms": round(float(np.mean(per)), 2) if per else None,
           "min_ms": round(min(meas), 1) if meas else None, "max_ms": round(max(meas), 1) if meas else None}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "measurements"}))


if __name__ == "__main__":
    main()
