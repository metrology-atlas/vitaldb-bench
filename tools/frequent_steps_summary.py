# SPDX-License-Identifier: MIT
"""The SNUADC's frequent-correction mode in numbers (metrology-atlas study clock-correction-scan, parameter
snuadc-frequent-steps): how many recordings, how often they step and by how much.

    python tools/frequent_steps_summary.py --modes $W/results/sync_full/snuadc_mode.json \
        --labels $W/results/sync_full/snuadc_mode_label.json \
        --periods $W/results/clock/periods_a.jsonl $W/results/clock/periods_b.jsonl --out frequent_steps.json

Rates from the correction scan's step counts (snuadc_mode.json: small corrections and hours per recording labelled
"small"); sizes from the clock_correction_periods.py run (SNUADC/ECG_II small_step_ms, per recording p10/p50/p90 of
its steps of 0.1 ms to 0.1 s), which covers only some of these recordings. Also the long recordings (2.5 h or more)
with no 0.49 s correction, by mode. Percentiles are numpy's (linear).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--periods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    modes = json.load(open(a.modes))
    labels = json.load(open(a.labels))
    small = [k for k, v in labels.items() if v == "small" and k in modes]
    rate = np.array([modes[k]["small"] / modes[k]["dur_h"] for k in small if modes[k]["dur_h"] > 0])
    rows = [json.loads(x) for f in a.periods for x in open(f) if x.strip()]
    sizes = [r["SNUADC/ECG_II"]["small_step_ms"] for r in rows
             if str(r.get("case")) in set(small) and (r.get("SNUADC/ECG_II") or {}).get("small_step_ms")]
    sz = np.array(sizes, float)
    long_ = [k for k, v in modes.items() if v["dur_h"] >= 2.5 and v["steps"] == 0]
    S = {"recordings": len(small),
         "small_corrections": int(sum(modes[k]["small"] for k in small)),
         "hours": round(float(sum(modes[k]["dur_h"] for k in small)), 1),
         "per_hour_p10_p50_p90": np.percentile(rate, [10, 50, 90]).round(2).tolist(),
         "size_recordings": int(len(sz)),
         "size_ms_recording_median_p10_p50_p90": np.percentile(sz[:, 1], [10, 50, 90]).round(1).tolist() if len(sz) else None,
         "size_ms_min_p10_max_p90": [round(float(sz[:, 0].min()), 1), round(float(sz[:, 2].max()), 1)] if len(sz) else None,
         "negative_recording_p10": int(np.sum(sz[:, 0] < 0)) if len(sz) else None,
         "long_without_049": {"recordings": len(long_),
                              "by_label": dict(Counter(labels.get(k, "?") for k in long_)),
                              "none_at_all": sum(modes[k]["small"] == 0 for k in long_),
                              "one_small": sum(modes[k]["small"] == 1 for k in long_),
                              "several_small": sum(modes[k]["small"] > 1 for k in long_)}}
    with open(a.out, "w") as fh:
        fh.write(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
