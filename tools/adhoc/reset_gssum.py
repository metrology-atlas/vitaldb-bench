# SPDX-License-Identifier: MIT
"""Where the signal lost at a SNUADC reset sits (metrology-atlas study reset-cluster-order): a summary of the
gap-search output of the reset-order run. Ad hoc: reads that run's saved files.

    python tools/adhoc/reset_gssum.py $W/results/clock/reset_order_v1

For PLETH, ART and CVP, from gs_*.jsonl: each reset's (and each control boundary's) strongest gap candidate
(position in samples from the cluster's first record, its length m in samples, its strength g). A confident
detection is g > 50 (also reported: g > 100) with m = 6-13 samples. Counts the resets that carry the channel, the
confident detections at resets and at the control boundaries ("base"), and how many reset detections fall in
positions 100-199, the last record on the old grid (and 0-199). Prints a line per channel; nothing is written.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def main() -> None:
    d = Path(sys.argv[1])
    rows = [json.loads(line) for f in sorted(d.glob("gs_*.jsonl")) for line in open(f)]
    for ch in ("PLETH", "ART", "CVP"):
        R = [e["top"][ch][0] for r in rows for e in r["resets"] if ch in e["top"]]
        B = [e["top"][ch][0] for r in rows for e in r["base"] if ch in e["top"]]
        for thr in (50, 100):
            rh = [x for x in R if x[2] > thr and 6 <= x[1] <= 13]
            bh = [x for x in B if x[2] > thr and 6 <= x[1] <= 13]
            print(f"{ch} thr {thr}: resets {len(R)}, confident {len(rh)} ({len(rh) / max(1, len(R)):.3f}); "
                  f"base {len(B)}, confident {len(bh)} ({len(bh) / max(1, len(B)):.3f})")
        rh = [x for x in R if x[2] > 50 and 6 <= x[1] <= 13]
        print(f"  in positions 100-199 (last old-grid record): {sum(100 <= p < 200 for p, m, g in rh)}; "
              f"0-199: {sum(0 <= p < 200 for p, m, g in rh)}; of {len(rh)}")
        print("  m:", collections.Counter(m for p, m, g in rh).most_common())


if __name__ == "__main__":
    main()
