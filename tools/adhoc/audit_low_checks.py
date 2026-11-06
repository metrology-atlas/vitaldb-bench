# SPDX-License-Identifier: MIT
"""Checks behind the low audit findings of 1 November 2026 that needed a number recomputed (metrology-atlas). Ad hoc:
reads saved outputs only; prints, writes nothing.

    python tools/adhoc/audit_low_checks.py $W

8, correction-periods: the SNUADC step-mode reset interval against the interval its rate predicts. From
results/clock/periods_{a,b}.jsonl: every interval between two consecutive resets (event_intervals_s of ECG II),
with the recording's rate (from the same stamps, so the comparison checks the threshold, not the rate);
the observed median and p10-p90, and the median of threshold / |rate| for thresholds of 0.4804 s (the stamp step),
0.498 s (the stamp step plus the 17.7 ms lost at a reset: the drift actually corrected) and 0.5 s.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def correction_periods(W: Path) -> None:
    gaps, rates = [], []
    for p in ("a", "b"):
        for line in open(W / "results" / "clock" / f"periods_{p}.jsonl"):
            t = json.loads(line).get("SNUADC/ECG_II")
            if not t or not t.get("rate_ppm") or t["rate_ppm"] >= 0:
                continue
            for g in t.get("event_intervals_s") or []:
                gaps.append(g)
                rates.append(abs(t["rate_ppm"]) * 1e-6)
    g, r = np.array(gaps) / 3600, np.array(rates)
    print(f"8 correction-periods: {g.size} intervals; observed median {np.median(g):.3f} h "
          f"(p10-p90 {np.percentile(g, 10):.2f}-{np.percentile(g, 90):.2f})")
    for thr in (0.4804, 0.498, 0.5):
        print(f"  {thr} s / |rate|: median {np.median(thr / r / 3600):.3f} h")


def main() -> None:
    W = Path(sys.argv[1])
    correction_periods(W)


if __name__ == "__main__":
    main()
