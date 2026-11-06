# SPDX-License-Identifier: MIT
"""Checks behind the low audit findings of 1 November 2026 that needed a number recomputed (metrology-atlas). Ad hoc:
reads saved outputs only; prints, writes nothing.

    python tools/adhoc/audit_low_checks.py $W

8, correction-periods: the SNUADC step-mode reset interval against the interval its rate predicts. From
results/clock/periods_{a,b}.jsonl: every interval between two consecutive resets (event_intervals_s of ECG II),
with the recording's rate (from the same stamps, so the comparison checks the threshold, not the rate);
the observed median and p10-p90, and the median of threshold / |rate| for thresholds of 0.4804 s (the stamp step),
0.498 s (the stamp step plus the 17.7 ms lost at a reset: the drift actually corrected) and 0.5 s.

54, pleth-hr-derivation: the adopted rule's (rmean|n|10|round) delay over recordings, from
examples/pleth_hr_derivation/part_*.jsonl: 5th-95th percentiles, robust SD (1.4826 x MAD), the window's span (10
pulse intervals at the recording's median interval, 5th-95th percentiles) and the share of recordings where
rounding reproduces more values exactly than truncation.
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


def pleth_hr(W: Path) -> None:
    d, span, rnd = [], [], []
    for p in sorted((W / "examples" / "pleth_hr_derivation").glob("part_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            c = r.get("cands") or {}
            a, b = c.get("rmean|n|10|round"), c.get("rmean|n|10|trunc")
            if not a:
                continue
            d.append(a[0])
            span.append(10 * r["median_interval_s"])
            if b:
                rnd.append(a[1] > b[1])
    d = np.array(d)
    mad = 1.4826 * np.median(np.abs(d - np.median(d)))
    print(f"54 pleth-hr-derivation: {d.size} recordings; delay median {np.median(d):.2f} s, p5-p95 "
          f"{np.percentile(d, 5):.2f}-{np.percentile(d, 95):.2f}; robust SD {mad:.3f} s; window p5-p95 "
          f"{np.percentile(span, 5):.2f}-{np.percentile(span, 95):.2f} s; round beats trunc in "
          f"{np.mean(rnd):.3f} of {len(rnd)}")


def main() -> None:
    W = Path(sys.argv[1])
    correction_periods(W)
    pleth_hr(W)


if __name__ == "__main__":
    main()
