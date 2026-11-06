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

1, recorder-pc-steps: recording pauses, from results/clock/pc_steps_*.jsonl: events where the SNUADC stamps jump
forward by more than 0.5 s and the BIS, where present, jumps by the same within 1 s; their count, files, and the
median and p10-p90 of the SNUADC jump, the Solar8000 and Primus excesses and the mean of the available jumps.

16, solar-snuadc-clock: the burst lag's slope on the adopted stamp model (burst_validation_m3/per_recording.csv)
in step-mode recordings without a whole reset interval (not in solar_snuadc_clock/reset_rates.csv) and in
recordings with no correction: count and median slope (ms/h).

45, continuity-gaps-segments: from signal_survey/tables, CardioQ FLOW gaps per hour of its span (median over
recordings) and its coverage (unbroken segments over span, median); BIS EEG1 gaps under 1 s as a share of the time
since the previous gap ended (or the first segment began), median and IQR (ppm).
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


def pauses(W: Path) -> None:
    ev, files = [], set()
    for p in sorted((W / "results" / "clock").glob("pc_steps_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            for e in r.get("events", []):
                s, b = e.get("snuadc_jump_s"), e.get("bis_jump_s")
                if s is None or s <= 0.5 or (b is not None and abs(b - s) > 1):
                    continue
                ev.append(e)
                files.add(r["case"])
    print(f"1 recorder-pc-steps: {len(ev)} pauses in {len(files)} files")
    for k in ("snuadc_jump_s", "solar_excess_s", "primus_excess_s", "mean"):
        v = [np.nanmean([x for x in (e.get("snuadc_jump_s"), e.get("bis_jump_s")) if x is not None]) if k == "mean"
             else e.get(k) for e in ev]
        v = np.array([x for x in v if x is not None], float)
        print(f"  {k}: median {np.median(v):.1f} s, p10-p90 {np.percentile(v, 10):.1f}-{np.percentile(v, 90):.1f} "
              f"({v.size})")


def burst_slopes(W: Path) -> None:
    import pandas as pd
    b = pd.read_csv(W / "burst_validation_m3" / "per_recording.csv")
    rr = set(pd.read_csv(W / "solar_snuadc_clock" / "reset_rates.csv").case)
    st = b[(b["mode"] == "step") & ~b.case.isin(rr)].slope_ms_h.dropna()
    no = b[b["mode"] == "none"].slope_ms_h.dropna()
    print(f"16 solar-snuadc-clock: step mode without a reset interval {st.size}, median {st.median():+.1f} ms/h; "
          f"no correction {no.size}, median {no.median():+.1f} ms/h")


def continuity(W: Path) -> None:
    import pandas as pd
    T = W / "signal_survey" / "tables"
    g = pd.concat(pd.read_parquet(f) for f in sorted(T.glob("gaps_*.parquet")))
    s = pd.concat(pd.read_parquet(f) for f in sorted(T.glob("segs_*.parquet")))
    m = pd.concat(pd.read_parquet(f, columns=["case", "track", "span"]) for f in sorted(T.glob("meta_*.parquet")))
    cq = m[m.track == "CardioQ__FLOW"].set_index("case").span
    ng = g[g.track == "CardioQ__FLOW"].groupby("case").size().reindex(cq.index, fill_value=0)
    cov = s[s.track == "CardioQ__FLOW"].groupby("case").d.sum().reindex(cq.index) / cq
    print(f"45 continuity-gaps-segments: CardioQ FLOW {cq.size} recordings; gaps per hour of span median "
          f"{np.median(ng / (cq / 3600)):.1f}; coverage median {np.nanmedian(cov):.3f}")
    ppm = []
    bs = s[s.track == "BIS__EEG1_WAV"].groupby("case").t.min()
    for c, d in g[g.track == "BIS__EEG1_WAV"].sort_values(["case", "t"]).groupby("case"):
        prev = bs.get(c, np.nan)
        for t, dd in zip(d.t, d.d):
            if dd < 1 and t > prev:
                ppm.append(dd / (t - prev) * 1e6)
            prev = t + dd
    ppm = np.array(ppm)
    print(f"  BIS EEG1 gaps under 1 s: {ppm.size}; median {np.median(ppm):.1f} ppm (IQR "
          f"{np.percentile(ppm, 25):.1f}-{np.percentile(ppm, 75):.1f})")


def main() -> None:
    W = Path(sys.argv[1])
    correction_periods(W)
    pleth_hr(W)
    pauses(W)
    burst_slopes(W)
    continuity(W)


if __name__ == "__main__":
    main()
