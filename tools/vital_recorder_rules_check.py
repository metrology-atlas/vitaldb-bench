# SPDX-License-Identifier: MIT
"""Test the Vital Recorder's SNUADC stamping rules, as described by the dataset's creator (H.-C. Lee, November 2026),
against the release (metrology-atlas: snuadc-long-uncorrected, clock-correction-scan, stamp-arithmetic).

    python tools/vital_recorder_rules_check.py --work $W --root $R --out $W/recorder_rules/summary.json \
        [--time-budget S]

The rules: the first sample of a segment is stamped with the PC time at arrival minus the block's duration; stamps
then advance by counting samples; at each read the counted time is compared with the PC clock. Behind by up to
0.5 s: blank samples are inserted to close the gap. Behind by more: a new segment starts at the PC time. Ahead by
more than 0.5 s: the excess samples at the end of the block are discarded and the count restarts from the PC time.
So a converter fast against the PC gains on it until 0.5 s ahead, then resets (step mode); a slow one is filled at
every read (frequent-correction mode); a fast one that never gains 0.5 s within a segment shows nothing.

Tests (rates in physiosync's convention, r = f_PC - f_SNUADC, negative = SNUADC fast):

1. **No correction ("none" recordings with a hum rate,** hum_snuadc/release/hum_rates.csv, 13 ppm robust SD
   against the bursts, hum-snuadc-rate). The rules allow only fast converters that never gain 0.5 s within a
   segment: -0.5 s / T <= r <= 0, T the longest SNUADC segment (signal_survey/tables/segs_*.parquet, ECG II, else
   the longest SNUADC track). Reported: the share inside, inside within 2 noise SD, and clearly outside on each side
   (r > +2 SD: a slow converter that was never filled; r < bound - 2 SD: a fast one that never reset).
2. **Step mode:** the reset-interval rates (signal_survey/continuity/snuadc_reset_drift_per_recording.csv) must all be
   fast (r < 0), and the rate times the interval is the correction (clock-correction-scan; not recomputed here).
3. **Frequent-correction mode ("small" recordings):** every forward stamp step of ECG II between 0.1 ms and 100 ms
   (a step and its reverse within two records left out), read from the files: its size, the share that is a whole
   number of samples (within 0.1 sample; blanks inserted keep the count's grid), and the implied rate (total forward
   steps over the recording's hours, positive = slow); and whether the signal is continuous across the step (the
   change from the last sample before it to the first after, against the median change over 10 samples either
   side: continuous if at most 3 times it), as it is if the step is time added to the stamps with no samples lost
   or inserted; the control is the same measure at as many ordinary record boundaries of the same recording. The hum rate is listed beside the implied rate where there is one, but it is not a test here: it
   adds the PC's rate against the grid taken from the recording's setup (hum-snuadc-rate), which does not hold for
   a recording whose stamps follow its own PC.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HUM_SD = 13.0          # ppm, robust SD of the hum rate against the bursts (hum-snuadc-rate)
FS = 500.0


def q5(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return [round(float(x), 2) for x in np.percentile(v, [5, 25, 50, 75, 95])] + [int(v.size)] if v.size else None


def longest_segments(W: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for f in sorted(glob.glob(str(W / "signal_survey" / "tables" / "segs_*.parquet"))):
        d = pd.read_parquet(f, columns=["case", "track", "d"])
        d = d[d.track.str.startswith("SNUADC__")]
        ecg = d[d.track == "SNUADC__ECG_II"].groupby("case").d.max()
        anyt = d.groupby("case").d.max()
        for c, v in anyt.items():
            out[int(c)] = float(ecg.get(c, v))
    return out


def test_none(W: Path, labels: dict, seg: dict) -> dict:
    R = list(csv.DictReader(open(W / "hum_snuadc" / "release" / "hum_rates.csv")))
    rows = []
    for r in R:
        c = int(r["case"])
        if labels.get(str(c)) != "none" or c not in seg or seg[c] <= 0:
            continue
        rate = float(r["rate_ppm"])
        bound = -0.5 / seg[c] * 1e6
        rows.append((rate, bound, seg[c] / 3600))
    rate = np.array([x[0] for x in rows])
    bound = np.array([x[1] for x in rows])
    inside = (rate >= bound) & (rate <= 0)
    near = (rate >= bound - 2 * HUM_SD) & (rate <= 2 * HUM_SD)
    return {"recordings": len(rows), "rate_ppm": q5(rate), "bound_ppm": q5(bound),
            "longest_segment_h": q5([x[2] for x in rows]),
            "inside_share": round(float(inside.mean()), 3), "inside_within_2sd_share": round(float(near.mean()), 3),
            "slow_beyond_2sd": int(np.sum(rate > 2 * HUM_SD)),
            "fast_beyond_bound_2sd": int(np.sum(rate < bound - 2 * HUM_SD))}


def test_step(W: Path, labels: dict) -> dict:
    s = pd.read_csv(W / "signal_survey" / "continuity" / "snuadc_reset_drift_per_recording.csv")
    s = s[[labels.get(str(int(c))) == "step" for c in s.case]]
    return {"recordings": int(len(s)), "fast_share": round(float(np.mean(s.ppm < 0)), 4), "rate_ppm": q5(s.ppm)}


def test_frequent(W: Path, root: Path, labels: dict, hum: dict, budget: float) -> dict:
    from vitaldb_reader.vital_file import read_vital
    cases = sorted(int(c) for c, v in labels.items() if v == "small")
    t0 = time.time()
    sizes, whole, cont, ctrl, per = [], [], [], [], []
    rng = np.random.default_rng(0)

    def continuous(v, b):
        local = np.median(np.abs(np.diff(np.r_[v[b - 11:b], v[b:b + 11]])))
        return abs(int(v[b]) - int(v[b - 1])) <= 3 * max(local, 1)
    done = 0
    for c in cases:
        if time.time() - t0 > budget:
            break
        p = root / "vital_files" / f"{c:04d}.vital"
        if not p.exists():
            continue
        tr = read_vital(p, track_names={"SNUADC/ECG_II"}).tracks.get("SNUADC/ECG_II")
        if tr is None or tr.record_times.size < 10:
            continue
        done += 1
        rt, rs = tr.record_times.astype(float), tr.record_sizes
        d = np.diff(rt) - rs[:-1] / FS
        idx = np.flatnonzero((d >= 1e-4) & (d < 0.1))
        rev = set()
        for i in idx:                                     # a step and its reverse within two records: a flick
            for j in (i + 1, i + 2):
                if j < d.size and abs(d[i] + d[j]) < 1e-4:
                    rev.update((i, j))
        idx = [i for i in idx if i not in rev]
        v = np.asarray(tr.values).astype(np.int64)
        off = tr.record_offsets
        for i in idx:
            sizes.append(d[i] * 1000)
            whole.append(abs(d[i] * FS - round(d[i] * FS)) < 0.1)
            b = int(off[i + 1])                              # first sample after the step
            if 11 <= b <= v.size - 12:
                cont.append(continuous(v, b))
        plain = np.flatnonzero(np.abs(d) < 1e-4)            # ordinary record boundaries: the control
        for i in rng.choice(plain, size=min(len(idx), plain.size), replace=False) if plain.size else []:
            b = int(off[i + 1])
            if 11 <= b <= v.size - 12:
                ctrl.append(continuous(v, b))
        hours = (rt[-1] - rt[0]) / 3600
        per.append({"case": c, "steps": len(idx), "hours": round(hours, 2),
                    "implied_ppm": round(float(np.sum(d[idx]) / (hours * 3600) * 1e6), 2) if hours > 0 else None,
                    "hum_ppm": hum.get(c)})
    both = [(x["implied_ppm"], x["hum_ppm"]) for x in per if x["hum_ppm"] is not None and x["implied_ppm"] is not None]
    return {"recordings_read": done, "of": len(cases), "steps": len(sizes), "step_ms": q5(sizes),
            "whole_sample_share": round(float(np.mean(whole)), 3) if whole else None,
            "step_ms_histogram_1ms": dict(zip(*[x.tolist() for x in np.unique(np.round(sizes).astype(int),
                                                                                 return_counts=True)])) if sizes else None,
            "continuous_share": round(float(np.mean(cont)), 3) if cont else None,
            "continuous_share_control": round(float(np.mean(ctrl)), 3) if ctrl else None,
            "implied_rate_ppm": q5([x["implied_ppm"] for x in per]),
            "implied_vs_hum": [[round(a, 1), round(b, 1)] for a, b in both]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--time-budget", type=float, default=1e9)
    a = ap.parse_args()
    W = a.work
    labels = json.load(open(W / "results" / "sync_full" / "snuadc_mode_label.json"))
    hum = {int(r["case"]): float(r["rate_ppm"]) for r in csv.DictReader(open(W / "hum_snuadc" / "release" / "hum_rates.csv"))}
    seg = longest_segments(W)
    S = {"none": test_none(W, labels, seg), "step": test_step(W, labels),
         "frequent": test_frequent(W, a.root, labels, hum, a.time_budget)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
