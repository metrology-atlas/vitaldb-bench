# SPDX-License-Identifier: MIT
"""Survey of every time series in VitalDB: value range and converter bits, extremes and plateaus (candidate clip and
leads-off codes), flatlines, gaps and continuous segments. A survey only: nothing is labelled or decided.
Per case, this machine only; aggregates leave.

    python tools/signal_survey.py scan --root R --cases 1,2,3 --out part.jsonl [--time-budget S]
    python tools/signal_survey.py agg part*.jsonl --out DIR

scan, per waveform track (track type 1):
  storage: sample format and its bits, gain, offset (physical = count * gain + offset), sample rate
  range: min / max count, p0.1 / p99.9, distinct counts, effective bits (log2 of the p0.1-p99.9 span + 1) and
    bits used (log2 of the max-min span + 1); physical min / max
  extremes: share of samples at the minimum and at the maximum count, longest run there (samples)
  plateaus: every level with runs of >= 0.5 s of identical samples, with the number of runs and their total seconds
    (top 8 levels by time): candidate clip levels inside the converter range and leads-off codes
  flatlines: runs of >= 1 s of identical samples at any level: count, total s, longest s, histogram of durations
  continuity: records are joined into continuous segments when the next record starts within 1.5 sample periods
    of the end of the previous one; histograms of segment and gap durations, coverage (sum of segment time over
    first-to-last span), overlaps
  non-finite samples, and the value grid as the converter step (float formats)
per numeric track (type 2): records, span, median interval, histogram of intervals in multiples of the median and
  in seconds, longest gap, longest run of an identical value (s), share of zeros, min / max.
agg: per signal name, distributions over cases of all the above; candidate clip / leads-off codes (plateau levels
  that recur across cases); summary.json and figures.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

FMT_BITS = {1: ("float32", 32), 2: ("float64", 64), 3: ("int8", 8), 4: ("uint8", 8), 5: ("int16", 16),
            6: ("uint16", 16), 7: ("int32", 32), 8: ("uint32", 32)}
GAP_EDGES = [0.0, 0.1, 1.0, 10.0, 60.0, 600.0, 3600.0, float("inf")]       # s, gaps beyond 1.5 periods
SEG_EDGES = [0.0, 1.0, 10.0, 60.0, 600.0, 3600.0, 4 * 3600.0, float("inf")]
FLAT_EDGES = [1.0, 2.0, 5.0, 10.0, 60.0, 600.0, float("inf")]
NUM_MULT_EDGES = [0.0, 1.5, 3.0, 10.0, float("inf")]                        # interval / median interval


def hist(x, edges) -> list[int]:
    return np.histogram(np.asarray(x, float), bins=edges)[0].astype(int).tolist() if len(x) else [0] * (len(edges) - 1)


def runs(x: np.ndarray):
    """Start index, length and value of each run of identical consecutive samples."""
    if x.size == 0:
        return np.zeros(0, int), np.zeros(0, int), x[:0]
    ch = np.flatnonzero(np.diff(x) != 0) + 1
    st = np.r_[0, ch]
    ln = np.diff(np.r_[st, x.size])
    return st, ln, x[st]


def continuity(rt: np.ndarray, sizes: np.ndarray, fs: float) -> dict:
    if rt.size == 0:
        return {}
    o = np.argsort(rt, kind="stable")
    rt, sizes = rt[o], sizes[o]
    end = rt + sizes / fs
    run_end = np.maximum.accumulate(end)
    step = rt[1:] - run_end[:-1]
    tol = 1.5 / fs
    brk = step > tol
    seg_start = np.r_[rt[0], rt[1:][brk]]
    seg_end = np.r_[run_end[:-1][brk], run_end[-1]]
    segs = seg_end - seg_start
    gaps = step[brk]
    span = float(run_end[-1] - rt[0])
    return {"span_s": round(span, 1), "segments": int(segs.size), "seg_hist": hist(segs, SEG_EDGES),
            "seg_longest_s": round(float(segs.max()), 1), "seg_median_s": round(float(np.median(segs)), 1),
            "gaps": int(gaps.size), "gap_hist": hist(gaps, GAP_EDGES), "gap_total_s": round(float(gaps.sum()), 1),
            "gap_longest_s": round(float(gaps.max()), 1) if gaps.size else 0.0,
            "coverage": round(float(segs.sum() / span), 4) if span > 0 else None,
            "overlaps": int((step < -tol).sum()), "overlap_s": round(float(-step[step < -tol].sum()), 2)}


def wave(tr) -> dict:
    fs = float(tr.srate)
    fmt = int(tr.fmt)
    name, bits = FMT_BITS.get(fmt, (str(fmt), None))
    out = {"fs": fs, "fmt": name, "storage_bits": bits, "gain": float(tr.gain), "offset": float(tr.offset)}
    x = np.asarray(tr.values)
    out["n"] = int(x.size)
    out.update(continuity(np.asarray(tr.record_times, float), np.asarray(tr.record_sizes, float), fs))
    if x.size == 0:
        return out
    if tr.is_integer_format:
        x = x.astype(np.int64)
        lo, hi = int(x.min()), int(x.max())
        bc = np.bincount(x - lo)
        cs = np.cumsum(bc) / x.size
        p001, p999 = lo + int(np.searchsorted(cs, 0.001)), lo + int(np.searchsorted(cs, 0.999))
        out.update({"min": lo, "max": hi, "p001": p001, "p999": p999, "distinct": int((bc > 0).sum()),
                    "eff_bits": round(math.log2(p999 - p001 + 1), 2), "bits_used": round(math.log2(hi - lo + 1), 2),
                    "phys_min": round(lo * tr.gain + tr.offset, 4), "phys_max": round(hi * tr.gain + tr.offset, 4),
                    "at_min": round(float(bc[0] / x.size), 6), "at_max": round(float(bc[-1] / x.size), 6)})
    else:
        fin = np.isfinite(x)
        out["nonfinite"] = round(float(1 - fin.mean()), 6)
        x = x[fin]
        if x.size == 0:
            return out
        out.update({"min": float(x.min()), "max": float(x.max()), "p001": float(np.percentile(x, 0.1)),
                    "p999": float(np.percentile(x, 99.9))})
        # float tracks: the value grid (smallest step between distinct values) stands in for the converter step
        u = np.unique(np.round(x[:2_000_000].astype(float), 6))
        d = np.diff(u)
        d = d[d > 1e-9]
        if d.size:
            step = float(np.percentile(d, 5))
            out["grid_step"] = round(step, 6)
            out["eff_bits"] = round(math.log2((out["p999"] - out["p001"]) / step + 1), 2)
            out["bits_used"] = round(math.log2((out["max"] - out["min"]) / step + 1), 2)
            out["distinct"] = int(u.size)
    st, ln, val = runs(x)
    if x.size and val.size:
        lo_, hi_ = val.min(), val.max()
        out["run_min"] = int(ln[val == lo_].max())
        out["run_max"] = int(ln[val == hi_].max())
    plate = ln >= max(1, int(0.5 * fs))
    if plate.any():
        lv = Counter()
        cnt = Counter()
        for v_, l_ in zip(val[plate].tolist(), ln[plate].tolist()):
            lv[v_] += l_ / fs
            cnt[v_] += 1
        out["plateaus"] = [[v_, cnt[v_], round(s_, 1)] for v_, s_ in lv.most_common(8)]
    flat = ln >= max(1, int(1.0 * fs))
    fl = ln[flat] / fs
    out["flat"] = {"runs": int(fl.size), "total_s": round(float(fl.sum()), 1),
                   "longest_s": round(float(fl.max()), 1) if fl.size else 0.0, "hist": hist(fl, FLAT_EDGES)}
    return out


def numeric(tr) -> dict:
    t = np.sort(np.asarray(tr.record_times, float))
    v = np.asarray(tr.values, float) if len(tr.values) else np.zeros(0)
    out = {"n": int(t.size)}
    if t.size < 3:
        return out
    dt = np.diff(t)
    med = float(np.median(dt))
    out.update({"span_s": round(float(t[-1] - t[0]), 1), "median_interval_s": round(med, 3),
                "interval_mult_hist": hist(dt / med if med > 0 else dt, NUM_MULT_EDGES),
                "gap_hist": hist(dt[dt > max(1.5 * med, 1e-9)], GAP_EDGES),
                "gap_longest_s": round(float(dt.max()), 1)})
    if v.size == t.size and v.size:
        fin = np.isfinite(v)
        if fin.any():
            vv = v[fin]
            out.update({"min": float(vv.min()), "max": float(vv.max()), "zeros": round(float((vv == 0).mean()), 4),
                        "nonfinite": round(float(1 - fin.mean()), 4)})
            o = np.argsort(np.asarray(tr.record_times, float), kind="stable")
            st, ln, val = runs(v[o])
            if ln.size:
                tt = np.asarray(tr.record_times, float)[o]
                en = np.r_[st[1:], v.size] - 1
                dur = tt[en] - tt[st]
                k = int(np.argmax(dur))
                out["longest_same_value_s"] = round(float(dur[k]), 1)
                out["longest_same_value"] = float(val[k]) if np.isfinite(val[k]) else None
    return out


def scan_case(root: Path, case: int) -> dict:
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: PLC0415

    v = read_vital(root / "vital_files" / f"{case:04d}.vital")
    res = {"case": case, "devices": {k: [d.type_name, d.port] for k, d in v.devices.items()}, "wave": {}, "num": {}}
    for n, tr in v.tracks.items():
        try:
            if tr.track_type == 1:
                res["wave"][n] = wave(tr)
            elif tr.track_type == 2:
                res["num"][n] = numeric(tr)
        except Exception as e:  # noqa: BLE001
            res.setdefault("errors", {})[n] = repr(e)[:120]
    return res


def scan(a) -> None:
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in open(out)} if out.exists() else set()
    t0 = time.time()
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        if time.time() - t0 > a.time_budget:
            break
        try:
            r = scan_case(Path(a.root), c)
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:200]}
        with open(out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan", "agg"])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=1e9)
    a = ap.parse_args()
    if a.mode == "scan":
        scan(a)
    else:
        from signal_survey_agg import agg  # noqa: PLC0415
        agg(a.inputs, Path(a.out))


if __name__ == "__main__":
    main()
