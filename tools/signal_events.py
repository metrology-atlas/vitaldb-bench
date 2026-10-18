# SPDX-License-Identifier: MIT
"""Fine-grained cache of signal events in every waveform track, for aggregation later. This machine only.

    python tools/signal_events.py scan --root R --cases 1,2,3 --out DIR [--time-budget S]

Writes DIR/<case>.npz. For each waveform track (key prefix: track name with '/' -> '__'):
  gap_t, gap_d      every gap between continuous segments: start (s from the case origin) and duration (s); records
                    are continuous when the next starts within 1.5 sample periods of the end of the previous
  seg_t, seg_d      every continuous segment: start and duration (s)
  flat_t, flat_d, flat_v   every run of identical consecutive samples lasting >= 50 ms (at least 3 samples): start,
                    duration (s, from the sample count) and level (raw code for integer tracks, value for float)
  ext_t, ext_d, ext_v      every run of >= 2 samples at the recording's minimum or maximum value (candidate clipping)
  code, count       every distinct raw value and how many samples have it (integer tracks: the stored code; float
                    tracks: the value on its grid, as float64), i.e. the full value histogram of the recording
  meta              json: srate, gain, offset, format, storage bits, samples, span
Times are seconds from 2100-01-01 00:00:00 UTC, the VitalDB case origin.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ORIGIN = 4102444800.0
FMT = {1: ("float32", 32), 2: ("float64", 64), 3: ("int8", 8), 4: ("uint8", 8), 5: ("int16", 16), 6: ("uint16", 16),
       7: ("int32", 32), 8: ("uint32", 32)}


def sample_times(rt, sizes, fs):
    """Start time of each sample (records placed at their stamps)."""
    rep = np.repeat(rt, sizes)
    first = np.repeat(np.cumsum(np.r_[0, sizes[:-1]]), sizes)
    return rep + (np.arange(rep.size) - first) / fs


def track_events(tr) -> dict:
    fs = float(tr.srate)
    rt = np.asarray(tr.record_times, float)
    sz = np.asarray(tr.record_sizes, int)
    x = np.asarray(tr.values)
    out = {}
    if rt.size == 0 or x.size == 0:
        return out
    o = np.argsort(rt, kind="stable")
    if not np.all(o == np.arange(o.size)):          # keep samples with their records if records are out of order
        starts = np.r_[0, np.cumsum(sz)[:-1]]
        x = np.concatenate([x[starts[k]:starts[k] + sz[k]] for k in o])
        rt, sz = rt[o], sz[o]
    end = rt + sz / fs
    run_end = np.maximum.accumulate(end)
    step = rt[1:] - run_end[:-1]
    brk = step > 1.5 / fs
    seg_t = np.r_[rt[0], rt[1:][brk]]
    seg_e = np.r_[run_end[:-1][brk], run_end[-1]]
    out["gap_t"] = (run_end[:-1][brk] - ORIGIN).astype(np.float64)
    out["gap_d"] = step[brk].astype(np.float32)
    out["seg_t"] = (seg_t - ORIGIN).astype(np.float64)
    out["seg_d"] = (seg_e - seg_t).astype(np.float32)
    integer = bool(tr.is_integer_format)
    v = x.astype(np.int64) if integer else x.astype(np.float64)
    fin = np.isfinite(v) if not integer else np.ones(v.size, bool)
    st_t = sample_times(rt, sz, fs) - ORIGIN
    ch = np.flatnonzero((v[1:] != v[:-1]) | ~fin[1:] | ~fin[:-1]) + 1
    rs = np.r_[0, ch]
    rl = np.diff(np.r_[rs, v.size])
    rv = v[rs]
    ok = fin[rs]
    m = ok & (rl >= max(3, int(np.ceil(0.05 * fs))))
    out["flat_t"], out["flat_d"], out["flat_v"] = st_t[rs[m]], (rl[m] / fs).astype(np.float32), rv[m]
    vv = v[fin]
    if vv.size:
        lo, hi = vv.min(), vv.max()
        m = ok & (rl >= 2) & ((rv == lo) | (rv == hi))
        out["ext_t"], out["ext_d"], out["ext_v"] = st_t[rs[m]], (rl[m] / fs).astype(np.float32), rv[m]
        code, cnt = np.unique(vv, return_counts=True)
        out["code"], out["count"] = code, cnt.astype(np.int64)
    name, bits = FMT.get(int(tr.fmt), (str(tr.fmt), None))
    out["meta"] = np.array(json.dumps({"fs": fs, "gain": float(tr.gain), "offset": float(tr.offset), "format": name,
                                       "storage_bits": bits, "n": int(x.size), "nonfinite": int((~fin).sum()),
                                       "span": float(run_end[-1] - rt[0])}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=1e9)
    a = ap.parse_args()
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: PLC0415

    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for c in [int(x) for x in a.cases.split(",") if x]:
        f = od / f"{c}.npz"
        if f.exists():
            continue
        if time.time() - t0 > a.time_budget:
            break
        arrays = {}
        try:
            v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital")
            for n, tr in v.tracks.items():
                if tr.track_type == 1:
                    for k, val in track_events(tr).items():
                        arrays[n.replace("/", "__") + "|" + k] = val
        except Exception as e:  # noqa: BLE001
            arrays["error"] = np.array(repr(e)[:200])
        tmp = od / f"{c}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.rename(f)


if __name__ == "__main__":
    main()
