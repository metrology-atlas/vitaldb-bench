# SPDX-License-Identifier: MIT
"""Record-level cache of every track in every case: message sizes, every step between records (gaps, overlaps and
clock steps, both directions), conversion factors, and for numeric tracks every interval, gap, unchanged run and
value. For aggregation later. This machine only.

    python tools/record_scan.py scan --root R --cases 1,2,3 --out DIR [--time-budget S]

Writes DIR/<case>.npz; keys are '<track with / as __>|<field>'. Times are seconds from 2100-01-01 00:00:00 UTC.
Waveform tracks (type 1), records in file order:
  rec_t0              time of the first record
  size_v, size_n      record sizes (samples) run-length encoded: value and run length, in file order
  step_i, step_s      every record boundary whose step is not zero: index k of the record before it, and
                      step = start(k+1) - (start(k) + size(k)/fs) in s (positive: gap; negative: overlap or backward
                      clock step); boundaries with |step| < 1 us are omitted (contiguous)
  meta                json: fs, gain, offset (physical = code * gain + offset), unit, mindisp, maxdisp, format,
                      device type and port, records, samples, records out of time order
Numeric tracks (type 2):
  iv_ms, iv_n         every interval between consecutive stamps (time-sorted), rounded to 1 ms: value and count
  gap_t, gap_d        every interval longer than twice the median interval: start and duration (s)
  run_t, run_d, run_n, run_v   every run of an unchanged value over 3 or more consecutive records: start,
                      duration (s), number of records, value
  val_v, val_n        every distinct value and how many records have it
  meta                json: unit, mindisp, maxdisp, format, device, records, median interval, duplicate stamps
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ORIGIN = 4102444800.0
FMT = {1: "float32", 2: "float64", 3: "int8", 4: "uint8", 5: "int16", 6: "uint16", 7: "int32", 8: "uint32"}


def rle(x):
    x = np.asarray(x)
    if x.size == 0:
        return x, np.zeros(0, int)
    ch = np.flatnonzero(np.diff(x) != 0) + 1
    st = np.r_[0, ch]
    return x[st], np.diff(np.r_[st, x.size])


def dev_of(v, tr):
    d = v.devices.get(getattr(tr, "device", None)) if hasattr(v, "devices") else None
    if d is None:
        name = tr.name.split("/")[0] if "/" in getattr(tr, "name", "") else None
        d = next((dd for dd in v.devices.values() if dd.type_name == name or getattr(dd, "name", None) == name), None)
    return [getattr(d, "type_name", None), getattr(d, "port", None)] if d is not None else [None, None]


def wave(v, n, tr) -> dict:
    fs = float(tr.srate)
    rt = np.asarray(tr.record_times, float)
    sz = np.asarray(tr.record_sizes, int)
    out = {}
    if rt.size == 0:
        return out
    step = rt[1:] - (rt[:-1] + sz[:-1] / fs)
    nz = np.flatnonzero(np.abs(step) >= 1e-6)
    sv, sn = rle(sz)
    out.update({"rec_t0": np.array(rt[0] - ORIGIN), "size_v": sv.astype(np.int32), "size_n": sn.astype(np.int32),
                "step_i": nz.astype(np.int32), "step_s": step[nz]})
    out["meta"] = np.array(json.dumps({
        "fs": fs, "gain": float(tr.gain), "offset": float(tr.offset), "unit": str(getattr(tr, "unit", "")),
        "mindisp": float(getattr(tr, "mindisp", np.nan)), "maxdisp": float(getattr(tr, "maxdisp", np.nan)),
        "format": FMT.get(int(tr.fmt), str(tr.fmt)), "device": dev_of(v, tr), "records": int(rt.size),
        "samples": int(sz.sum()), "out_of_order": int((np.diff(rt) < 0).sum())}))
    return out


def numeric(v, n, tr) -> dict:
    t = np.asarray(tr.record_times, float)
    x = np.asarray(tr.values, float) if len(tr.values) else np.zeros(0)
    out = {}
    if t.size == 0:
        return out
    o = np.argsort(t, kind="stable")
    t = t[o]
    x = x[o] if x.size == t.size else x
    dt = np.diff(t)
    if dt.size:
        ms, cnt = np.unique(np.round(dt * 1000).astype(np.int64), return_counts=True)
        out["iv_ms"], out["iv_n"] = ms, cnt.astype(np.int64)
        med = float(np.median(dt))
        g = np.flatnonzero(dt > 2 * med) if med > 0 else np.zeros(0, int)
        out["gap_t"], out["gap_d"] = t[g] - ORIGIN, dt[g]
    else:
        med = None
    if x.size == t.size and x.size:
        fin = np.isfinite(x)
        ch = np.flatnonzero((x[1:] != x[:-1]) | ~fin[1:] | ~fin[:-1]) + 1
        st = np.r_[0, ch]
        ln = np.diff(np.r_[st, x.size])
        en = st + ln - 1
        m = (ln >= 3) & fin[st]
        out["run_t"], out["run_d"], out["run_n"], out["run_v"] = t[st[m]] - ORIGIN, t[en[m]] - t[st[m]], ln[m].astype(np.int32), x[st[m]]
        vv, vn = np.unique(x[fin], return_counts=True)
        out["val_v"], out["val_n"] = vv, vn.astype(np.int64)
    out["meta"] = np.array(json.dumps({
        "unit": str(getattr(tr, "unit", "")), "mindisp": float(getattr(tr, "mindisp", np.nan)),
        "maxdisp": float(getattr(tr, "maxdisp", np.nan)), "format": FMT.get(int(tr.fmt), str(tr.fmt)),
        "device": dev_of(v, tr), "records": int(t.size), "median_interval": med,
        "duplicate_stamps": int((dt == 0).sum()) if dt.size else 0}))
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
            arrays["devices"] = np.array(json.dumps({k: [d.type_name, d.port] for k, d in v.devices.items()}))
            for n, tr in v.tracks.items():
                fn = wave if tr.track_type == 1 else numeric if tr.track_type == 2 else None
                if fn is None:
                    continue
                try:
                    for k, val in fn(v, n, tr).items():
                        arrays[n.replace("/", "__") + "|" + k] = val
                except Exception as e:  # noqa: BLE001
                    arrays[n.replace("/", "__") + "|error"] = np.array(repr(e)[:200])
        except Exception as e:  # noqa: BLE001
            arrays["error"] = np.array(repr(e)[:200])
        tmp = od / f"{c}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.rename(f)


if __name__ == "__main__":
    main()
