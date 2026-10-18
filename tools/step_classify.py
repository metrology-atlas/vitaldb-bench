# SPDX-License-Identifier: MIT
"""Classify every step between waveform records: lost data, resynchronisation, duplicate or interleave. This machine only.

    python tools/step_classify.py scan --root R --cases 1,2,3 --out DIR [--time-budget S]

Writes DIR/<case>.parquet, one row per non-zero step (|step| >= 1 us) between consecutive records of a waveform track
(capped at 3,000 backward steps per track: the duplicate / interleave recordings have hundreds of thousands).
Columns:
  case, track, device, fs, k, t (end of record k, s from the case origin), step (s), msg (s, the track's commonest
    record duration), steps_msg (step / msg)
  remainder      forward steps longer than one message: (step mod msg) / msg, folded to [-0.5, 0.5); 0 = whole messages
  identical      record k+1 has exactly the same samples as record k
  joins_k_1      record k+1 continues record k-1 better than record k (the interleave signature)
  join_z         continuity across the step: |first sample of k+1 - linear extrapolation of the last 3 samples of k|,
                 in units of the median absolute one-step prediction error inside record k; small (<~3) means the
                 waveform carries on as if no samples were missing
  other_devices  number of other devices (of SNUADC, Primus, BIS, CardioQ) with a forward step > 1 s within 1 s of this one
  devices_present
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ORIGIN = 4102444800.0
DEVICES = ("SNUADC", "Primus", "BIS", "CardioQ")


def track_steps(n, tr):
    fs = float(tr.srate)
    rt = np.asarray(tr.record_times, float)
    sz = np.asarray(tr.record_sizes, int)
    if rt.size < 3:
        return None
    vals, cnt = np.unique(sz, return_counts=True)
    msg = vals[np.argmax(cnt)] / fs
    step = rt[1:] - (rt[:-1] + sz[:-1] / fs)
    idx = np.flatnonzero(np.abs(step) >= 1e-6)
    back = idx[step[idx] < 0]
    if back.size > 3000:
        idx = np.sort(np.r_[idx[step[idx] >= 0], back[:3000]])
    x = np.asarray(tr.values)
    num = np.issubdtype(x.dtype, np.number)
    x = x.astype(float) if num else None
    st = np.r_[0, np.cumsum(sz)[:-1]]
    rows = []
    for k in idx:
        s = float(step[k])
        r = dict(track=n, fs=fs, k=int(k), t=float(rt[k] + sz[k] / fs - ORIGIN), step=s, msg=float(msg), steps_msg=s / msg)
        r["remainder"] = float(((s / msg) + 0.5) % 1.0 - 0.5) if s > msg else np.nan
        if x is not None:
            a = x[st[k]:st[k] + sz[k]]
            b = x[st[k + 1]:st[k + 1] + sz[k + 1]]
            r["identical"] = bool(a.size == b.size and np.array_equal(a, b))
            if k >= 1 and a.size >= 4 and b.size >= 1:
                p = x[st[k - 1]:st[k - 1] + sz[k - 1]]
                r["joins_k_1"] = bool(p.size and abs(b[0] - p[-1]) < abs(b[0] - a[-1]))
                pred = 2 * a[-1] - a[-2]
                e = np.abs(a[2:] - (2 * a[1:-1] - a[:-2]))
                scale = np.median(e) if e.size and np.median(e) > 0 else (np.mean(e) if e.size and np.mean(e) > 0 else 1.0)
                r["join_z"] = float(abs(b[0] - pred) / scale)
        rows.append(r)
    return rows


def scan_case(root, case):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: PLC0415

    v = read_vital(Path(root) / "vital_files" / f"{case:04d}.vital")
    rows = []
    for n, tr in v.tracks.items():
        if tr.track_type != 1 or "#" in n:
            continue
        dev = n.split("/")[0]
        rr = track_steps(n, tr)
        if rr:
            for r in rr:
                r["device"] = dev
            rows.extend(rr)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.insert(0, "case", case)
    big = df[df.step > 1.0][["device", "t"]]
    present = set(df.device) | {n.split("/")[0] for n, tr in v.tracks.items() if tr.track_type == 1}
    od = np.zeros(len(df), int)
    for dv in DEVICES:
        tt = np.sort(big.t[big.device == dv].to_numpy())
        if tt.size == 0:
            continue
        j = np.searchsorted(tt, df.t.to_numpy())
        near = np.zeros(len(df), bool)
        for jj in (j - 1, j):
            ok = (jj >= 0) & (jj < tt.size)
            near[ok] |= np.abs(tt[jj[ok]] - df.t.to_numpy()[ok]) <= 1.0
        od += near & (df.device.to_numpy() != dv)
    df["other_devices"] = od
    df["devices_present"] = ",".join(sorted(d for d in present if d in DEVICES))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=1e9)
    a = ap.parse_args()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for c in [int(x) for x in a.cases.split(",") if x]:
        f = od / f"{c}.parquet"
        if f.exists():
            continue
        if time.time() - t0 > a.time_budget:
            break
        try:
            df = scan_case(a.root, c)
        except Exception as e:  # noqa: BLE001
            df = pd.DataFrame([{"case": c, "track": "error", "device": repr(e)[:200]}])
        if df.empty:
            df = pd.DataFrame([{"case": c, "track": "none"}])
        tmp = od / f"{c}.tmp.parquet"
        df.to_parquet(tmp)
        tmp.rename(f)


if __name__ == "__main__":
    main()
