# SPDX-License-Identifier: MIT
"""Every non-zero step between waveform records, from the tools/record_scan.py cache, as one table per part.

    python tools/record_steps_collect.py SRC --part I --of N --out DIR

Columns: case, track, device, fs, k (record index before the step), t (s from the case origin, end of record k), step (s;
positive = gap, negative = overlap / backward), size_before, size_after (samples), msg (the track's commonest record size
in samples).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    src = Path(sys.argv[1]); a = dict(zip(sys.argv[2::2], sys.argv[3::2]))
    i, n, out = int(a["--part"]), int(a["--of"]), Path(a["--out"])
    rows = []
    for f in sorted(src.glob("*.npz")):
        if not f.stem.isdigit() or int(f.stem) % n != i:
            continue
        z = np.load(f)
        for k in z.files:
            if not k.endswith("|step_s"):
                continue
            tr = k[:-7]
            st = z[k]
            if st.size == 0:
                continue
            idx = z[tr + "|step_i"]
            m = json.loads(str(z[tr + "|meta"]))
            fs = m["fs"]
            sizes = np.repeat(z[tr + "|size_v"], z[tr + "|size_n"])
            vals, cnt = np.unique(sizes, return_counts=True)
            msg = int(vals[np.argmax(cnt)])
            steps_full = np.zeros(sizes.size)
            steps_full[idx + 1] = st                     # step before record idx+1
            start = float(z[tr + "|rec_t0"]) + np.r_[0, np.cumsum(sizes[:-1])] / fs + np.cumsum(steps_full)
            end_k = start[idx] + sizes[idx] / fs
            for j, s_ in enumerate(st):
                kk = int(idx[j])
                rows.append((int(f.stem), tr, m["device"][0], fs, kk, float(end_k[j]), float(s_), int(sizes[kk]),
                             int(sizes[kk + 1]) if kk + 1 < sizes.size else -1, msg))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["case", "track", "device", "fs", "k", "t", "step", "size_before", "size_after", "msg"]).to_parquet(out / f"steps_{i}.parquet")
    print(i, len(rows))


if __name__ == "__main__":
    main()
