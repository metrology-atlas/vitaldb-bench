# SPDX-License-Identifier: MIT
"""Collect gaps, segments, per-case value summaries and conversion factors from the tools/signal_events.py cache
into one table per part (for fast analysis). This machine only.

    python tools/signal_events_collect.py SRC --part I --of N --out DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    src = Path(sys.argv[1])
    a = dict(zip(sys.argv[2::2], sys.argv[3::2]))
    part, of, out = int(a["--part"]), int(a["--of"]), Path(a["--out"])
    G, S, M = [], [], []
    for f in sorted(src.glob("*.npz")):
        if not f.stem.isdigit() or int(f.stem) % of != part:
            continue
        case = int(f.stem)
        z = np.load(f)
        tracks = sorted({k.split("|")[0] for k in z.files if "|" in k})
        for tr in tracks:
            if tr + "|meta" not in z.files:
                continue
            meta = json.loads(str(z[tr + "|meta"]))
            gt, gd = z[tr + "|gap_t"], z[tr + "|gap_d"]
            st, sd = z[tr + "|seg_t"], z[tr + "|seg_d"]
            for t_, d_ in zip(gt, gd):
                G.append((case, tr, float(t_), float(d_)))
            nseg = st.size
            for i, (t_, d_) in enumerate(zip(st, sd)):
                S.append((case, tr, float(t_), float(d_), i == 0, i == nseg - 1))
            med = None
            if tr + "|code" in z.files:
                c, n = z[tr + "|code"], z[tr + "|count"]
                cs = np.cumsum(n)
                med = float(c[np.searchsorted(cs, cs[-1] / 2)])
                p05 = float(c[np.searchsorted(cs, cs[-1] * 0.05)])
                p95 = float(c[np.searchsorted(cs, cs[-1] * 0.95)])
            else:
                p05 = p95 = None
            M.append((case, tr, meta["fs"], meta["gain"], meta["offset"], meta["format"], meta["n"], meta["span"], med, p05, p95))
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(G, columns=["case", "track", "t", "d"]).to_parquet(out / f"gaps_{part}.parquet")
    pd.DataFrame(S, columns=["case", "track", "t", "d", "first", "last"]).to_parquet(out / f"segs_{part}.parquet")
    pd.DataFrame(M, columns=["case", "track", "fs", "gain", "offset", "format", "n", "span", "median_code", "p05_code", "p95_code"]).to_parquet(out / f"meta_{part}.parquet")
    print(part, len(G), len(S), len(M))


if __name__ == "__main__":
    main()
