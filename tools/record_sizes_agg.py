# SPDX-License-Identifier: MIT
"""Message (record) sizes per waveform track, from the tools/record_scan.py cache. Aggregates only.

    python tools/record_sizes_agg.py SRC --part I --of N --out DIR      (one part; then 'merge')
    python tools/record_sizes_agg.py merge DIR
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def part(src, i, n, out):
    rec = defaultdict(Counter)      # track -> size (samples) -> records
    fs = {}
    dev = defaultdict(Counter)
    cases = Counter()
    for f in sorted(Path(src).glob("*.npz")):
        if not f.stem.isdigit() or int(f.stem) % n != i:
            continue
        z = np.load(f)
        for k in z.files:
            if k.endswith("|size_v"):
                tr = k[:-7]
                for v_, n_ in zip(z[k].tolist(), z[tr + "|size_n"].tolist()):
                    rec[tr][v_] += n_
                m = json.loads(str(z[tr + "|meta"]))
                fs.setdefault(tr, m["fs"])
                dev[tr][str(m["device"][0])] += 1
                cases[tr] += 1
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"sizes_{i}.json").write_text(json.dumps({tr: {"fs": fs[tr], "cases": cases[tr], "device": dict(dev[tr]),
                                                                 "sizes": dict(rec[tr])} for tr in rec}))


def merge(out):
    tot = defaultdict(Counter); fs = {}; cases = Counter(); dev = defaultdict(Counter)
    for p in sorted(Path(out).glob("sizes_*.json")):
        for tr, v in json.loads(p.read_text()).items():
            tot[tr].update({int(k): c for k, c in v["sizes"].items()})
            fs[tr] = v["fs"]; cases[tr] += v["cases"]; dev[tr].update(v["device"])
    res = {}
    for tr in sorted(tot, key=lambda t: -cases[t]):
        c = tot[tr]; n = sum(c.values())
        top = c.most_common(5)
        res[tr] = {"cases": cases[tr], "fs": fs[tr], "device": dev[tr].most_common(1)[0][0], "records": n,
                   "top_sizes": [[s, round(k / n, 4), round(s / fs[tr], 4) if fs[tr] else None] for s, k in top],
                   "distinct_sizes": len(c)}
    (Path(out) / "message_sizes.json").write_text(json.dumps(res, indent=1))
    for tr, r in res.items():
        if r["cases"] >= 20:
            print(tr, r["device"], r["fs"], "Hz,", r["cases"], "cases | size (samples), share, seconds:", r["top_sizes"][:3], "| distinct", r["distinct_sizes"])


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(sys.argv[2])
    else:
        a = dict(zip(sys.argv[2::2], sys.argv[3::2]))
        part(sys.argv[1], int(a["--part"]), int(a["--of"]), a["--out"])
