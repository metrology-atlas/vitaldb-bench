# SPDX-License-Identifier: MIT
"""Per-minute arterial pressure series with line accesses, for trend analyses. Per case, this machine only.

    python tools/art_minute_series.py --root R --cases 1,2 --out out.jsonl

SNUADC/ART in mmHg; lineaccess defaults. For every whole minute clear of any access (incl. recovery),
occlusion or unavailable stretch and with a pulse: MAP (mean), SBP and DBP (medians of 2-s 95th/5th percentiles).
Also the accesses (start, end, recovered, kinds) and occlusions. Times are absolute (file stamps).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lineaccess"))
from lineaccess import line_accesses
from pat import sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--cases", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
done = set()
if Path(a.out).exists():
    done = {json.loads(l)["case"] for l in open(a.out)}
with open(a.out, "a") as fh:
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        res = {"case": c}
        try:
            tr = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={"SNUADC/ART"}).tracks.get("SNUADC/ART")
            if tr is None or not tr.record_times.size:
                raise ValueError("no ART")
            t = sample_times(tr)
            x = tr.physical().astype(np.float64)
            acc = line_accesses(x, float(tr.srate), t=t)
            busy = np.array([(q.start, q.end if q.kind == "unavailable" else q.recovered) for q in acc]) if acc else np.empty((0, 2))
            res["accesses"] = [[round(q.start, 1), round(q.end, 1), round(q.recovered, 1), "+".join(q.kinds)] for q in acc if q.kind == "access"]
            res["occlusions"] = [[round(q.start, 1), round(q.recovered, 1)] for q in acc if q.kind == "occlusion"]
            m0 = np.floor(t[0] / 60) * 60
            rows = []
            for m in np.arange(m0, t[-1], 60):
                if busy.size and np.any((busy[:, 0] < m + 60) & (busy[:, 1] > m)):
                    continue
                i, j = np.searchsorted(t, m), np.searchsorted(t, m + 60)
                seg = x[i:j]
                seg = seg[np.isfinite(seg)]
                if seg.size < 0.8 * 500 * 60:
                    continue
                n = seg.size // 1000
                w = seg[: n * 1000].reshape(n, 1000)
                p95, p5 = np.percentile(w, 95, axis=1), np.percentile(w, 5, axis=1)
                sbp, dbp, mp = float(np.median(p95)), float(np.median(p5)), float(np.mean(seg))
                if sbp - dbp < 10 or not 30 <= mp <= 160:
                    continue
                rows.append([m, round(mp, 1), round(sbp, 1), round(dbp, 1)])
            res["minutes"] = rows
        except Exception as ex:  # noqa: BLE001
            res["error"] = f"{type(ex).__name__}: {ex}"[:300]
        fh.write(json.dumps(res) + "\n")
        fh.flush()
