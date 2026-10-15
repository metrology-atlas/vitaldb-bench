# SPDX-License-Identifier: MIT
"""Non-invasive blood pressure (Solar8000 NIBP) readings in VitalDB, and their agreement with the arterial line.
Per case, this machine only; aggregates leave.

    python tools/nibp_scan.py --root R --cases 1,2 --out out.jsonl [--art]

The Solar8000 NIBP tracks repeat the last result every ~2 s, so a new measurement is taken where the
(SBP, DBP, MBP) triple changes (a repeat of exactly the same triple is missed). Per case: the reading times
(seconds from the first NIBP record) and values. With --art: SNUADC/ART over [t-40, t-5] s before each reading
(the cuff measures during the half-minute before its result), with windows touching a line access, occlusion or
unavailable stretch (lineaccess) left out: ART MAP, SBP (median 2-s 95th percentile), DBP (5th), and the
time since the last zero access and whether that zero was flagged suspect (|dDBP| > 7 mmHg, 20 s windows).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

N = ("Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Solar8000/NIBP_MBP")


def readings(v):
    tr = {n: v.tracks.get(n) for n in N}
    if any(t is None or not t.record_times.size for t in tr.values()):
        return None
    ts = np.asarray(tr[N[0]].record_times, dtype=float)
    vals = []
    for n in N:
        t = np.asarray(tr[n].record_times, dtype=float)
        x = np.asarray(tr[n].physical(), dtype=float)[: t.size]
        vals.append(np.interp(ts, t, x, left=np.nan, right=np.nan) if t.size != ts.size or not np.allclose(t, ts) else x)
    V = np.c_[vals].T
    ok = np.all(np.isfinite(V), axis=1) & (V[:, 0] > 30) & (V[:, 0] < 300) & (V[:, 1] > 10) & (V[:, 1] < V[:, 0])
    ts, V = ts[ok], V[ok]
    if ts.size == 0:
        return None
    new = np.r_[True, np.any(np.diff(V, axis=0) != 0, axis=1)]
    return ts[new], V[new], float(ts[0]), float(ts[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--art", action="store_true")
    ap.add_argument("--zero", default=None, help="art_zero_jumps jsonl (20 s) for the zero accesses")
    a = ap.parse_args()
    Z = {}
    if a.zero:
        for l in open(a.zero):
            r = json.loads(l)
            Z[r["case"]] = [(q["start_abs"], q["recovered_abs"], (q.get("delta") or {}).get("dbp"))
                            for q in r.get("accesses", []) if q.get("air") and "start_abs" in q]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in open(a.out)}
    with open(a.out, "a") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            res = {"case": c}
            try:
                names = set(N) | ({"SNUADC/ART"} if a.art else set())
                v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names=names)
                rd = readings(v)
                if rd is None:
                    res["nibp"] = 0
                else:
                    ts, V, t_first, t_last = rd
                    res.update({"nibp": int(ts.size), "first": t_first, "span_s": round(t_last - t_first, 1),
                                "t": [round(x - t_first, 1) for x in ts], "v": V.round(0).astype(int).tolist()})
                    tr = v.tracks.get("SNUADC/ART") if a.art else None
                    if tr is not None and tr.record_times.size:
                        from lineaccess import line_accesses  # noqa: PLC0415
                        from pat import sample_times  # noqa: PLC0415
                        t = sample_times(tr)
                        x = tr.physical().astype(np.float64)
                        acc = line_accesses(x, float(tr.srate), t=t)
                        busy = [(q.start, q.end if q.kind == "unavailable" else q.recovered) for q in acc]
                        zs = sorted(Z.get(c, []))
                        art = []
                        for tm in ts:
                            a0, a1 = tm - 40, tm - 5
                            if any(s < a1 and e > a0 for s, e in busy):
                                art.append(None)
                                continue
                            i, j = np.searchsorted(t, a0), np.searchsorted(t, a1)
                            seg = x[i:j]
                            seg = seg[np.isfinite(seg)]
                            n = seg.size // 1000
                            if seg.size < 0.8 * 500 * 35 or n < 5:
                                art.append(None)
                                continue
                            w = seg[: n * 1000].reshape(n, 1000)
                            p95, p5 = np.percentile(w, 95, axis=1), np.percentile(w, 5, axis=1)
                            if np.median(p95 - p5) < 10:
                                art.append(None)
                                continue
                            prev = [z for z in zs if z[1] <= tm]
                            art.append([round(float(np.mean(seg)), 1), round(float(np.median(p95)), 1), round(float(np.median(p5)), 1),
                                        None if not prev else round(tm - prev[-1][1], 0),
                                        None if not prev or prev[-1][2] is None else bool(abs(prev[-1][2]) > 7)])
                        res["art"] = art
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(res) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
