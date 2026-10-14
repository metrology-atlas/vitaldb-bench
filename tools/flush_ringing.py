# SPDX-License-Identifier: MIT
"""Fast-flush (square-wave) response of the arterial line at every flush release. Per case, this machine only.

    python tools/flush_ringing.py --root R --cases 1,2 --out out.jsonl

For each "high" event of lineaccess (the flush device holding the line at bag pressure), the release is the
steepest fall within 1 s of the event's end, after a plateau > 180 mmHg. From the release:
- fall_ms: time from 80% to 20% of the drop from plateau to the settled level b (median of 150-300 ms after);
- ringing: an undershoot below b by > 5 mmHg within 80 ms. Then the first undershoot (m1), the next peak (p1)
  and the next trough (m2): damped period from m1 to m2 (else 2 x m1->p1), half-cycle decrement
  r = (p1 - b) / (b - m1), damping coefficient zeta = -ln r / sqrt(pi^2 + ln^2 r), natural frequency
  fn = fd / sqrt(1 - zeta^2);
- no undershoot: overdamped, or a slow release (the release speed cannot be told from the waveform).
Also MAP, SBP, DBP over the 20 s before the flush's access and after its recovery when clean (from lineaccess).
The ART samples are the monitor's analogue output digitised at 500 Hz, so the monitor's own filtering is part of
the response.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lineaccess"))
from lineaccess import line_accesses, line_events
from pat import sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

FS = 500.0


def response(t, x, e_end):
    i0, i1 = np.searchsorted(t, e_end - 1.0), np.searchsorted(t, e_end + 1.0)
    seg = x[i0:i1]
    if seg.size < 400 or not np.all(np.isfinite(seg)):
        return None
    sm = np.convolve(seg, np.ones(3) / 3, "same")
    j = int(np.argmin(np.diff(sm)))
    if j < 75 or j + 200 > seg.size:
        return None
    plateau = float(np.median(seg[j - 75: j - 15]))
    b = float(np.median(seg[j + 75: j + 150]))
    if plateau < 180 or b > 160 or b < 20 or plateau - b < 60:
        return None
    drop = plateau - b
    k = seg[j - 25: j + 75]
    i80 = np.flatnonzero(k <= b + 0.8 * drop)
    i20 = np.flatnonzero(k <= b + 0.2 * drop)
    fall = (i20[0] - i80[0]) / FS * 1000 if i80.size and i20.size else None
    out = {"t": round(float(t[i0 + j]), 3), "plateau": round(plateau, 1), "settled": round(b, 1), "fall_ms": fall}
    w = seg[j: j + 150]                               # 300 ms after the release
    m1 = int(np.argmin(w[:40]))
    if b - w[m1] <= 5:
        out["ringing"] = False
        return out
    p1 = m1 + int(np.argmax(w[m1: m1 + 40]))
    m2 = p1 + int(np.argmin(w[p1: p1 + 40])) if p1 + 2 < w.size else None
    a1, a2 = b - w[m1], w[p1] - b
    out.update({"ringing": True, "undershoot": round(float(a1), 1), "overshoot": round(float(a2), 1)})
    if a2 <= 0.5 or p1 == m1:
        out["zeta"] = 1.0
        return out
    r = min(a2 / a1, 0.999)
    lr = np.log(r)
    zeta = float(-lr / np.sqrt(np.pi ** 2 + lr ** 2))
    T = (m2 - m1) / FS if m2 is not None and m2 > p1 and b - w[m2] > 0 else 2 * (p1 - m1) / FS
    fd = 1 / T if T > 0 else None
    out.update({"zeta": round(zeta, 3), "fd_hz": None if fd is None else round(fd, 1),
                "fn_hz": None if fd is None else round(fd / np.sqrt(max(1 - zeta ** 2, 1e-3)), 1)})
    return out


def window(t, x, a0, a1):
    i, j = np.searchsorted(t, a0), np.searchsorted(t, a1)
    seg = x[i:j]
    seg = seg[np.isfinite(seg)]
    n = seg.size // 1000
    if n < 5:
        return None
    w = seg[: n * 1000].reshape(n, 1000)
    sbp, dbp = float(np.median(np.percentile(w, 95, axis=1))), float(np.median(np.percentile(w, 5, axis=1)))
    if sbp - dbp < 10 or not 30 <= np.mean(seg) <= 160:
        return None
    return [round(float(np.mean(seg)), 1), round(sbp, 1), round(dbp, 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in open(a.out)}
    with open(a.out, "a") as fh:
        for c in [int(v) for v in a.cases.split(",") if v]:
            if c in done:
                continue
            res = {"case": c, "flushes": []}
            try:
                tr = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={"SNUADC/ART"}).tracks.get("SNUADC/ART")
                if tr is None or not tr.record_times.size:
                    raise ValueError("no ART")
                t = sample_times(tr)
                x = tr.physical().astype(np.float64)
                ev = line_events(x, FS, t=t)
                acc = line_accesses(x, FS, t=t)
                busy = [(q.start, q.end if q.kind == "unavailable" else q.recovered) for q in acc]
                for q in acc:
                    if q.kind != "access" or "high" not in q.kinds:
                        continue
                    others = [bz for bz in busy if bz[0] != q.start]
                    pre = post = None
                    if not any(s < q.start - 5 and e > q.start - 25 for s, e in others):
                        pre = window(t, x, q.start - 25, q.start - 5)
                    if not any(s < q.recovered + 25 and e > q.recovered + 5 for s, e in others):
                        post = window(t, x, q.recovered + 5, q.recovered + 25)
                    rel = [response(t, x, e.end) for e in ev if "high" in e.kinds and q.start - 1 <= e.start and e.end <= q.end + 1]
                    rel = [r for r in rel if r]
                    res["flushes"].append({"start": round(q.start, 1), "recovered": round(q.recovered, 1), "kinds": "+".join(q.kinds),
                                           "pre": pre, "post": post, "releases": rel})
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(res) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
