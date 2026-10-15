# SPDX-License-Identifier: MIT
"""Arterial pressure before and after line accesses, to find recalibration jumps at zeroing. Per case, this
machine only; aggregates leave.

    python tools/art_zero_jumps.py --root R --cases 1,2 --out out.jsonl

Per case (SNUADC/ART in mmHg; lineaccess defaults):
- every access, with its kinds; for each "low" (open to air) event inside it, the air reading: 1-s medians on the open-to-air plateau (|p| < 15 mmHg),
  first and last, and the step between them (zeroing while open to air shows
  as a step from the old offset to 0);
- MAP (mean), SBP and DBP (medians of 2-s 95th/5th percentiles) and the pulse width at 50% height (median
  per beat, also relative to the beat interval) over [start-guard-win, start-guard] and
  [recovered+guard, recovered+guard+win] (defaults win 60 s, guard 10 s), both windows clear of any other access, occlusion or unavailable stretch;
- the same for occlusions (no recalibration expected) and for random clean pairs of windows separated by an
  access-like gap (natural change);
- with --minutes-out / --nibp-out, the per-minute ART series and the NIBP readings with their ART windows, in the
  formats of tools/art_minute_series.py and tools/nibp_scan.py --art (one read of the file for all three);
- NIBP (Solar8000/NIBP_MBP) within 10 min before the access and within 10 min after recovery: ART MAP over
  +-30 s around it minus NIBP MAP, before and after.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from lineaccess import line_accesses, line_events
from pat import sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--cases", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--win", type=float, default=60.0, help="length of the before/after windows (s)")
ap.add_argument("--guard", type=float, default=10.0, help="gap between a window and the event (s)")
ap.add_argument("--minutes-out", default=None, help="also write the per-minute ART series (as art_minute_series.py)")
ap.add_argument("--nibp-out", default=None, help="also write the NIBP readings with ART windows (as nibp_scan.py --art)")
a = ap.parse_args()


def stats(t, x, a0, a1):
    i, j = np.searchsorted(t, a0), np.searchsorted(t, a1)
    seg = x[i:j]
    seg = seg[np.isfinite(seg)]
    if seg.size < 0.8 * 500 * (a1 - a0):
        return None
    n = seg.size // 1000
    if n < 5:
        return None
    w_ = seg[: n * 1000].reshape(n, 1000)
    if not (40 <= np.mean(seg) <= 150) or np.median(np.percentile(w_, 95, axis=1) - np.percentile(w_, 5, axis=1)) < 10:
        return None                                   # not a pulsatile arterial pressure
    w = seg[: n * 1000].reshape(n, 1000)
    out = {"map": float(np.mean(seg)), "sbp": float(np.median(np.percentile(w, 95, axis=1))),
           "dbp": float(np.median(np.percentile(w, 5, axis=1)))}
    out.update(pulse_width(seg, 500.0))
    return out


def pulse_width(seg, fs):
    """Median per-beat pulse width at 50% of the pulse height (foot to systolic peak), the beat interval, and
    their ratio (width / RR), from systolic peaks in the window."""
    pk, _ = signal.find_peaks(seg, distance=int(0.3 * fs), prominence=10)
    W, RR = [], []
    for i, p in enumerate(pk):
        a = max(p - int(0.4 * fs), 0)
        f = a + int(np.argmin(seg[a:p + 1]))
        lev = seg[f] + 0.5 * (seg[p] - seg[f])
        left = f + int(np.argmax(seg[f:p + 1] >= lev)) if p > f else p
        b = min(p + int(0.8 * fs), seg.size - 1)
        after = np.flatnonzero(seg[p:b] < lev)
        if after.size == 0 or left <= f:
            continue
        r = p + after[0]
        # linear interpolation at both crossings
        lx = left - 1 + (lev - seg[left - 1]) / (seg[left] - seg[left - 1]) if seg[left] != seg[left - 1] else left
        rx = r - 1 + (seg[r - 1] - lev) / (seg[r - 1] - seg[r]) if seg[r] != seg[r - 1] else r
        W.append((rx - lx) / fs)
        if i:
            RR.append((p - pk[i - 1]) / fs)
    if len(W) < 5:
        return {"pw50_ms": None, "rr_ms": None, "pw50_rr": None}
    w_, rr = float(np.median(W)), (float(np.median(RR)) if RR else None)
    return {"pw50_ms": round(w_ * 1000, 1), "rr_ms": None if rr is None else round(rr * 1000, 1),
            "pw50_rr": None if rr is None else round(w_ / rr, 4)}


def clear(busy, a0, a1):
    return not any(s < a1 and e > a0 for s, e in busy)


def diff(b, c):
    if b is None or c is None:
        return None
    return {k: (round(c[k] - b[k], 4) if b[k] is not None and c[k] is not None else None) for k in b}


def write_minutes(c, t, x, acc, busy):
    """Per-minute ART (as tools/art_minute_series.py)."""
    bz = np.array(busy) if busy else np.empty((0, 2))
    rows = []
    for m in np.arange(np.floor(t[0] / 60) * 60, t[-1], 60):
        if bz.size and np.any((bz[:, 0] < m + 60) & (bz[:, 1] > m)):
            continue
        i, j = np.searchsorted(t, m), np.searchsorted(t, m + 60)
        seg = x[i:j]
        seg = seg[np.isfinite(seg)]
        if seg.size < 0.8 * 500 * 60:
            continue
        n = seg.size // 1000
        w = seg[: n * 1000].reshape(n, 1000)
        sbp, dbp, mp = float(np.median(np.percentile(w, 95, axis=1))), float(np.median(np.percentile(w, 5, axis=1))), float(np.mean(seg))
        if sbp - dbp < 10 or not 30 <= mp <= 160:
            continue
        rows.append([m, round(mp, 1), round(sbp, 1), round(dbp, 1)])
    with open(a.minutes_out, "a") as f:
        f.write(json.dumps({"case": c, "accesses": [[round(q.start, 1), round(q.end, 1), round(q.recovered, 1), "+".join(q.kinds)]
                                                   for q in acc if q.kind == "access"],
                            "occlusions": [[round(q.start, 1), round(q.recovered, 1)] for q in acc if q.kind == "occlusion"],
                            "minutes": rows}) + "\n")


def write_nibp(c, v, t, x, busy):
    """NIBP readings with clean ART over [t-40, t-5] (as tools/nibp_scan.py --art)."""
    from nibp_scan import readings  # noqa: PLC0415
    res = {"case": c}
    rd = readings(v)
    if rd is None:
        res["nibp"] = 0
    else:
        ts, V, t_first, t_last = rd
        res.update({"nibp": int(ts.size), "first": t_first, "span_s": round(t_last - t_first, 1),
                    "t": [round(q - t_first, 1) for q in ts], "v": V.round(0).astype(int).tolist()})
        art = []
        for tm in ts:
            a0, a1 = tm - 40, tm - 5
            if any(s_ < a1 and e_ > a0 for s_, e_ in busy):
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
            art.append([round(float(np.mean(seg)), 1), round(float(np.median(p95)), 1), round(float(np.median(p5)), 1), None, None])
        res["art"] = art
    with open(a.nibp_out, "a") as f:
        f.write(json.dumps(res) + "\n")


rng = np.random.default_rng(a.seed)
done = set()
if Path(a.out).exists():
    done = {json.loads(l)["case"] for l in open(a.out)}
with open(a.out, "a") as fh:
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        res = {"case": c, "accesses": [], "occlusions": [], "controls": [], "win": a.win, "guard": a.guard}
        try:
            v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital",
                           track_names={"SNUADC/ART", "Solar8000/NIBP_MBP", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP"})
            tr = v.tracks.get("SNUADC/ART")
            if tr is None or not tr.record_times.size:
                raise ValueError("no ART")
            t = sample_times(tr)
            res["art_start_abs"], res["art_end_abs"] = round(float(t[0]), 3), round(float(t[-1]), 3)
            x = tr.physical().astype(np.float64)
            fs = float(tr.srate)
            ev = line_events(x, fs, t=t)
            acc = line_accesses(x, fs, t=t)
            busy = [(q.start, q.end if q.kind == "unavailable" else q.recovered) for q in acc]
            if a.minutes_out:
                write_minutes(c, t, x, acc, busy)
            if a.nibp_out:
                write_nibp(c, v, t, x, busy)
            nb = v.tracks.get("Solar8000/NIBP_MBP")
            nt = nv = np.empty(0)
            if nb is not None and nb.record_times.size:
                nt = np.asarray(nb.record_times, dtype=float)
                nv = np.asarray(nb.physical(), dtype=float)[: nt.size]
                ok = np.isfinite(nv) & (nv > 20) & (nv < 200)
                nt, nv = nt[ok], nv[ok]

            def nibp_diff(t_lo, t_hi, last):
                k = np.flatnonzero((nt >= t_lo) & (nt <= t_hi))
                if not k.size:
                    return None
                k = k[-1] if last else k[0]
                s = stats(t, x, nt[k] - 30, nt[k] + 30)
                return None if s is None else round(s["map"] - nv[k], 2)

            durs = []
            for q in acc:
                if q.kind == "unavailable":
                    continue
                others = [b for b in busy if not (b[0] == q.start)]
                b0, b1 = q.start - a.guard - a.win, q.start - a.guard
                c0, c1 = q.recovered + a.guard, q.recovered + a.guard + a.win
                ok = clear(others, b0, b1) and clear(others, c0, c1)
                pre, post = (stats(t, x, b0, b1), stats(t, x, c0, c1)) if ok else (None, None)
                row = {"start_rel": round(q.start - t[0], 1), "dur": round(q.recovered - q.start, 1), "kinds": q.kinds,
                       "start_abs": round(q.start, 3), "end_abs": round(q.end, 3), "recovered_abs": round(q.recovered, 3),
                       "delta": diff(pre, post), "pre": pre and {k: (round(v_, 4) if v_ is not None else None) for k, v_ in pre.items()}}
                if q.kind == "access":
                    air = []
                    for e in ev:
                        if e.start >= q.start - 1 and e.end <= q.end + 1 and "low" in e.kinds:
                            i, j = np.searchsorted(t, e.start), np.searchsorted(t, e.end)
                            seg = x[i:j]
                            n = seg.size // int(fs)
                            if n >= 2:
                                m = np.median(seg[: n * int(fs)].reshape(n, int(fs)), axis=1)
                                m = m[np.abs(m) < 15]          # the open-to-air plateau only
                                if m.size >= 2:
                                    air.append({"dur": round(e.end - e.start, 1), "n": int(m.size), "first": round(float(m[0]), 2),
                                                "last": round(float(m[-1]), 2), "step": round(float(m[-1] - m[0]), 2)})
                    row["air"] = air
                    row["nibp_pre"] = nibp_diff(q.start - 600, q.start - 10, True)
                    row["nibp_post"] = nibp_diff(q.recovered + 10, q.recovered + 600, False)
                    res["accesses"].append(row)
                    durs.append(q.recovered - q.start)
                elif q.kind == "occlusion":
                    res["occlusions"].append(row)
            # random controls: same gap distribution as this case's accesses
            gaps = durs or [60.0]
            n_ok, tries = 0, 0
            while n_ok < 20 and tries < 400:
                tries += 1
                T = rng.uniform(t[0] + 100, t[-1] - 400)
                g = float(rng.choice(gaps))
                b0, b1, c0, c1 = T - a.guard - a.win, T - a.guard, T + g + a.guard, T + g + a.guard + a.win
                if not (clear(busy, b0, c1)):
                    continue
                d = diff(stats(t, x, b0, b1), stats(t, x, c0, c1))
                if d:
                    res["controls"].append(d)
                    n_ok += 1
        except Exception as ex:  # noqa: BLE001
            res["error"] = f"{type(ex).__name__}: {ex}"[:300]
        fh.write(json.dumps(res) + "\n")
        fh.flush()
