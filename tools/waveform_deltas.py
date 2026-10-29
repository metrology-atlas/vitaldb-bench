# SPDX-License-Identifier: MIT
"""Waveform delta scan (release-ready). This machine only; per-recording output stays in the work folder.

    python delta_scan.py ROOT OUT.jsonl PART NPARTS [--cases 1,161,...] [--minutes 30]

Per waveform track (>= 20 Hz), on up to --minutes of the longest stretch of contiguous records (within records):
- holds: zero-delta share; 'sandwiched' zero share (a zero delta between two deltas of >= 3 codes, which smooth
  sampling cannot produce); run lengths of held values (1..6+);
- rhythms: the three strongest lines above 20 Hz (0.01 Hz resolution) in the zero-delta indicator and in the
  zero-second-difference indicator, each with its strength over the local median;
- equal-delta runs (>= 3 equal non-zero deltas): share of deltas in them, split by |delta| = 1 and >= 2, and the
  run-length histogram for |delta| >= 2;
- float tracks: the quantum (smallest non-zero |delta|) and share of deltas that are whole multiples of it;
- value range and the share of samples at each end.

Usage: python tools/waveform_deltas.py ROOT OUT.jsonl PART NPARTS [--cases 1,161,...] [--minutes 30]

Whole release 28 Oct 2026: 8 parts, about 70 min in WSL.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy import signal
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

def longest_run(t, max_s):
    rt, rs, fs = t.record_times, np.asarray(t.record_sizes), float(t.srate)
    if rt.size < 2: return None
    ok = np.abs(np.diff(rt) - rs[:-1] / fs) < 0.5 / fs
    best = cur = bs = cs = 0
    for i, g in enumerate(ok):
        cur = cur + 1 if g else 0
        if cur == 1: cs = i
        if cur > best: best, bs = cur, cs
    starts = np.r_[0, np.cumsum(rs)[:-1]]
    a, b = starts[bs], starts[bs + best] + rs[bs + best]
    return np.asarray(t.values[a:b], float)[: int(max_s * fs)], fs

def lines(ind, fs):
    ind = ind - ind.mean()
    if ind.size < fs * 120 or not np.any(ind): return []
    f, P = signal.welch(ind, fs, nperseg=min(ind.size, int(fs * 100)), noverlap=0)
    m = f > 5; f, P = f[m], P[m]
    if P.size < 50: return []
    med = signal.medfilt(P, 201) + 1e-30; r = P / med
    pk, _ = signal.find_peaks(r, distance=max(1, int(0.5 / (f[1] - f[0]))))
    pk = pk[np.argsort(r[pk])[::-1][:3]]
    return [[round(float(f[i]), 2), round(float(r[i]), 1)] for i in pk]

def stats(x, fs, integer):
    x = x[np.isfinite(x)]
    if x.size < fs * 120: return None
    d = np.diff(x); z = d == 0
    out = {"n": int(d.size), "zero_share": round(float(z.mean()), 4)}
    big = np.abs(d) >= (3 if integer else 3 * max(np.min(np.abs(d[d != 0])) if np.any(d) else 1, 1e-12))
    sand = z[1:-1] & big[:-2] & big[2:]
    out["sandwiched_zero_share"] = round(float(sand.sum() / max(1, (big[:-2] & big[2:]).sum())), 4)
    out["steep_pairs"] = int((big[:-2] & big[2:]).sum())
    same = np.r_[False, d[1:] == d[:-1]]
    h = np.diff(np.flatnonzero(np.diff(np.r_[0, z.astype(np.int8), 0])))[::2] + 1 if z.any() else np.array([], int)
    out["held_runs_1_to_6plus"] = np.bincount(np.minimum(h, 6), minlength=7)[1:].tolist() if h.size else []
    out["zero_delta_lines"] = lines(z.astype(float), fs)
    d2z = np.diff(d) == 0
    out["zero_curvature_lines"] = lines((d2z & (d[1:] != 0)).astype(float), fs)
    eq = (d[1:] == d[:-1]) & (d[1:] != 0)
    st = np.flatnonzero(np.diff(np.r_[0, eq.astype(np.int8), 0]) == 1); en = np.flatnonzero(np.diff(np.r_[0, eq.astype(np.int8), 0]) == -1)
    L = en - st + 1; val = np.abs(d[st])
    k = L >= 2                                              # >= 3 equal deltas
    q = np.min(np.abs(d[d != 0])) if np.any(d) else 1.0
    one = np.isclose(val, q)
    out["equal_runs_share_step1"] = round(float(np.sum(L[k & one] + 1) / d.size), 5)
    out["equal_runs_share_step2plus"] = round(float(np.sum(L[k & ~one] + 1) / d.size), 5)
    out["equal_runs_len_hist_step2plus_3to8plus"] = np.bincount(np.minimum(L[k & ~one] + 1, 8), minlength=9)[3:].tolist()
    # null for equal-delta runs: the same signal smoothed and re-rounded (a smooth curve sampled and quantised)
    if integer and x.size > 1000:
        xs = np.round(signal.savgol_filter(x, 7, 3))
        ds = np.diff(xs); eqs = (ds[1:] == ds[:-1]) & (ds[1:] != 0) & (np.abs(ds[1:]) > q)
        out["equal_step2plus_pairs_share"] = round(float(np.mean((d[1:] == d[:-1]) & (np.abs(d[1:]) > q))), 5)
        out["equal_step2plus_pairs_share_smoothed_null"] = round(float(np.mean(eqs)), 5)
    if not integer:
        nz = np.abs(d[d != 0]); out["quantum"] = float(nz.min()) if nz.size else None
        # lattice search: the coarsest step g (0.2 down to 0.02) for which the most non-zero deltas lie within 2% of a multiple
        sub = nz[:200000]
        best = (0.0, None)
        for g in np.linspace(0.2, 0.02, 3601):
            r = sub / g; share = float(np.mean(np.abs(r - np.round(r)) < 0.02))
            if share > best[0]: best = (share, float(g))
        out["lattice_step"] = round(best[1], 6) if best[1] else None; out["lattice_share"] = round(best[0], 4)
        # null for the lattice share: same search on deltas jittered by a uniform +-half step
        out["lattice_share_null"] = round(float(np.mean(np.abs((sub + np.random.default_rng(0).uniform(-0.5, 0.5, sub.size) * best[1]) / best[1] - np.round((sub + np.random.default_rng(0).uniform(-0.5, 0.5, sub.size) * best[1]) / best[1])) < 0.02)), 4) if best[1] else None
    vmin, vmax = float(x.min()), float(x.max())
    out.update(vmin=vmin, vmax=vmax, at_vmin=round(float(np.mean(x == vmin)), 5), at_vmax=round(float(np.mean(x == vmax)), 5))
    return out

ap = argparse.ArgumentParser(); ap.add_argument("root", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("part", type=int); ap.add_argument("nparts", type=int); ap.add_argument("--cases"); ap.add_argument("--minutes", type=float, default=30)
a = ap.parse_args()
files = [int(c) for c in a.cases.split(",")] if a.cases else sorted(int(p.stem) for p in (a.root / "vital_files").glob("*.vital"))
mine = [c for i, c in enumerate(files) if i % a.nparts == a.part]
done = {json.loads(l)["case"] for l in a.out.read_text().splitlines()} if a.out.exists() else set()
for c in mine:
    if c in done: continue
    t0 = time.time(); res = {"case": c, "tracks": {}}
    try:
        v = read_vital(a.root / "vital_files" / f"{c:04d}.vital")
        for k, t in v.tracks.items():
            if t.srate and t.srate >= 20 and t.record_times.size:
                r = longest_run(t, a.minutes * 60)
                if r is None: continue
                s = stats(r[0], r[1], bool(getattr(t, "is_integer_format", False)))
                if s: s["srate"] = float(t.srate); res["tracks"][k] = s
    except Exception as e:
        res["error"] = repr(e)[:200]
    res["s"] = round(time.time() - t0, 1)
    with a.out.open("a") as f: f.write(json.dumps(res) + "\n")
