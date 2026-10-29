# SPDX-License-Identifier: MIT
"""NTP pilot, version 2 (frequent-mode SNUADC: every record stamped on PC arrival). This machine only.
Per minute on the SNUADC sample count: PC minus count = median(stamp - count time) of its records; grid minus count =
120 Hz mains phase (10 s blocks, unwrapped) / (2 pi 120). PC minus grid = their difference. Reports slope, residual,
and minute-to-minute jumps beyond the noise.

Usage: python tools/ntp_mains_pilot2.py ROOT CASE [CASE ...]   (in the folder holding the mains_hum.py scan parts release_0.jsonl and release_1.jsonl)
"""
import sys, json
from pathlib import Path
import numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
root = Path(sys.argv[1]); scan = {}
for f in ("release_0.jsonl", "release_1.jsonl"):
    for l in open(f):
        d = json.loads(l); scan[d["case"]] = d
for c in map(int, sys.argv[2:]):
    T = scan[c]["tracks"]
    k = max((k for k in T if k.startswith("SNUADC")), key=lambda k: max([e["snr"] for e in T[k]["lines"] if e["h"] == 2] or [0]))
    tr = read_vital(root / "vital_files" / f"{c:04d}.vital", track_names={k}).tracks[k]
    fs = float(tr.srate); rt, rs = tr.record_times, tr.record_sizes; starts = np.r_[0, np.cumsum(rs)[:-1]]
    x = np.nan_to_num(tr.physical().astype(float)); n = int(10 * fs); nb = x.size // n
    z = (x[: nb * n] * np.exp(-2j * np.pi * 120 * np.arange(nb * n) / fs)).reshape(nb, n).mean(1)
    gmc = np.unwrap(np.angle(z)) / (2 * np.pi * 120)                     # grid minus count, s (relative)
    tb = (np.arange(nb) * n + n / 2) / fs                                # block centre, count time
    smc = rt - starts / fs                                               # stamp minus count, per record
    gaps = np.where(np.abs(np.diff(smc)) > 2.0)[0]
    seg_edges = np.r_[0, gaps + 1, rt.size]
    pts = []
    for a, b in zip(seg_edges[:-1], seg_edges[1:]):                      # continuous stretches only
        tc = starts[a:b] / fs
        for m0 in np.arange(tc[0], tc[-1] - 60, 60):
            sel = (tc >= m0) & (tc < m0 + 60)
            if sel.sum() < 20: continue
            bsel = (tb >= m0) & (tb < m0 + 60)
            if bsel.sum() < 3 or np.median(np.abs(z[bsel])) < 0.3 * np.median(np.abs(z)): continue
            pts.append((m0 + 30, np.median(smc[a:b][sel]), np.mean(gmc[bsel]), sel.sum()))
    if len(pts) < 20: print(c, k, "too few clean minutes"); continue
    P = np.array(pts); t = (P[:, 0] - P[0, 0]) / 3600
    pc = (P[:, 1] - P[0, 1]) * 1000; gr = (P[:, 2] - P[0, 2]) * 1000; y = pc - gr
    jit = np.median(np.abs(np.diff(pc))) * 1.48
    pr = np.polyfit(t, y, 1); res = y - np.polyval(pr, t); dy = np.diff(y)
    thr = max(5 * np.median(np.abs(dy - np.median(dy))) * 1.48, 3)
    big = np.where(np.abs(dy - np.median(dy)) > thr)[0]
    print(f"case {c} [{k.split('/')[1]}]: {len(P)} minutes over {t[-1]:.1f} h; PC-minus-count changes minute to minute by {jit:.1f} ms (robust SD)")
    print(f"   PC minus count slope {np.polyfit(t, pc, 1)[0]/3.6:+.1f} ppm; grid minus count slope {np.polyfit(t, gr, 1)[0]/3.6:+.1f} ppm; PC minus grid slope {pr[0]/3.6:+.1f} ppm, residual SD {np.std(res):.1f} ms")
    print(f"   jumps in PC minus grid beyond {thr:.1f} ms: {len(big)}" + (": " + ", ".join(f"{t[i+1]:.2f} h ({dy[i]:+.1f} ms)" for i in big[:10]) if len(big) else ""))
