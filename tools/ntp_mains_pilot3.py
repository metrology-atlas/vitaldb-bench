# SPDX-License-Identifier: MIT
"""NTP pilot: PC time minus grid time at each SNUADC stamp correction (frequent-correction recordings). This machine only.
On the SNUADC sample count: stamp offset (stamp minus count time) jumps at each correction; the 120 Hz mains phase
gives grid time minus count time. Right after a correction the stamp sits on PC time, so
PC minus grid = (stamp - count) - (grid - count) there, up to a constant.

Usage: python tools/ntp_mains_pilot3.py ROOT CASE [CASE ...]   (same folder as ntp_mains_pilot2.py)
"""
import sys, json
from pathlib import Path
import numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
root = Path(sys.argv[1])
SCAN = {}
for f in ("release_0.jsonl", "release_1.jsonl"):
    for l in open(f):
        d = json.loads(l); SCAN[d["case"]] = d
for c in map(int, sys.argv[2:]):
    best = None
    T = SCAN[c]["tracks"]
    kk = max((k for k in T if k.startswith("SNUADC")), key=lambda k: max([e["snr"] for e in T[k]["lines"] if e["h"] == 2] or [0]))
    for k in (kk,):
        v = read_vital(root / "vital_files" / f"{c:04d}.vital", track_names={k}); tr = v.tracks.get(k)
        if tr is None or not tr.record_times.size: continue
        fs = float(tr.srate); x = np.nan_to_num(tr.physical().astype(float)); n = int(10 * fs); nb = x.size // n
        z = (x[: nb * n] * np.exp(-2j * np.pi * 120 * np.arange(nb * n) / fs)).reshape(nb, n).mean(1)
        q = np.median(np.abs(z)) / (np.median(np.abs(np.diff(z))) + 1e-12)
        if best is None or q > best[0]: best = (q, k, tr, z, fs, n)
    q, k, tr, z, fs, n = best
    rt, rs = tr.record_times, tr.record_sizes; starts = np.r_[0, np.cumsum(rs)[:-1]]
    step = np.diff(rt) - rs[:-1] / fs                             # stamp jump at each record boundary
    cum = np.r_[0, np.cumsum(step)]                               # stamp minus count time, per record
    corr = np.where(np.abs(step) > 0.001)[0] + 1                  # record index right after each correction
    gaps = np.where(np.abs(step) > 1.0)[0] + 1
    centre = np.arange(z.size) * n + n / 2
    good = np.abs(z) > 0.3 * np.median(np.abs(z))
    ph = np.unwrap(np.angle(z)); grid_minus_count = ph / (2 * np.pi * 120)
    out = []
    for r in corr:
        if r in gaps: continue
        b = int(np.searchsorted(centre, starts[r]))
        if b >= z.size or not good[max(b - 1, 0):b + 2].all(): continue
        out.append((rt[r], cum[r] - grid_minus_count[b], step[r - 1]))
    if len(out) < 5: print(c, k, "too few clean corrections"); continue
    O = np.array(out); t = (O[:, 0] - O[0, 0]) / 3600; y = (O[:, 1] - O[0, 1]) * 1000
    p = np.polyfit(t, y, 1); res = y - np.polyval(p, t); dy = np.diff(y); dt = np.diff(t) * 60
    big = np.where(np.abs(dy - np.median(dy)) > 5)[0]
    print(f"case {c} [{k.split('/')[1]}, line quality {q:.0f}]: {len(O)} corrections over {t[-1]:.1f} h; correction size median {np.median(O[:,2])*1000:+.1f} ms "
          f"(IQR {np.percentile(O[:,2]*1000,25):+.1f} to {np.percentile(O[:,2]*1000,75):+.1f})")
    print(f"   PC minus grid: slope {p[0]/3.6:+.1f} ppm; residual SD {np.std(res):.1f} ms; correction-to-correction change median {np.median(dy):+.2f} ms "
          f"(spacing median {np.median(dt):.1f} min); changes >5 ms off the median: {len(big)}"
          + (f" at {', '.join(f'{t[i+1]:.2f} h ({dy[i]:+.1f} ms)' for i in big[:8])}" if len(big) else ""))
