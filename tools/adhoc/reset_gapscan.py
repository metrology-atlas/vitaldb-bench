# SPDX-License-Identifier: MIT
"""Scan for a pause in sampling near each SNUADC reset: at each sample boundary, does a time gap of m samples explain the waveform better?

At boundary i the 20 samples before are placed at t = -19..0 and the 20 after at t = 1+m..20+m; one quadratic is
fitted to all 40, and the residual for m = 1..15 is compared with m = 0. The inserted zero sample is removed first.

    python tools/adhoc/reset_gapscan.py ROOT OUT.jsonl SHARD NSHARDS clock_fits*.jsonl

Per case, this machine only; aggregates leave. Kept because results in the atlas study reset-cluster-order
came from it.
"""
import json
import sys
import time
import numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
from timenet_connectors.datasets.vitaldb.vitaldb.resets import reorder_track
R, out, shard, nsh = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
fits = {json.loads(l)['case']: json.loads(l) for f in sys.argv[5:] for l in open(f)}
cases = [c for c, f in sorted(fits.items()) if f.get('snuadc_mode') == 'step'][shard::nsh]
done = set()
try: done = {json.loads(l)['case'] for l in open(out)}
except FileNotFoundError: pass
CH = ['SNUADC/ART', 'SNUADC/PLETH', 'SNUADC/CVP', 'SNUADC/ECG_II', 'SNUADC/ECG_V5']
USE = ['SNUADC/ART', 'SNUADC/PLETH', 'SNUADC/CVP']
H = 20; MS = np.arange(0, 16)
PROJ = []
for m in MS:
    t = np.r_[np.arange(-H + 1, 1), np.arange(1, H + 1) + m].astype(float) / H
    X = np.c_[np.ones_like(t), t, t * t]
    PROJ.append(np.eye(2 * H) - X @ np.linalg.pinv(X))
PROJ = np.stack(PROJ)            # (M, 40, 40)
def scan(x, lo, hi):
    idx = np.arange(lo, hi)
    Y = x[idx[:, None] + np.arange(-H + 1, H + 1)[None, :]]          # (n, 40)
    ok = np.all(np.isfinite(Y), 1)
    Y = np.nan_to_num(Y)
    Rr = np.einsum('mij,nj->nmi', PROJ, Y)
    sse = (Rr ** 2).sum(-1)                                            # (n, M)
    sse[~ok] = np.nan
    return sse
t0 = time.time(); fo = open(out, 'a')
for c in cases:
    if c in done: continue
    if time.time() - t0 > 130: break
    v = read_vital(f'{R}/vital_files/{c:04d}.vital', track_names=set(CH))
    raw, xs, firsts, offs, n = {}, {}, None, None, None
    for name in CH:
        tr = v.tracks.get(name)
        if tr is None or tr.record_times.size < 100: continue
        tr2, d = reorder_track(tr)
        if firsts is None: firsts, offs, n = d.first_records, tr2.record_offsets, tr2.record_offsets[-1]
        if tr2.record_offsets[-1] != n: continue
        raw[name] = np.asarray(tr2.values).astype(np.int64); xs[name] = tr2.physical().astype(float)
    row = {'case': c, 'resets': [], 'base': []}
    allzero = None
    for k in raw:
        zz = raw[k] == 0
        allzero = zz if allzero is None else allzero & zz
    def window(z, lo=-300, hi=800):
        """Per channel: best m, its SSE drop over sigma^2, and the noise level, at each boundary in [z+lo, z+hi)."""
        a, b = z + lo - H - 5, z + hi + H + 5
        keep = ~allzero[a:b] if allzero is not None else np.ones(b - a, bool)
        pos = np.arange(a, b)[keep] - z                                   # original positions, zero removed
        res = {}
        for k in USE:
            if k not in xs: continue
            x = xs[k][a:b][keep]
            if np.nanstd(x) == 0: continue
            sse = scan(x, H, x.size - H)
            s2 = np.nanmedian(sse[:, 0]) / (2 * H - 3) + 1e-12
            gain = (sse[:, :1] - sse) / s2
            j = np.nanargmax(np.nan_to_num(gain, nan=-1), 1)
            g = gain[np.arange(gain.shape[0]), j]
            p = pos[H:x.size - H]
            top = np.argsort(-np.nan_to_num(g))[:3]
            res[k.split('/')[1]] = [(int(p[t]), int(MS[j[t]]), round(float(g[t]), 1)) for t in top]
        return res
    for r in (firsts or ()):
        z = int(offs[r])
        if z < 5000 or z + 5000 > n: continue
        zeros = (np.flatnonzero(allzero[z - 300:z + 800]) - 300).tolist() if allzero is not None else []
        row['resets'].append({'zeros': zeros, 'top': window(z)})
    rng = np.random.default_rng(c)
    for z in rng.choice(np.arange(10000, n - 10000), 3, replace=False):
        row['base'].append({'top': window(int(z))})
    fo.write(json.dumps(row) + '\n'); fo.flush()
