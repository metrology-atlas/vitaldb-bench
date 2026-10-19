# SPDX-License-Identifier: MIT
"""The one-sample glitch at each SNUADC reset: where it sits, what it holds, and how much time the signal loses across it.

    python tools/adhoc/reset_glitch.py ROOT OUT.jsonl SHARD NSHARDS clock_fits*.jsonl

Per case, this machine only; aggregates leave. Kept because results in the atlas study reset-cluster-order came
from it.
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
W = 6
def across(x, g, k=1):
    """Missing samples across k bad samples at g..g+k-1: lines fitted on each side, compared at g+k."""
    a = x[g - W:g]; b = x[g + k:g + k + W]
    if a.size < W or b.size < W or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))): return None
    ia = np.arange(-W, 0); ib = np.arange(k, k + W)
    pa = np.polyfit(ia, a, 1); pb = np.polyfit(ib, b, 1)
    s1, s2 = pa[0], pb[0]; sl = (s1 + s2) / 2
    noise = max(np.std(a - np.polyval(pa, ia)), np.std(b - np.polyval(pb, ib)))
    if np.sign(s1) != np.sign(s2) or abs(s1 - s2) > 0.4 * max(abs(s1), abs(s2)) or abs(sl) < 8 * noise / W: return None
    return float((np.polyval(pb, k) - np.polyval(pa, k)) / sl), float(abs(sl) / (noise + 1e-12))
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
        xs[name] = tr2.physical().astype(float); raw[name] = np.asarray(tr2.values)
    row = {'case': c, 'resets': []}
    for r in (firsts or ()):
        z = int(offs[r])
        if z < 2000 or z + 2000 > n: continue
        # the glitch: a sample far off the line through its neighbours in every channel
        score = None
        for k in xs:
            seg = raw[k][z - 600:z + 1100].astype(np.int64)
            e = np.abs(seg[1:-1] - (seg[:-2] + seg[2:]) // 2)
            q = (e > 200).astype(int)          # ADC counts: far beyond any physiological step between samples
            score = q if score is None else score + q
        idx = np.flatnonzero(score >= max(2, len(xs) - 1)) + 1 - 600   # relative to z
        gl = []
        for p in idx:
            g = z + int(p)
            # length of the bad run: consecutive samples off the line in most channels
            info = {'pos': int(p), 'raw': {k.split('/')[1]: int(raw[k][g]) for k in xs},
                    'nbr': {k.split('/')[1]: [int(raw[k][g - 1]), int(raw[k][g + 1])] for k in xs}, 'gap': {}}
            for k, x in xs.items():
                a = across(x, g)
                if a is not None: info['gap'][k.split('/')[1]] = [round(a[0], 2), round(a[1], 1)]
            gl.append(info)
        # control: the same estimate across a random good sample in the same stretch
        ctl = {}
        rng = np.random.default_rng(r)
        for k, x in xs.items():
            vals = [across(x, z + int(q)) for q in rng.integers(-1500, -700, 40)]
            ctl[k.split('/')[1]] = [round(a[0], 2) for a in vals if a is not None]
        row['resets'].append({'z': z, 'sizes': [int(s) for s in np.diff(offs[r - 1:r + 7])], 'glitches': gl, 'ctl': ctl})
    fo.write(json.dumps(row) + '\n'); fo.flush()
