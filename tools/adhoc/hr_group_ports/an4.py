# SPDX-License-Identifier: MIT
"""COM12 against other ports from the fs.py scan (rows.json, f0-f2.jsonl).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json, numpy as np
from collections import Counter, defaultdict
R = {r['case']: r for r in json.load(open('rows.json'))}
F = [json.loads(l) for i in range(3) for l in open(f'f{i}.jsonl')]
print('errors', sum('error' in f for f in F))
F = [f for f in F if 'error' not in f]
def grp(f):
    r = R[f['case']]; return ('COM12' if r['port'] == 'COM12' else 'other'), r['g']
for k in ('dt_p50', 'n2', 'n1', 'frac_ms_p50', 'pleth_same_time', 'eq_pleth', 'solar_first_s'):
    out = []
    for port in ('COM12', 'other'):
        for g in ('early', 'late', 'uncertain'):
            v = np.array([f[k] for f in F if grp(f) == (port, g) and f.get(k) is not None], float)
            if v.size: out.append(f"{port}/{g}: {np.percentile(v, [10, 50, 90]).round(3).tolist()} n={v.size}")
    print(k); [print('   ', o) for o in out]
for port in ('COM12', 'other'):
    t = defaultdict(Counter)
    for f in F:
        p, g = grp(f)
        if p == port: t['|'.join(x.split('/')[1] for x in f['has'] if x.startswith('Solar'))][g] += 1
    print(port, {a: dict(c) for a, c in t.items()})
