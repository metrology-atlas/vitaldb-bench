# SPDX-License-Identifier: MIT
"""HR delay group (early/late) against recording-setup fields: which field separates the groups (rows.json from hr_port.py).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json, numpy as np
from collections import Counter, defaultdict
R = [r for r in json.load(open('rows.json'))]
def cv(rs, k, minn=10):
    t = defaultdict(Counter)
    for r in rs:
        if r['g'] in ('early', 'late'): t[str(r[k])][r['g']] += 1
    lv = {a: c for a, c in t.items() if sum(c.values()) >= minn}
    n = sum(sum(c.values()) for c in lv.values())
    if n == 0 or len(lv) < 2: return None
    E = sum(c['early'] for c in lv.values()); L = n - E
    chi = 0
    for c in lv.values():
        m = sum(c.values())
        for g, tot in (('early', E), ('late', L)):
            e = m*tot/n
            if e: chi += (c[g]-e)**2/e
    V = (chi/n)**0.5
    pur = sum(max(c.values()) for c in lv.values())/n
    return dict(V=round(V, 2), purity=round(pur, 2), base=round(max(E, L)/n, 2), levels=len(lv), n=n,
                top={a: dict(c) for a, c in sorted(lv.items(), key=lambda kv: -sum(kv[1].values()))[:8]})
keys = ['sig', 'devset', 'soln', 'ver', 'soltr', 'soltrset', 'snu_mode', 'snu_unit', 'bis_unit', 'dept', 'optype', 'ecg', 'ane', 'emop']
for port in ('COM26', 'COM5', 'COM12', 'ALL'):
    rs = R if port == 'ALL' else [r for r in R if r['port'] == port]
    print('====', port, Counter(r['g'] for r in rs))
    for k in keys:
        x = cv(rs, k)
        if x: print(f"  {k:9s} V={x['V']:.2f} purity={x['purity']:.2f} base={x['base']:.2f} lv={x['levels']} n={x['n']}")
