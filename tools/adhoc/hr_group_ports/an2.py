# SPDX-License-Identifier: MIT
"""HR delay groups by device-port signature for COM26, COM5 and COM12 (rows.json).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json
from collections import Counter, defaultdict
R = json.load(open('rows.json'))
for port in ('COM26', 'COM5', 'COM12'):
    t = defaultdict(Counter)
    for r in R:
        if r['port'] == port: t[r['sig']][r['g']] += 1
    print('==', port)
    for s, c in sorted(t.items(), key=lambda kv: -sum(kv[1].values())):
        if sum(c.values()) >= 5: print(f"   {dict(c)!s:55s} {s}")
