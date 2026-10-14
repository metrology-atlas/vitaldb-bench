# SPDX-License-Identifier: MIT
"""HR delay groups by PC key, the ports of Solar8000, SNUADC, Primus and BIS (rows.json). Source of hr_port_split_summary.json.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json, numpy as np
from collections import Counter, defaultdict
R = json.load(open('rows.json'))
def key(r):
    d = dict(x.split(':') for x in r['sig'].split('|') if x)
    return '|'.join(f"{k}:{d.get(k, '-')}" for k in ('Solar8000', 'SNUADC', 'Primus', 'BIS'))
t = defaultdict(Counter)
for r in R: t[key(r)][r['g']] += 1
cls = [(k, c) for k, c in t.items() if c['early'] + c['late'] >= 20]
n = sum(c['early'] + c['late'] for _, c in cls)
pure = [(k, c) for k, c in cls if max(c['early'], c['late']) / (c['early'] + c['late']) >= 0.9]
npure = sum(c['early'] + c['late'] for _, c in pure)
print('PC keys with >=20 classified cases', len(cls), 'cases', n, '| pure (>=90%)', len(pure), 'cases', npure, round(npure / n, 3))
acc = sum(max(c['early'], c['late']) for _, c in cls) / n
print('accuracy of majority-group-per-key', round(acc, 3))
mixed = sorted([(k, dict(c)) for k, c in cls if (k, c) not in pure], key=lambda kv: -sum(kv[1].values()))
for k, c in mixed: print('  mixed', c, k)
# leave-one-out: classify uncertain? report uncertain in pure keys
unc = sum(c['uncertain'] for _, c in pure)
print('uncertain cases sitting in pure keys', unc, 'of', sum(c['uncertain'] for c in t.values()))
S = {'pc_keys_ge20': len(cls), 'classified_cases': n, 'pure_keys_ge90pct': len(pure), 'cases_in_pure_keys': npure,
     'majority_accuracy': round(acc, 3), 'uncertain_in_pure_keys': unc, 'uncertain_total': sum(c['uncertain'] for c in t.values()),
     'mixed_keys': [{'key_solar_port': k.split('|')[0], 'counts': c} for k, c in mixed]}
json.dump(S, open('pc_key_summary.json', 'w'), indent=1)
