# SPDX-License-Identifier: MIT
"""COM12 cases: delay distribution and within-case block classification of the HR group (rows.json).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json, numpy as np
from collections import Counter, defaultdict
R = json.load(open('rows.json'))
P = [r for r in R if r['port'] == 'COM12']
for g in ('early', 'late', 'uncertain'):
    d = np.array([r['d'] for r in P if r['g'] == g]); print(g, len(d), np.percentile(d, [10, 50, 90]).round(2))
print('delay histogram COM12', np.histogram([r['d'] for r in P], bins=np.arange(0, 7.5, 0.5))[0])
# within-case: blocks classified
for g in ('early', 'late'):
    rs = [r for r in P if r['g'] == g and r['nb'] >= 4]
    mixed = [r for r in rs if r['late_blocks'] >= 2 and r['early_blocks'] >= 2]
    print(g, 'cases>=4 blocks', len(rs), 'with >=2 blocks of each group', len(mixed))
# all ports for comparison
rs = [r for r in R if r['g'] in ('early', 'late') and r['nb'] >= 4]
print('ALL mixed-block share', round(np.mean([(r['late_blocks'] >= 2 and r['early_blocks'] >= 2) for r in rs]), 3))
rs = [r for r in P if r['g'] in ('early', 'late') and r['nb'] >= 4]
print('COM12 mixed-block share', round(np.mean([(r['late_blocks'] >= 2 and r['early_blocks'] >= 2) for r in rs]), 3))
# case order runs test
s = [r for r in sorted(P, key=lambda r: r['case']) if r['g'] in ('early', 'late')]
x = [r['g'] == 'late' for r in s]
runs = 1 + sum(a != b for a, b in zip(x, x[1:])); n1 = sum(x); n0 = len(x) - n1; n = len(x)
mu = 2*n1*n0/n + 1; var = 2*n1*n0*(2*n1*n0 - n)/(n*n*(n-1))
print('runs', runs, 'expected', round(mu, 1), 'z', round((runs - mu)/var**0.5, 2))
# late share by case-id decile
ids = np.array([r['case'] for r in s]); xs = np.array(x)
for q in range(5):
    lo, hi = np.percentile(ids, [q*20, q*20+20]); m = (ids >= lo) & (ids <= hi)
    print(f'  caseid {int(lo)}-{int(hi)} late share {xs[m].mean():.2f} n={m.sum()}')
# repeat subjects
by = defaultdict(list)
for r in R:
    if r['g'] in ('early', 'late'): by[r['subj']].append(r)
same = diff = 0
for v in by.values():
    for i in range(len(v)):
        for j in range(i+1, len(v)):
            if v[i]['port'] == v[j]['port'] == 'COM12':
                same += v[i]['g'] == v[j]['g']; diff += v[i]['g'] != v[j]['g']
print('COM12 repeat-subject pairs same/diff', same, diff)
# HR interval, age, dept, optype, units
def tab(k):
    t = defaultdict(Counter)
    for r in P:
        if r['g'] in ('early', 'late'): t[str(r[k])][r['g']] += 1
    return {a: dict(c) for a, c in sorted(t.items(), key=lambda kv: -sum(kv[1].values()))[:8]}
for k in ('dept', 'optype', 'bis_unit', 'snu_unit', 'snu_mode', 'ane', 'emop', 'ecg'): print(k, tab(k))
for g in ('early', 'late'):
    hi = np.array([r['hr_int'] for r in P if r['g'] == g and r['hr_int'] is not None and r['hr_int'] == r['hr_int']])
    ag = np.array([float(r['age']) for r in P if r['g'] == g and r['age'] not in (None, '')])
    print(g, 'block interval p50', np.round(np.median(hi), 3), 'age p50', np.median(ag))
