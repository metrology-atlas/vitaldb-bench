# SPDX-License-Identifier: MIT
"""Join HR delay groups with each case's device ports into rows.json (argv: VitalDB root, hr_delay_group_labels.csv, device JSONL glob).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import csv, json, glob, sys, math
from collections import Counter, defaultdict
import numpy as np
R = sys.argv[1]
H = {int(r['case']): r for r in csv.DictReader(open(sys.argv[2]))}
DV = {}
for p in glob.glob(sys.argv[3]):
    for l in open(p):
        try: e = json.loads(l)
        except Exception: continue
        DV[e['case']] = e
SC = {}
for p in sys.argv[4].split(','):
    for f in glob.glob(p):
        for l in open(f):
            try: e = json.loads(l)
            except Exception: continue
            if 'tracks' in e and 'HR' in e['tracks']: SC[e['case']] = e['tracks']['HR']
U = {int(r['case']): r for r in csv.DictReader(open(sys.argv[5]))}
C = {int(r['caseid']): r for r in csv.DictReader(open(R + '/clinical_data.csv', encoding='utf-8-sig'))}
mix = json.loads(next(iter(H.values()))['mixture'])
mu, sd, w = map(np.array, (mix['mu'], mix['sd'], mix['w']))
def plate(x, se=0.0):
    v = sd**2 + se**2; l = w*np.exp(-0.5*(x-mu)**2/v)/np.sqrt(v); return l[1]/l.sum()
rows = []
for c, h in H.items():
    d = DV.get(c, {}); dev = d.get('devices', {}); cl = C.get(c, {}); u = U.get(c, {})
    sol = [k for k in d.get('tracks', {}) if k.startswith('Solar8000/')]
    b = [x['d'] for x in SC.get(c, {}).get('blocks', []) if not x['edge']]
    bl = [plate(x, 0.2) for x in b]
    sdrift = u.get('snuadc_drift_ppm') or ''
    rows.append(dict(case=c, g=h['group'], d=float(h['delay_s']), port=h['solar_port'],
        sig='|'.join(f'{k}:{v[1]}' for k, v in sorted(dev.items())),
        devset='|'.join(sorted(dev)), soln=(dev.get('Solar8000') or ['?'])[0], ver=d.get('version'),
        soltr=len(sol), soltrset='|'.join(sorted(s.split('/')[1] for s in sol)),
        snu_mode=('none' if not sdrift else 'step' if float(sdrift) < -20 else 'small'),
        snu_unit=u.get('snuadc_unit') or '-', bis_unit=u.get('bis_unit') or '-',
        subj=cl.get('subjectid'), dept=cl.get('department'), optype=cl.get('optype'), ecg=cl.get('preop_ecg'),
        ane=cl.get('ane_type'), age=cl.get('age'), emop=cl.get('emop'),
        nb=len(b), late_blocks=sum(p >= 0.9 for p in bl), early_blocks=sum(p <= 0.1 for p in bl),
        hr_int=float(np.median([x['interval'] for x in SC.get(c, {}).get('blocks', []) if x.get('interval') is not None] or [np.nan])) if SC.get(c, {}).get('blocks') else None))
json.dump(rows, open(sys.argv[6], 'w'), default=float)
print(len(rows), sum(1 for r in rows if r['sig']), sum(1 for r in rows if r['nb']))
