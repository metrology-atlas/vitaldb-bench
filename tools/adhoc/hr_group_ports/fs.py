# SPDX-License-Identifier: MIT
"""Per-case scan of Solar8000 and SNUADC tracks for the HR port analysis (argv: VitalDB root, output JSONL, case list).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json, sys, numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
R, out, cases = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',') if x]
done = set()
try: done = {json.loads(l)['case'] for l in open(out)}
except FileNotFoundError: pass
names = {'Solar8000/HR', 'Solar8000/PLETH_HR', 'Solar8000/ART_MBP', 'Solar8000/PLETH_SPO2', 'Solar8000/NIBP_SBP', 'SNUADC/ECG_II'}
for c in cases:
    if c in done: continue
    try:
        v = read_vital(f'{R}/vital_files/{c:04d}.vital', track_names=names)
        e = {'case': c}
        hr = v.tracks.get('Solar8000/HR'); ph = v.tracks.get('Solar8000/PLETH_HR')
        t = np.sort(hr.record_times); dt = np.diff(t)
        e['dt_p50'] = float(np.median(dt)); e['dt_p10_90'] = np.percentile(dt, [10, 90]).round(3).tolist()
        e['n2'] = float(np.mean(np.abs(dt - 2) < 0.2)); e['n1'] = float(np.mean(np.abs(dt - 1) < 0.2))
        ph0 = (t - t[0]) % 2.0
        e['phase_sd'] = float(np.std(np.angle(np.mean(np.exp(1j*np.pi*ph0)))))
        e['frac_ms_p50'] = float(np.median((t % 1) * 1000))
        x = hr.physical()
        if ph is not None and ph.record_times.size:
            tp = ph.record_times; xp = ph.physical()
            i = np.searchsorted(tp, hr.record_times); i = np.clip(i, 0, tp.size - 1)
            same_t = np.abs(tp[i] - hr.record_times) < 0.05
            e['pleth_same_time'] = float(same_t.mean())
            e['eq_pleth'] = float(np.mean(x[same_t] == xp[i][same_t])) if same_t.any() else None
        e['solar_first_s'] = float(t[0] - v.tracks['SNUADC/ECG_II'].record_times.min()) if 'SNUADC/ECG_II' in v.tracks else None
        e['has'] = sorted(k for k in v.tracks)
    except Exception as ex:
        e = {'case': c, 'error': repr(ex)[:200]}
    with open(out, 'a') as fh: fh.write(json.dumps(e) + '\n')
