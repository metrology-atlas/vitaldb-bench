# SPDX-License-Identifier: MIT
"""How often each device clock is corrected: correction times, sizes and intervals per track. This machine only.

    python tools/clock_correction_periods.py ROOT cases.json out.jsonl [time_budget_s]

For SNUADC/ECG_II and BIS/EEG1_WAV: physiosync.stamp_clock correction events and the intervals between
consecutive events inside one stretch. For those and Primus/AWP: small stamp steps (0.1 ms to 0.1 s, single-record
flicks excluded), their sizes and spacing, and steps of 0.1 s or more (gaps and large corrections). Resumable;
stops after time_budget_s (default 160, for short sessions; pass a large value for a full run).
"""

import json
import sys
import time

import numpy as np
from physiosync import stamp_clock
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
R, cases_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
BUDGET = float(sys.argv[4]) if len(sys.argv) > 4 else 160.0
cases = json.load(open(cases_json))
done = set()
try: done = {json.loads(l)['case'] for l in open(out)}
except FileNotFoundError: pass
TR = {'SNUADC/ECG_II': -1, 'BIS/EEG1_WAV': 1, 'Primus/AWP': None}
t0 = time.time()
with open(out, 'a') as fh:
    for c in cases:
        if c in done: continue
        if time.time() - t0 > BUDGET: break
        res = {'case': c}
        try:
            v = read_vital(f'{R}/vital_files/{c:04d}.vital', track_names=set(TR))
            for name, sign in TR.items():
                tr = v.tracks.get(name)
                if tr is None or tr.record_times.size < 10: continue
                t = tr.record_times; sz = tr.record_sizes[:t.size]; fs = float(tr.srate)
                st = tr.record_offsets[:t.size]
                d = np.diff(t) - sz[:-1] / fs                     # stamp step between records (s)
                r = {'hours': round(float((t[-1] - t[0]) / 3600), 3), 'records': int(t.size), 'fs': fs}
                if sign is not None:
                    clk = stamp_clock(t, st, fs, correction_sign=sign)
                    et = np.array(clk['corrections']['event_times_s'])
                    # stretch of each event: record index -> stretch id
                    br = clk['breaks']
                    ev_rec = np.searchsorted(t, et)
                    sid = np.searchsorted(br, ev_rec, side='right') - 1
                    iv = [float(et[i + 1] - et[i]) for i in range(et.size - 1) if sid[i + 1] == sid[i]]
                    r.update(rate_ppm=round(clk['rate_ppm'], 2), events=int(et.size), event_intervals_s=[round(x, 1) for x in iv],
                             median_event_s=clk['corrections']['median_event_s'])
                # small steps: 0.1 ms <= |d| < 0.1 s (single-record flicks excluded: a step and its reverse)
                flick = np.zeros(d.size, bool)
                pair = (np.abs(d[:-1] + d[1:]) < 1e-4) & (np.abs(d[:-1]) >= 1e-4)
                flick[:-1] |= pair; flick[1:] |= pair
                small = (np.abs(d) >= 1e-4) & (np.abs(d) < 0.1) & ~flick
                big = np.abs(d) >= 0.1
                si = np.flatnonzero(small)
                r.update(small_steps=int(si.size), small_step_ms=np.percentile(d[si] * 1000, [10, 50, 90]).round(3).tolist() if si.size else None,
                         small_step_gap_s=round(float(np.median(np.diff(t[si + 1]))), 1) if si.size > 1 else None,
                         big_steps=int(big.sum()), big_step_s=[round(float(x), 3) for x in d[big][:40]],
                         big_step_gap_s=[round(float(x), 1) for x in np.diff(t[np.flatnonzero(big) + 1])[:40]])
                res[name] = r
        except Exception as ex:  # noqa: BLE001
            res['error'] = f'{type(ex).__name__}: {ex}'[:200]
        fh.write(json.dumps(res) + '\n'); fh.flush()
