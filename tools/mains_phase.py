# SPDX-License-Identifier: MIT
"""Phase-difference test: BIS 60 Hz mains phase minus half the SNUADC 120 Hz mains phase, at common times.
The grid's own wander cancels; the slope is the two devices' relative clock rate, compared with the stamp model's
r_eeg - r_ecg. This machine only (per-case printout, case ids stay local).
Each device: 1 s demodulation blocks along its own sample count, within runs of contiguous records; block times =
run's first stamp + count / nominal rate. Pairs within the overlap of one BIS run and one SNUADC run.

Usage: python tools/mains_phase.py ROOT RATES.csv CASE [CASE ...]
"""
import csv, sys
from pathlib import Path
import numpy as np
BLK = 10
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

def phase_runs(tr, f_line, min_s=600):
    rt, rs, fs = tr.record_times, tr.record_sizes, float(tr.srate)
    ok = np.abs(np.diff(rt) - rs[:-1] / fs) < 0.5 / fs
    starts = np.r_[0, np.cumsum(rs)[:-1]]
    x = np.nan_to_num(tr.physical().astype(float))
    runs, i = [], 0
    while i < rt.size:
        j = i
        while j < rt.size - 1 and ok[j]: j += 1
        a, b = starts[i], starts[j] + rs[j]
        if (b - a) / fs >= min_s:
            seg = x[a:b]; k = np.arange(seg.size)
            z = seg * np.exp(-2j * np.pi * f_line * k / fs)
            n = int(fs * BLK); m = z[: z.size // n * n].reshape(-1, n).mean(1)
            t = rt[i] + (np.arange(m.size) * n + n / 2) / fs
            runs.append((t, np.unwrap(np.angle(m)), np.abs(m)))
        i = j + 1
    return runs

root = Path(sys.argv[1]); rates = {x['case']: x for x in csv.DictReader(open(sys.argv[2]))}
for c in sys.argv[3:]:
    v = read_vital(root / "vital_files" / f"{int(c):04d}.vital",
                   track_names={"SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/CVP", "SNUADC/ART", "BIS/EEG1_WAV", "BIS/EEG2_WAV"})
    r = rates[c]; model = float(r['r_eeg']) - float(r['r_ecg'])
    res = []
    for ks in ("SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/CVP", "SNUADC/ART"):
        if v.tracks.get(ks) is None or not v.tracks[ks].record_times.size: continue
        S = phase_runs(v.tracks[ks], 120.0)
        for kb in ("BIS/EEG1_WAV", "BIS/EEG2_WAV"):
            if v.tracks.get(kb) is None or not v.tracks[kb].record_times.size: continue
            B = phase_runs(v.tracks[kb], 60.0)
            for tb, pb, ab in B:
                for ts, ps, as_ in S:
                    lo, hi = max(tb[0], ts[0]), min(tb[-1], ts[-1])
                    if hi - lo < 600: continue
                    sel = (tb >= lo) & (tb <= hi)
                    d = pb[sel] - np.interp(tb[sel], ts, ps) / 2.0
                    d = np.unwrap(d * 2) / 2      # half-phase ambiguity of the 120 Hz line
                    t = tb[sel] - tb[sel][0]
                    p = np.polyfit(t, d, 1); resid = float(np.std(d - np.polyval(p, t)))
                    res.append((p[0] / (2 * np.pi * 60) * 1e6, resid, (hi - lo) / 60, ks.split('/')[1], kb.split('/')[1]))
    if not res: print(c, "no overlapping runs"); continue
    res.sort(key=lambda e: e[1])
    print(f"case {c:>5}: model BIS-SNUADC {model:+6.1f} ppm | best pairs (ppm, phase resid rad, minutes, tracks): " +
          "; ".join(f"{e[0]:+.1f} ({e[1]:.2f}, {e[2]:.0f} min, {e[3]}/{e[4]})" for e in res[:3]))
