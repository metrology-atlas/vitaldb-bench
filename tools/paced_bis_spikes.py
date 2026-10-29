# SPDX-License-Identifier: MIT
"""Pacing spikes in the BIS EEG? Beat-triggered average of BIS EEG around SNUADC paced beats, 3-min windows.
This machine only. Reports per case the averaged response: peak-to-peak, the narrowest feature (width at half height
of the sharpest deflection, in samples), and a spike index = max |first difference| / robust SD of the
first difference of the average outside the response, against a control with beat times shifted by half a period.
Also the SNUADC ECG average's sharpest feature, to see whether a spike survives there.

Usage: python tools/paced_bis_spikes.py ROOT   (the paced recordings and their pacing periods are listed in the script)
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pat import r_peaks, sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
root = Path(sys.argv[1])

def trig_avg(t, x, fs, beats, lo=-1.6, hi=0.6):
    i0 = np.searchsorted(t, beats + lo); L = int((hi - lo) * fs)
    ok = (i0 > 0) & (i0 + L < x.size) & (np.abs(t[np.clip(i0, 0, t.size - 1)] - (beats + lo)) < 2 / fs)
    if ok.sum() < 30: return None
    return np.mean(np.stack([x[i:i + L] for i in i0[ok]]), 0), ok.sum()

def spikiness(a):
    d = np.diff(a); s = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-12
    return float(np.max(np.abs(d)) / s)

for c, P in [(830, 1.002), (1219, 0.998), (2701, 0.860), (4368, 1.204), (4631, 1.000), (5133, 0.860), (6378, 0.860)]:
    v = read_vital(root / "vital_files" / f"{c:04d}.vital", track_names={"SNUADC/ECG_II", "BIS/EEG1_WAV", "BIS/EEG2_WAV"})
    e = v.tracks.get("SNUADC/ECG_II")
    if e is None or not e.record_times.size: print(c, "no ECG"); continue
    te = sample_times(e); xe = np.nan_to_num(e.physical().astype(float)); fe = float(e.srate)
    beats = te[r_peaks(xe, fe)]
    rr = np.diff(beats); paced = np.r_[False, np.abs(rr - P) < 0.012]          # beats at the pacing interval
    pb = beats[paced]
    res = {"case": c, "paced_beats": int(pb.size)}
    a = trig_avg(te, xe - signal.medfilt(xe, 251), fe, pb, -0.3, 0.3)
    if a: res["ecg_spike_index"] = round(spikiness(a[0]), 1)
    for k in ("BIS/EEG1_WAV", "BIS/EEG2_WAV"):
        b = v.tracks.get(k)
        if b is None or not b.record_times.size: continue
        tb = sample_times(b); xb = np.nan_to_num(b.physical().astype(float)); fb = float(b.srate)
        xb = signal.sosfiltfilt(signal.butter(2, 1, "highpass", fs=fb, output="sos"), xb)
        idx, ctl, amp = [], [], []
        for w0 in np.arange(pb[0], pb[-1], 180):
            sel = pb[(pb >= w0) & (pb < w0 + 180)]
            if sel.size < 60: continue
            A = trig_avg(tb, xb, fb, sel); C = trig_avg(tb, xb, fb, sel + P / 2)
            if A and C:
                idx.append(spikiness(A[0])); ctl.append(spikiness(C[0])); amp.append(np.ptp(A[0]) / (np.ptp(C[0]) + 1e-12))
        if idx:
            res[k.split("/")[1]] = {"windows": len(idx), "spike_index_median": round(float(np.median(idx)), 1),
                                    "control_median": round(float(np.median(ctl)), 1),
                                    "beat_locked_ptp_vs_control": round(float(np.median(amp)), 2)}
    print(json.dumps(res))
