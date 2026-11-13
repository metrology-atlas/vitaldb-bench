# SPDX-License-Identifier: MIT
"""How far PulseDB's PPG band-pass could move a PPG fiducial, if it were applied causally rather than zero-phase.

    python examples/ppg_filter_delay_sim.py [--out OUT.json]

PulseDB (Wang et al. 2023, section 2.4.2) filters the PPG with a 4th-order Chebyshev-II band-pass at 0.5-8 Hz before
peak detection, and does not say whether the filter is zero-phase. This simulates a synthetic pulse train (two
Gaussians per beat) at 500 Hz, decimates it to 125 Hz as PulseDB does, and filters it causally (sosfilt) and
zero-phase (sosfiltfilt) with stopband attenuations of 20 and 40 dB (PulseDB does not give it), at heart rates of
60, 90 and 120 /min. It reports the median shift of the systolic peak (parabolic sub-sample) against the unfiltered
125 Hz signal. Synthetic and illustrative: the shape of the pulse and the stopband are assumptions.
"""
import argparse, json
import numpy as np
from scipy import signal

FS = 125.0

def pulse_train(hr, fs=500.0, dur=20.0):
    t = np.arange(0, dur, 1 / fs)
    x = np.zeros_like(t)
    for b in np.arange(1, dur - 1, 60.0 / hr):
        x += np.exp(-0.5 * ((t - b - 0.18) / 0.06) ** 2) + 0.4 * np.exp(-0.5 * ((t - b - 0.45) / 0.09) ** 2)
    return t, x

def peak_times(y, t):
    p, _ = signal.find_peaks(y, distance=int(0.3 * FS))
    out = []
    for i in p:
        if 3 < t[i] < 17 and y[i] > 0.3 * y.max():
            a, b, c = y[i - 1], y[i], y[i + 1]
            out.append(t[i] + 0.5 * (a - c) / (a - 2 * b + c) / FS)
    return np.array(out)

def shift(p, ref):
    return float(np.median([p[np.argmin(abs(p - q))] - q for q in ref]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); a = ap.parse_args()
    rows = []
    for rs in (20, 40):
        sos = signal.cheby2(4, rs, [0.5, 8], btype="bandpass", fs=FS, output="sos")
        for hr in (60, 90, 120):
            t, x = pulse_train(hr)
            x125 = signal.resample_poly(x, 1, 4); t125 = t[::4]
            ref = peak_times(x125, t125)
            rows.append({"stopband_db": rs, "hr_per_min": hr,
                         "causal_ms": round(1000 * shift(peak_times(signal.sosfilt(sos, x125), t125), ref), 1),
                         "zero_phase_ms": round(1000 * shift(peak_times(signal.sosfiltfilt(sos, x125), t125), ref), 1)})
    for r in rows: print(r)
    if a.out:
        with open(a.out, "w") as f: json.dump({"rows": rows}, f, indent=1)

if __name__ == "__main__":
    main()
