# SPDX-License-Identifier: MIT
"""Pulse arrival time (ECG R-peak to PPG pulse) from the original .vital files. Aggregates only.

    python tools/pat.py --root R --cases cases.json --out pat.jsonl [--windows 10] [--window-s 60] [--time-budget 150]

For each case, up to ``--windows`` random windows where SNUADC/ECG_II and SNUADC/PLETH both have data.
Each track keeps its own sample times: record time stamp + sample index / 500 Hz, as the file has them.

- R-peaks: ECG band-passed 5-25 Hz, peaks of |signal| at least 0.25 s apart, then moved to the
  extreme of the band-passed ECG (sign chosen per window) within +/-40 ms.
- PPG: band-passed 0.5-10 Hz. For each R-peak, the steepest upstroke in the next 0.8 s (and before the
  next R-peak + 0.3 s) gives three times: the foot (intersecting tangents: the tangent at the steepest
  point meets the level of the minimum in the 0.4 s before it), the steepest point, and the systolic
  peak (the maximum in the 0.4 s after it).
- PAT = PPG time - R-peak time, per beat.
- A pairing-free check: every PPG upstroke's foot relative to every R-peak within +/-1 s.

Also reports how the two tracks' record time stamps relate (same stamps, or offset).
Output: one line per case with histograms (2 ms bins, -1 to +1 s) and quantiles; no signal values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy import signal

from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

ECG, PPG = "SNUADC/ECG_II", "SNUADC/PLETH"
EDGES = np.arange(-1.0, 1.0 + 1e-9, 0.002)


def sample_times(tr) -> np.ndarray:
    rec = np.repeat(np.arange(tr.record_times.size), tr.record_sizes)
    within = np.arange(rec.size) - tr.record_offsets[rec]
    return tr.record_times[rec] + within / float(tr.srate)


def r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    sos = signal.butter(2, (5, 25), btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)
    a = np.abs(y)
    prom = 0.3 * (np.quantile(a, 0.995) - np.quantile(a, 0.05))
    pk, _ = signal.find_peaks(a, distance=int(0.25 * fs), prominence=prom)
    if pk.size < 5:
        return np.empty(0, int)
    sign = 1.0 if np.median(y[pk]) >= 0 else -1.0
    w = int(0.04 * fs)
    out = [max(p - w, 0) + int(np.argmax(sign * y[max(p - w, 0) : p + w + 1])) for p in pk]
    return np.unique(np.array(out))


def ppg_points(x: np.ndarray, fs: float):
    sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)
    d = np.gradient(y) * fs
    return y, d


def analyse_window(te, xe, tp, xp, fs) -> dict:
    out = {"foot": [], "slope": [], "peak": [], "corr": []}
    rp = r_peaks(xe, fs)
    if rp.size < 5:
        return out
    y, d = ppg_points(xp, fs)
    tr = te[rp]
    # every PPG upstroke, found on its own (no pairing): its foot time relative to every R-peak within
    # +/-1 s. Peaks of this histogram repeat at the R-R interval, so a delay longer than one beat shows.
    up, _ = signal.find_peaks(d, distance=int(0.3 * fs), prominence=0.3 * (np.quantile(d, 0.99) - np.quantile(d, 0.5)))
    feet = []
    for m in up:
        lo = max(0, m - int(0.4 * fs))
        feet.append(tp[m] - (y[m] - float(np.min(y[lo : m + 1]))) / d[m])
    feet = np.array(feet)
    for t0 in tr:
        rel = feet[(feet > t0 - 1) & (feet < t0 + 1)] - t0
        out["corr"].extend(rel.tolist())
    for i, t0 in enumerate(tr):
        t_end = min(t0 + 0.8, (tr[i + 1] + 0.3) if i + 1 < tr.size else t0 + 0.8)
        a, b = np.searchsorted(tp, t0), np.searchsorted(tp, t_end)
        if b - a < 10:
            continue
        m = a + int(np.argmax(d[a:b]))
        if d[m] <= 0 or m <= 0 or m >= d.size - 1:
            continue
        lo = max(0, m - int(0.4 * fs))
        vmin = float(np.min(y[lo : m + 1]))
        t_foot = tp[m] - (y[m] - vmin) / d[m]
        hi = min(y.size, m + int(0.4 * fs))
        t_peak = tp[m + int(np.argmax(y[m:hi]))]
        out["foot"].append(t_foot - t0)
        out["slope"].append(tp[m] - t0)
        out["peak"].append(t_peak - t0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--time-budget", type=float, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t_start = time.time()
    slowest = 0.0
    with out.open("a") as fh:
        for c in json.loads(Path(a.cases).read_text())["cases"]:
            if c in done:
                continue
            if time.time() - t_start + max(2 * slowest, 5) > a.time_budget:
                break
            t1 = time.time()
            res: dict = {"case": c}
            try:
                case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG})
                e, p = case.tracks.get(ECG), case.tracks.get(PPG)
                if e is None or p is None or not e.record_times.size or not p.record_times.size:
                    res["skip"] = "no ECG_II or PLETH"
                else:
                    fs = float(e.srate)
                    te, tp = sample_times(e), sample_times(p)
                    xe, xp = e.physical(), p.physical()
                    # how the two tracks' record stamps relate
                    common = np.intersect1d(np.round(e.record_times, 6), np.round(p.record_times, 6))
                    res["records"] = {"ecg": int(e.record_times.size), "ppg": int(p.record_times.size), "same_stamp": int(common.size)}
                    lo, hi = max(te[0], tp[0]), min(te[-1], tp[-1])
                    rng = random.Random(c)
                    hist = {k: np.zeros(EDGES.size - 1, int) for k in ("foot", "slope", "peak", "corr")}
                    vals = {k: [] for k in hist}
                    used = 0
                    rrs = []
                    for _ in range(a.windows * 3):
                        if used >= a.windows or hi - lo < a.window_s + 10:
                            break
                        s = rng.uniform(lo, hi - a.window_s)
                        ea, eb = np.searchsorted(te, s), np.searchsorted(te, s + a.window_s)
                        pa, pb = np.searchsorted(tp, s - 1), np.searchsorted(tp, s + a.window_s + 1)
                        # continuous in both tracks: sample spacing within 1% of nominal
                        if eb - ea < 0.95 * a.window_s * fs or pb - pa < 0.95 * (a.window_s + 2) * fs:
                            continue
                        if np.max(np.abs(np.diff(te[ea:eb]) - 1 / fs)) > 0.01 / fs or np.max(np.abs(np.diff(tp[pa:pb]) - 1 / fs)) > 0.01 / fs:
                            continue
                        xw, pw = xe[ea:eb], xp[pa:pb]
                        if not np.isfinite(xw).all() or not np.isfinite(pw).all() or np.ptp(pw) == 0 or np.ptp(xw) == 0:
                            continue
                        r = analyse_window(te[ea:eb], xw, tp[pa:pb], pw, fs)
                        rpk = r_peaks(xw, fs)
                        if rpk.size > 2:
                            rrs.append(float(np.median(np.diff(te[ea:eb][rpk]))))
                        used += 1
                        for k in hist:
                            v = np.array(r[k])
                            hist[k] += np.histogram(v, EDGES)[0]
                            vals[k] += r[k]
                    res["windows"] = used
                    res["beats"] = len(vals["foot"])
                    res["rr_median_s"] = float(np.median(rrs)) if rrs else None
                    for k in hist:
                        v = np.array(vals[k])
                        res[k] = {"hist": hist[k].tolist(),
                                  "q": [float(x) for x in np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])] if v.size else None}
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"
            res["seconds"] = time.time() - t1
            fh.write(json.dumps(res) + "\n")
            slowest = max(slowest, res["seconds"])
    print(json.dumps({"done": sum(1 for _ in out.open()), "seconds": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
