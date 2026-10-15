# SPDX-License-Identifier: MIT
"""Effective temporal resolution of ECG and PPG fiducials in VitalDB (Ruffolo et al. 2025, section 3.3).

    python tools/fid_res.py --root R --cases cases.json --out fid_res.jsonl [--part 0 --of 4] [--minutes 60]

On the longest continuous stretch (up to ``--minutes``) of SNUADC/ECG_II and SNUADC/PLETH in each case,
fiducials are found with several detectors, some on the raw stored samples and some on filtered copies:

ECG   ``ecg_bp``      extreme of the 5-25 Hz band-passed ECG (sample)
      ``ecg_raw``     extreme of the raw ECG within +/-20 ms of it
      ``ecg_nk``      NeuroKit2 ecg_clean + ecg_peaks (as in the paper)
PPG   ``ppg_kav_raw`` BioSPPy Kavsaoglu 2016 onsets on the raw PPG
      ``ppg_kav_f``   the same on the BioSPPy-filtered PPG (Butterworth 1-8 Hz, order 4; as in the paper)
      ``ppg_foot_raw`` raw minimum in the 0.4 s before each upstroke
      ``ppg_up_raw``  largest raw first difference near each upstroke
      ``ppg_up_f``    steepest point of the 0.5-10 Hz band-passed PPG
      ``ppg_peak_raw`` raw maximum in the 0.4 s after each upstroke

For each detector, physiosync.fiducial_resolution on the IBIs (R-R and onset-onset durations) and on the
differences of consecutive IBIs (the paper's figure 4). Output: histograms and test statistics only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from pat import ECG, PPG, r_peaks, sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import (
    read_vital,
)

from physiosync import fiducial_resolution

warnings.filterwarnings("ignore")


def ecg_detectors(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    out = {}
    rp = r_peaks(x, fs)
    out["ecg_bp"] = rp
    sos = signal.butter(2, (5, 25), btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)
    sign = 1.0 if rp.size and np.median(y[rp]) >= 0 else -1.0
    w = int(0.02 * fs)
    out["ecg_raw"] = np.unique([max(p - w, 0) + int(np.argmax(sign * x[max(p - w, 0) : p + w + 1])) for p in rp]).astype(np.int64)
    try:
        import neurokit2 as nk

        _, info = nk.ecg_peaks(nk.ecg_clean(x, sampling_rate=int(fs)), sampling_rate=int(fs), method="neurokit")
        out["ecg_nk"] = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    except Exception as e:  # noqa: BLE001
        out["ecg_nk_error"] = repr(e)
    return out


def ppg_detectors(raw: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    out = {}
    x = raw.astype(np.float64)
    sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)
    d = np.gradient(y)
    up, _ = signal.find_peaks(d, distance=int(0.3 * fs), prominence=0.3 * (np.quantile(d, 0.99) - np.quantile(d, 0.5)))
    out["ppg_up_f"] = up
    dr = np.diff(x)
    w = int(0.06 * fs)
    ups = [u for u in up if w <= u < dr.size - w]
    out["ppg_up_raw"] = np.array([u - w + int(np.argmax(dr[u - w : u + w])) for u in ups], dtype=np.int64)
    lo4 = int(0.4 * fs)
    out["ppg_foot_raw"] = np.array([max(u - lo4, 0) + int(np.argmin(x[max(u - lo4, 0) : u + 1])) for u in ups], dtype=np.int64)
    out["ppg_peak_raw"] = np.array([u + int(np.argmax(x[u : u + lo4])) for u in ups if u + lo4 <= x.size], dtype=np.int64)
    try:
        from biosppy.signals import ppg as bppg
        from biosppy.signals import tools as st

        out["ppg_kav_raw"] = np.asarray(bppg.find_onsets_kavsaoglu2016(signal=x, sampling_rate=fs, max_BPM=200)["onsets"], dtype=np.int64)
        f, _, _ = st.filter_signal(signal=x, ftype="butter", band="bandpass", order=4, frequency=[1, 8], sampling_rate=fs)
        out["ppg_kav_f"] = np.asarray(bppg.find_onsets_kavsaoglu2016(signal=f, sampling_rate=fs, max_BPM=200)["onsets"], dtype=np.int64)
    except Exception as e:  # noqa: BLE001
        out["ppg_kav_error"] = repr(e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--max-k", type=int, default=16)
    ap.add_argument("--time-budget", type=float, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    cases = [c for i, c in enumerate(json.loads(Path(a.cases).read_text())["cases"]) if i % a.of == a.part]
    t_start, slowest = time.time(), 0.0
    with out.open("a") as fh:
        for c in cases:
            if c in done:
                continue
            if time.time() - t_start + max(1.5 * slowest, 5) > a.time_budget:
                break
            t1 = time.time()
            res: dict = {"case": c}
            case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, PPG})
            e, p = case.tracks.get(ECG), case.tracks.get(PPG)
            if e is None or p is None or not e.record_times.size or not p.record_times.size:
                res["skip"] = "no ECG_II or PLETH"
            else:
                fs = float(e.srate)
                res["srate"] = {"ecg": fs, "ppg": float(p.srate)}
                te, tp = sample_times(e), sample_times(p)
                gap = np.flatnonzero(np.abs(np.diff(te) - 1 / fs) > 0.01 / fs)
                be = np.r_[0, gap + 1, te.size]
                k = int(np.argmax(np.diff(be)))
                ea, eb = be[k], min(be[k + 1], be[k] + int(a.minutes * 60 * fs))
                pa, pb = np.searchsorted(tp, te[ea]), np.searchsorted(tp, te[eb - 1])
                if eb - ea < 600 * fs or pb - pa < 600 * fs or np.max(np.abs(np.diff(tp[pa:pb]) - 1 / fs)) > 0.01 / fs:
                    res["skip"] = "no 10 min continuous stretch"
                else:
                    res["minutes"] = round((eb - ea) / fs / 60, 1)
                    xe = e.physical()[ea:eb].astype(np.float64)
                    raw_p = np.asarray(p.values[pa:pb])
                    fids = {**ecg_detectors(xe, fs), **ppg_detectors(raw_p, fs)}
                    rng = (int(60 / 200 * fs), int(60 / 30 * fs))
                    res["detectors"] = {}
                    for name, idx in fids.items():
                        if name.endswith("_error"):
                            res[name] = idx
                            continue
                        idx = np.unique(idx)
                        r_ibi = fiducial_resolution(idx, fs=fs, ibi_range=rng, max_k=a.max_k)
                        r_dd = fiducial_resolution(idx, fs=fs, ibi_range=rng, max_k=a.max_k, quantity="ibi_diff", diff_limit=40)
                        res["detectors"][name] = {"beats": int(idx.size), "ibi": r_ibi.to_dict(), "ibi_diff": r_dd.to_dict()}
            res["seconds"] = round(time.time() - t1, 2)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            slowest = max(slowest, res["seconds"])
    print(json.dumps({"part": a.part, "done": sum(1 for _ in out.open()), "seconds": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
