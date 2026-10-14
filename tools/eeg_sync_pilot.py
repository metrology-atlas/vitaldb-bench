# SPDX-License-Identifier: MIT
"""Can heartbeats in the BIS EEG synchronise it with the ECG? Pilot, per case, this machine only.

    python tools/eeg_sync_pilot.py --root R --beats DIR --cases 1,2,3 --out out.jsonl [--minutes 10]
        [--ext DIR_WITH_consensus_peaks_AND_hrvmatch]

For a window in the middle of each benchmark stretch (beats_export: sub-sample ECG R peaks):

1. ECG artefact in the EEG: BIS/EEG1_WAV and EEG2_WAV (128 Hz) averaged around every R peak (-0.3..+0.5 s),
   per channel: the size of the averaged deflection against the averaged baseline spread (the artefact's
   signal-to-noise), and the lag of its largest deflection after the R peak.
2. Beat detection in the EEG with consensus_peaks (Goodwin et al. 2023; both polarities, the one with more
   beats kept), and alignment of those beats to the ECG R peaks with hrvmatch (windows of 20 inter-beat
   intervals, density filter, RANSAC): the share of EEG beats within 50 ms of an R peak after alignment,
   the offset and its slope (ppm) over the window.
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
from pat import ECG, sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

warnings.filterwarnings("ignore")
EEG = ("BIS/EEG1_WAV", "BIS/EEG2_WAV")


def window(t: np.ndarray, x: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.searchsorted(t, a), np.searchsorted(t, b)
    return t[i:j], x[i:j]


def triggered(t: np.ndarray, x: np.ndarray, r: np.ndarray, fs: float) -> dict:
    """R-triggered average of a (band-passed) EEG channel."""
    sos = signal.butter(2, (1, 40), btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, np.nan_to_num(x - np.nanmedian(x)))
    lags = np.arange(-0.3, 0.5, 1 / fs)
    segs = []
    for rt in r:
        idx = np.searchsorted(t, rt + lags)
        if idx[0] > 0 and idx[-1] < t.size and np.all(np.abs(t[idx] - (rt + lags)) < 1.5 / fs):
            segs.append(y[idx])
    if len(segs) < 50:
        return {"beats": len(segs)}
    S = np.array(segs)
    avg = S.mean(0)
    base = avg[lags < -0.15]
    noise = S.std(0).mean() / np.sqrt(S.shape[0])        # spread of the average expected from noise alone
    k = int(np.argmax(np.abs(avg - base.mean()) * (lags >= -0.1)))
    return {"beats": int(S.shape[0]), "peak_uv": round(float(avg[k] - base.mean()), 2),
            "snr": round(float(abs(avg[k] - base.mean()) / noise), 1), "lag_ms": round(float(lags[k] * 1000), 1),
            "single_beat_snr": round(float(abs(avg[k] - base.mean()) / S.std(0).mean()), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--ext", default=str(Path.home() / "ext"))
    a = ap.parse_args()
    sys.path[:0] = [str(Path(a.ext) / "consensus_peaks"), str(Path(a.ext) / "hrvmatch")]
    from consensus_peaks.consensus_peaks import get_consensus_peaks
    from consensus_peaks.consensus_signal import get_consensus_signal
    from consensus_peaks.individual_rpeaks import get_individual_rpeaks
    from peakalignment import align_peaks

    with open(a.out, "a") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            res: dict = {"case": c}
            t1 = time.time()
            try:
                z = np.load(Path(a.beats) / f"{c:04d}.npz")
                r_t = z["r_t"]
                mid = 0.5 * (r_t[0] + r_t[-1])
                wa, wb = mid - a.minutes * 30, mid + a.minutes * 30
                r = r_t[(r_t >= wa) & (r_t < wb)]
                v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names=set(EEG) | {ECG})
                res["eeg_tracks"] = [n for n in EEG if n in v.tracks and v.tracks[n].record_times.size]
                for n in res["eeg_tracks"]:
                    tr = v.tracks[n]
                    fs = float(tr.srate)
                    t, x = window(sample_times(tr), tr.physical().astype(np.float64), wa, wb)
                    if t.size < fs * 60:
                        res[n] = {"skip": "no EEG in the window"}
                        continue
                    row = {"fs": fs, "trig": triggered(t, x, r, fs)}
                    # beats in the EEG: consensus over all cleaning x detection pairs, both polarities
                    xx = np.nan_to_num(x - np.nanmedian(x))
                    rp = get_individual_rpeaks(xx, fs)
                    cons = get_consensus_peaks(get_consensus_signal(rp))
                    best = max((False, True), key=lambda inv: len(cons[inv][0]))
                    pk = np.asarray(cons[best][0], dtype=np.int64)
                    pk = pk[(pk >= 0) & (pk < t.size)]
                    row["consensus_beats"], row["inverted"], row["ecg_beats"] = int(pk.size), bool(best), int(r.size)
                    if pk.size >= 60:
                        eeg_ms = t[pk] * 1000.0
                        ecg_ms = r * 1000.0
                        try:
                            aligned = align_peaks(eeg_ms, ecg_ms, eeg_ms.copy(), max_offset=60_000)
                            d = np.array([np.min(np.abs(ecg_ms - x_)) for x_ in aligned])
                            off = aligned - eeg_ms[: aligned.size] if aligned.size == eeg_ms.size else None
                            row["aligned_share_50ms"] = round(float(np.mean(d < 50)), 3)
                            if off is not None and np.isfinite(off).sum() > 10:
                                ok = np.isfinite(off)
                                kk = np.polyfit((eeg_ms[ok] - eeg_ms[0]) / 1000, off[ok], 1)
                                row["offset_ms"] = round(float(np.median(off[ok])), 1)
                                row["offset_slope_ppm"] = round(float(kk[0] * 1000), 2)
                        except Exception as ex:  # noqa: BLE001
                            row["align_error"] = f"{type(ex).__name__}: {ex}"[:200]
                        # simple check without the alignment: EEG beats within 50 ms of an R peak as recorded
                        d0 = np.array([np.min(np.abs(ecg_ms - x_)) for x_ in eeg_ms])
                        row["raw_share_50ms"] = round(float(np.mean(d0 < 50)), 3)
                        row["raw_lag_ms_median"] = round(float(np.median([(x_ - ecg_ms[np.argmin(np.abs(ecg_ms - x_))]) for x_ in eeg_ms])), 1)
                    res[n] = row
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            res["seconds"] = round(time.time() - t1, 1)
            fh.write(json.dumps(res) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
