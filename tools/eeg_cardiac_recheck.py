# SPDX-License-Identifier: MIT
"""Recheck of the cardiac artefact in BIS EEG with the SNUADC/BIS offset known. The first pilot
(eeg_sync_pilot) averaged the EEG -0.3..+0.5 s around each R peak on raw stamps, but the same event is
stamped ~0.8 s earlier in BIS, so an artefact would have fallen outside that window. Here both devices are on
the stamp-model times (as tools/sa_corrected.py, variant model) and the R-triggered average spans -2..+1 s.
Per case, three 10-min windows: peak-to-noise of the average, its lag, single-beat SNR; and the case's
burst-measured EEG-ECG offset for comparison. Aggregates only leave this machine.

    python tools/eeg_cardiac_recheck.py --root R --cases ... --units U --step-rates S --bursts corr.jsonl --out out.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
import vent_sync_pilot as V
from pat import r_peaks
from physiosync import corrected_times, stamp_clock
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

ap = argparse.ArgumentParser()
for k in ("--root", "--cases", "--units", "--step-rates", "--bursts", "--out"):
    ap.add_argument(k, required=True)
a = ap.parse_args()
U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
sr = json.loads(Path(a.step_rates).read_text())
B = {json.loads(l)["case"]: json.loads(l) for l in open(a.bursts)}


def bis_times(tr, u):
    st = tr.record_offsets[: tr.record_times.size]
    clk = stamp_clock(tr.record_times, st, float(tr.srate), correction_sign=1)
    if clk["rate_source"] == "fit":
        un = u.get("bis_unit_mean_ppm") or u.get("bis_drift_ppm")
        if un:
            clk = stamp_clock(tr.record_times, st, float(tr.srate), rate_ppm=float(un), correction_sign=1)
    return corrected_times(clk, st, tr.record_sizes[: st.size])


done = set()
if Path(a.out).exists():
    done = {json.loads(l)["case"] for l in open(a.out)}
with open(a.out, "a") as fh:
    for c in [int(x) for x in a.cases.split(",")]:
        if c in done:
            continue
        res = {"case": c}
        try:
            v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={"SNUADC/ECG_II", "BIS/EEG1_WAV", "BIS/EEG2_WAV"})
            u = U.get(c, {})
            te, _ = V.snu_times(v.tracks["SNUADC/ECG_II"], u, sr)
            xe = v.tracks["SNUADC/ECG_II"].physical().astype(float)
            g = [x for x in B.get(c, {}).get("model", {}).get("burst", {}).get("ecg~eeg1", [])
                 if x[2] >= 0.3 and x[2] - x[3] >= 0.15]
            res["burst_offset_ms"] = float(np.median([x[1] for x in g])) if len(g) >= 3 else None
            for ch in ("EEG1", "EEG2"):
                tr = v.tracks[f"BIS/{ch}_WAV"]
                tb = bis_times(tr, u)
                xb = tr.physical().astype(float)[: tb.size]
                fs = float(tr.srate)
                yb = signal.sosfiltfilt(signal.butter(2, (1, 40), btype="bandpass", fs=fs, output="sos"),
                                        np.nan_to_num(xb - np.nanmedian(xb)))
                lags = np.arange(-2.0, 1.0, 1 / fs)
                rows = []
                span = te[-1] - te[0]
                for f in (0.3, 0.5, 0.7):
                    w0 = te[0] + f * span - 300
                    m = (te >= w0) & (te < w0 + 600)
                    idx = np.flatnonzero(m)
                    if idx.size < 500 * 300:
                        continue
                    seg = np.where(np.isfinite(xe[idx]), xe[idx], np.nanmedian(xe[idx]))
                    rt = te[idx][r_peaks(seg, 500.0)]
                    S = []
                    for r in rt:
                        j = np.searchsorted(tb, r + lags)
                        if j[0] > 0 and j[-1] < tb.size and np.all(np.abs(tb[j] - (r + lags)) < 1.5 / fs):
                            S.append(yb[j])
                    if len(S) < 100:
                        continue
                    S = np.array(S)
                    avg = S.mean(0) - S.mean(0).mean()
                    noise = S.std(0).mean() / np.sqrt(len(S))
                    k = int(np.argmax(np.abs(avg)))
                    rows.append({"beats": len(S), "pnr": round(float(abs(avg[k]) / noise), 1), "lag_ms": round(float(lags[k] * 1000), 1),
                                 "peak_uv": round(float(avg[k]), 2), "single_snr": round(float(abs(avg[k]) / S.std(0).mean()), 3)})
                res[ch] = rows
        except Exception as ex:  # noqa: BLE001
            res["error"] = f"{type(ex).__name__}: {ex}"[:300]
        fh.write(json.dumps(res) + "\n")
        fh.flush()
