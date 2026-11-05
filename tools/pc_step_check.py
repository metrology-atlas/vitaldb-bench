# SPDX-License-Identifier: MIT
"""Is a simultaneous jump of the PC-stamped numerics a PC clock step or a recorder stall? (metrology-atlas study
recorder-pc-steps). This machine only.

    python tools/pc_step_check.py --root R --events 184:10020.0 3805:13736.4 5213:15026.6 --out pc_step_check.json

A PC clock step of D moves every record the recorder stamps on arrival (Solar8000 and Primus numerics) by D for
good, while waveforms stamped from their device's own sample count (SNUADC) carry on; so after it the numerics sit
D later against the waveforms. A recorder stall of D loses or delays the numerics for D and then they are stamped
on the right time again: their lag behind the waveforms is unchanged. Per event, before (``--win`` s ending
``--gap`` s before it) and after (the same, starting ``--gap`` s after it):

* the lag of Solar8000/ART_MBP behind the mean of SNUADC/ART per beat window, and of Solar8000/HR behind the heart
  rate from SNUADC/ECG_II R peaks: each numeric is compared on a 0.5 s grid with the waveform quantity shifted by
  -60..+60 s, both differenced over 10 s so slow trends do not dominate; the lag is the shift of the highest
  correlation;
* every stamp jump (interval more than 0.3 s from the track's median) within ``--around`` s of the event in
  SNUADC/ECG_II, SNUADC/ART, BIS/EEG1_WAV, Primus/CO2, Primus/AWP, Solar8000/HR and Primus/ETCO2, and the next
  jumps after it in SNUADC/ECG_II and BIS/EEG1_WAV.

The lag change (after minus before) is about D for a PC step and about 0 for a stall. Writes one JSON file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import numpy as np
from vitaldb_reader.vital_file import read_vital

sys.path.insert(0, str(Path(__file__).parent))
from pat import r_peaks, sample_times  # noqa: E402

TRACKS = ("SNUADC/ECG_II", "SNUADC/ART", "BIS/EEG1_WAV", "Primus/CO2", "Primus/AWP", "Solar8000/HR", "Primus/ETCO2")
GRID = 0.5
MAXLAG = 60.0
DIFF = 10.0


def wave(tr) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample times on the track's own stamps (pat.sample_times) and the physical values."""
    return sample_times(tr).astype(float), np.nan_to_num(tr.physical().astype(float)), float(tr.srate)


def jumps(tr, t0: float) -> list[tuple[float, float]]:
    t = tr.record_times.astype(float) - t0
    if t.size < 20:
        return []
    d = np.diff(t)
    m = float(np.median(d))
    k = np.flatnonzero(np.abs(d - m) > 0.3)
    return [(round(float(t[i + 1]), 2), round(float(d[i] - m), 3)) for i in k]


def r_rate(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Heart rate (bpm) at each R peak (pat.r_peaks), from the RR interval ending there."""
    rt = t[r_peaks(x, fs)]
    rr = np.diff(rt)
    ok = (rr > 0.3) & (rr < 2.0)
    return rt[1:][ok], 60.0 / rr[ok]


def art_mean(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean arterial pressure over successive 2 s windows."""
    w = int(2 * fs)
    m = x.size // w
    if m < 2:
        return np.empty(0), np.empty(0)
    return t[: m * w].reshape(m, w).mean(1), x[: m * w].reshape(m, w).mean(1)


def lag(num_t, num_v, ref_t, ref_v, lo, hi) -> dict:
    """Lag (s) of a numeric behind a waveform quantity in [lo, hi): the shift that best aligns them."""
    g = np.arange(lo, hi, GRID)
    sel = (num_t >= lo - 30) & (num_t < hi + 30)
    if sel.sum() < 20 or ref_t.size < 20:
        return {"lag_s": None, "r": None}
    y = np.interp(g, num_t[sel], num_v[sel])
    k = int(DIFF / GRID)
    dy = y[k:] - y[:-k]
    best = (None, -2.0)
    for L in np.arange(-MAXLAG, MAXLAG + GRID / 2, GRID):
        x = np.interp(g - L, ref_t, ref_v, left=np.nan, right=np.nan)
        dx = x[k:] - x[:-k]
        ok = np.isfinite(dx) & np.isfinite(dy)
        if ok.sum() < 50 or np.std(dx[ok]) == 0 or np.std(dy[ok]) == 0:
            continue
        r = float(np.corrcoef(dx[ok], dy[ok])[0, 1])
        if r > best[1]:
            best = (float(L), r)
    return {"lag_s": best[0], "r": round(best[1], 3) if best[0] is not None else None}


def one(root: Path, case: int, te: float, win: float, gap: float, around: float) -> dict:
    v = read_vital(root / "vital_files" / f"{case:04d}.vital")
    t0 = min(tr.record_times[0] for tr in v.tracks.values() if tr.record_times.size)
    out = {"case": case, "event_s": te, "jumps_near": {}, "next_jumps": {}, "lags": {}}
    for name in TRACKS:
        tr = v.tracks.get(name)
        if tr is None or not tr.record_times.size:
            continue
        j = jumps(tr, t0)
        out["jumps_near"][name] = [x for x in j if abs(x[0] - te) <= around]
        if name in ("SNUADC/ECG_II", "BIS/EEG1_WAV"):
            out["next_jumps"][name] = [x for x in j if x[0] > te + around][:3]
    pairs = []
    a, art = v.tracks.get("SNUADC/ART"), v.tracks.get("Solar8000/ART_MBP")
    if a is not None and art is not None and a.record_times.size and art.record_times.size:
        t, x, fs = wave(a)
        rt, rv = art_mean(t - t0, x, fs)
        ok = (rv > 20) & (rv < 200)
        pairs.append(("ART_MBP", art.record_times.astype(float) - t0, art.physical().astype(float), rt[ok], rv[ok]))
    e, hr = v.tracks.get("SNUADC/ECG_II"), v.tracks.get("Solar8000/HR")
    if e is not None and hr is not None and e.record_times.size and hr.record_times.size:
        t, x, fs = wave(e)
        sel = (t - t0 > te - gap - win - MAXLAG - 30) & (t - t0 < te + gap + win + MAXLAG + 30)
        rt, rv = r_rate(t[sel] - t0, x[sel], fs)
        pairs.append(("HR", hr.record_times.astype(float) - t0, hr.physical().astype(float), rt, rv))
    for name, nt, nv, rt, rv in pairs:
        okn = np.isfinite(nv) & (nv > 0)
        b = lag(nt[okn], nv[okn], rt, rv, te - gap - win, te - gap)
        f = lag(nt[okn], nv[okn], rt, rv, te + gap, te + gap + win)
        ch = (f["lag_s"] - b["lag_s"]) if b["lag_s"] is not None and f["lag_s"] is not None else None
        out["lags"][name] = {"before": b, "after": f, "change_s": ch}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--events", nargs="+", required=True, help="CASE:SECONDS (from the recording's first record)")
    ap.add_argument("--win", type=float, default=1200.0)
    ap.add_argument("--gap", type=float, default=60.0)
    ap.add_argument("--around", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    for ev in a.events:
        c, s = ev.split(":")
        res.append(one(a.root, int(c), float(s), a.win, a.gap, a.around))
        print(json.dumps({"case": res[-1]["case"], "lags": res[-1]["lags"]}))
    Path(a.out).write_text(json.dumps({"params": vars(a) | {"root": str(a.root)}, "events": res}, indent=1) + "\n")


if __name__ == "__main__":
    main()
