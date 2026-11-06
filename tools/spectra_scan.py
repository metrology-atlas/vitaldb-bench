# SPDX-License-Identifier: MIT
"""Power spectra of every waveform, every gap-free minute, across the release (metrology-atlas: each signal's
spectrum figure, and the study of the waveforms' frequency content). This machine only for the scan; the
summaries and figures leave.

    python tools/spectra_scan.py synth --out $W/spectra/synth.json
    python tools/spectra_scan.py scan --root $R --out $W/spectra/part_K --part K --of N [--time-budget S]
    python tools/spectra_agg.py $W/spectra --out $W/spectra/agg

Every waveform track of at least 50 Hz, on its value grid (integers as stored; floats on the grid of their smallest
step, as tools/waveform_survey.py), in one-minute gap-free epochs on an absolute 60 s grid. Welch spectrum per epoch
(Hann, 50% overlap, segment 2**round(log2(8 fs)) samples, mean removed per segment), in steps^2/Hz; levels in dB
above the rounding floor q = 1 / (6 fs).

scan writes, per part, part_K.jsonl (one line per recording) and part_K.npz (per-minute arrays):

* **Per recording and track:** the median spectrum over the recording's minutes, and its 25th and 75th
  percentiles, on 160 log-spaced bins from 0.1 Hz to Nyquist (dB above rounding), with the step in physical units so
  a figure can show physical power per Hz; the number of minutes.
* **Per minute and track** (written with the lines every 20 recordings, so a stopped run resumes cleanly): floor_db (80-98% of Nyquist), lf_db (0.2-1 Hz), mid_db (5-15 Hz), hi_db (30-45 Hz) and
  top_db (55-75% of Nyquist, mains lines excluded), and the mains lines at 60, 120 and 180 Hz (median within +-0.6 Hz
  over the median of a ring 2-6 Hz either side, dB: negative a notch, positive a line); and a coarse shape: 24
  log-spaced bins from 1 Hz to Nyquist (dB above rounding, float16), for finding changes within a recording.
* **SNUADC channel pairs and the BIS's two channels:** cross-spectra summed over the minutes both channels have;
  coherence and phase at 60, 120 and 180 Hz and coherence in a reference band (55-70% of Nyquist, mains excluded),
  with the phase as a time difference: later_us, positive when the second channel's samples are taken later than
  the first's (its copy of the line is ahead in phase), checked by synth. A mains line reaches each channel by its
  own path, so the difference is a sampling delay only where the two pick it up the same way (ECG II and V5, the
  BIS's two channels); between unlike channels it is mostly coupling phase.

synth checks the low-pass fit (corner recovered within 10%; the order only when the corner is below 0.3 of Nyquist,
flagged order_ok), the notch measure and the channel delay (about 10 us at two minutes) on synthetic data. The
summaries and figures are made by tools/spectra_agg.py from these files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from waveform_survey import nseg, on_grid, runs  # noqa: E402

MIN_FS = 50.0
EPOCH_S = 60.0
NBIN, NSHAPE = 160, 24
MAINS = (60.0, 120.0, 180.0)
PAIRS = (("SNUADC/ECG_II", "SNUADC/ART"), ("SNUADC/ECG_II", "SNUADC/PLETH"), ("SNUADC/ART", "SNUADC/PLETH"),
         ("SNUADC/ECG_II", "SNUADC/ECG_V5"), ("SNUADC/ART", "SNUADC/CVP"), ("BIS/EEG1_WAV", "BIS/EEG2_WAV"))
FLUSH = 20          # recordings per write of lines and minute rows
FEAT = ("floor_db", "lf_db", "mid_db", "hi_db", "top_db", "m60_db", "m120_db", "m180_db")


def bins(fs: float) -> np.ndarray:
    return np.geomspace(0.1, fs / 2, NBIN + 1)


def shape_edges(fs: float) -> np.ndarray:
    return np.geomspace(1.0, fs / 2, NSHAPE + 1)


def binned(f: np.ndarray, P: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Median of the Welch values in each bin; a bin narrower than the resolution (none inside it) takes the value
    interpolated at its centre (log frequency), between the first nonzero frequency and Nyquist."""
    idx = np.digitize(f, edges) - 1
    out = np.full(edges.size - 1, np.nan)
    order = np.argsort(idx, kind="stable")
    ks, starts = np.unique(idx[order], return_index=True)
    for k, s, e in zip(ks, starts, list(starts[1:]) + [idx.size]):
        if 0 <= k < out.size:
            out[k] = np.median(P[order[s:e]])
    c = np.sqrt(edges[:-1] * edges[1:])
    miss = np.isnan(out) & (c >= f[1]) & (c <= f[-1])
    if miss.any():
        out[miss] = np.interp(np.log(c[miss]), np.log(f[1:]), P[1:])
    return out


def db(x: np.ndarray | float, q: float):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(np.asarray(x) / q)


def band(f, P, lo, hi, q, excl_mains=False):
    sel = (f >= lo) & (f <= hi)
    if excl_mains:
        for m in MAINS:
            sel &= np.abs(f - m) > 2.0
    return float(db(np.median(P[sel]), q)) if sel.sum() >= 3 and np.median(P[sel]) > 0 else np.nan


def mains_level(f, P, m):
    pk = np.abs(f - m) <= 0.6
    ring = (np.abs(f - m) >= 2) & (np.abs(f - m) <= 6)
    if not pk.any() or ring.sum() < 3 or np.median(P[ring]) <= 0:
        return np.nan
    return float(10 * np.log10(np.median(P[pk]) / np.median(P[ring])))


def features(f, P, fs, q) -> np.ndarray:
    ny = fs / 2
    v = [band(f, P, 0.8 * ny, 0.98 * ny, q), band(f, P, 0.2, 1.0, q), band(f, P, 5, 15, q) if ny > 15 else np.nan,
         band(f, P, 30, 45, q) if ny > 45 else np.nan, band(f, P, 0.55 * ny, 0.75 * ny, q, True)]
    v += [mains_level(f, P, m) if m + 6 < ny else np.nan for m in MAINS]
    return np.array(v, float)


def epochs(tr) -> dict[int, tuple[int, int]]:
    """Gap-free one-minute epochs on the absolute 60 s grid: minute index -> (start, end) value index."""
    fs = float(tr.srate)
    out = {}
    for a, b, t in runs(tr):
        k = int(np.ceil(t / EPOCH_S))
        while True:
            s = a + int(round((k * EPOCH_S - t) * fs))
            e = s + int(EPOCH_S * fs)
            if e > b:
                break
            out[k] = (s, e)
            k += 1
    return out


def track_spectra(tr, minute_rows: list, case: int, name: str) -> tuple[dict | None, dict]:
    fs = float(tr.srate)
    integer = bool(tr.is_integer_format)
    vals = np.asarray(tr.values)
    q = 1.0 / (6.0 * fs)
    E = epochs(tr)
    spec, grid = [], {}
    step_phys = None
    for k, (s, e) in E.items():
        g, step = on_grid(vals[s:e], integer)
        if g is None:
            continue
        f, P = signal.welch(g.astype(float), fs, window="hann", nperseg=min(nseg(fs), g.size), detrend="constant")
        spec.append(binned(f, P, bins(fs)))
        sh = db(binned(f, P, shape_edges(fs)), q).astype(np.float16)
        minute_rows.append((case, name, k, features(f, P, fs, q), sh))
        grid[k] = g
        if step_phys is None:
            step_phys = float(step * tr.gain) if integer else float(step)
    if not spec:
        return None, grid
    S = np.array(spec)
    with np.errstate(all="ignore"):
        med, p25, p75 = (db(np.nanpercentile(S, p, axis=0), q) for p in (50, 25, 75))
    r2 = lambda x: [None if not np.isfinite(v) else round(float(v), 2) for v in x]  # noqa: E731
    return {"fs": fs, "integer": integer, "step_phys": step_phys, "unit": tr.unit, "minutes": int(len(spec)),
            "psd_db": r2(med), "psd_db_p25": r2(p25), "psd_db_p75": r2(p75)}, grid


def pair_spectra(ga: dict, gb: dict, fs: float) -> dict | None:
    ks = sorted(set(ga) & set(gb))
    if not ks:
        return None
    n = min(nseg(fs), int(EPOCH_S * fs))
    Sxy = Sxx = Syy = None
    for k in ks:
        x, y = ga[k].astype(float), gb[k].astype(float)
        if x.size != y.size:
            continue
        f, pxy = signal.csd(x, y, fs, window="hann", nperseg=n, detrend="constant")
        _, pxx = signal.welch(x, fs, window="hann", nperseg=n, detrend="constant")
        _, pyy = signal.welch(y, fs, window="hann", nperseg=n, detrend="constant")
        Sxy = pxy if Sxy is None else Sxy + pxy
        Sxx = pxx if Sxx is None else Sxx + pxx
        Syy = pyy if Syy is None else Syy + pyy
    if Sxy is None:
        return None
    coh = np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-300)
    out = {"minutes": len(ks)}
    for m in MAINS:
        if m + 2 < fs / 2:
            i = int(np.argmin(np.abs(f - m)))
            ph = float(np.angle(Sxy[i]))           # scipy: conj(X) Y, positive when y's line is ahead in phase
            out[f"coh_{int(m)}"] = round(float(coh[i]), 4)
            out[f"later_us_{int(m)}"] = round(ph / (2 * np.pi * m) * 1e6, 2)
    ref = (f >= 0.55 * fs / 2) & (f <= 0.70 * fs / 2)
    for m in MAINS:
        ref &= np.abs(f - m) > 2
    out["coh_ref"] = round(float(np.median(coh[ref])), 4) if ref.any() else None
    return out


def scan(a) -> None:
    """Resumable. The per-recording lines and the per-minute rows are written together, every FLUSH recordings
    (minute rows first), so a stopped run loses at most the batch in hand and never leaves a recording with its line
    but without its minutes. On start, a recording counts as done only if it has a line and, when it has tracks, its
    minutes in this part's .npz files; the others are scanned again (agg keeps a recording's last line)."""
    from vitaldb_reader.vital_file import TRACK_WAVE, read_vital
    root = Path(a.root)
    cases = sorted(int(p.stem) for p in (root / "vital_files").glob("*.vital"))
    cases = [c for i, c in enumerate(cases) if i % a.of == a.part]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jl = out.with_suffix(".jsonl")
    npzs = sorted(out.parent.glob(out.name + "_m*.npz"))
    have = set()
    for f in npzs:
        have |= set(np.unique(np.load(f)["case"]).tolist())
    done = set()
    if jl.exists():
        for x in jl.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if not r.get("tracks") or r["case"] in have:
                    done.add(r["case"])
    t0 = time.time()
    n_npz = len(npzs)
    rows: list = []
    lines: list = []

    def flush():
        nonlocal n_npz, rows, lines
        if rows:
            save_minutes(rows, out, n_npz)
            n_npz += 1
        with jl.open("a") as fh:
            fh.writelines(lines)
        rows, lines = [], []

    for c in cases:
        if c in done:
            continue
        if time.time() - t0 > a.time_budget:
            break
        t1 = time.time()
        rec = {"case": c, "tracks": {}, "pairs": {}}
        try:
            v = read_vital(root / "vital_files" / f"{c:04d}.vital")
            grids, fss = {}, {}
            for name, tr in v.tracks.items():
                if tr.track_type != TRACK_WAVE or float(tr.srate) < MIN_FS or not tr.record_times.size:
                    continue
                s, g = track_spectra(tr, rows, c, name)
                if s:
                    rec["tracks"][name] = s
                    grids[name], fss[name] = g, float(tr.srate)
            for x, y in PAIRS:
                if x in grids and y in grids and fss[x] == fss[y]:
                    p = pair_spectra(grids[x], grids[y], fss[x])
                    if p:
                        rec["pairs"][f"{x}|{y}"] = p
            del v, grids
        except Exception as ex:  # noqa: BLE001
            rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
        rec["secs"] = round(time.time() - t1, 1)
        lines.append(json.dumps(rec) + "\n")
        if len(lines) >= FLUSH:
            flush()
    flush()
    print(json.dumps({"part": a.part, "seconds": round(time.time() - t0, 1)}))


def save_minutes(rows: list, out: Path, k: int) -> None:
    names = sorted({r[1] for r in rows})
    np.savez_compressed(f"{out}_m{k:03d}.npz", case=np.array([r[0] for r in rows], np.int32),
                        track=np.array([names.index(r[1]) for r in rows], np.int16), track_names=np.array(names),
                        minute=np.array([r[2] for r in rows], np.int64),
                        feat=np.array([r[3] for r in rows], np.float32), feat_names=np.array(FEAT),
                        shape=np.array([r[4] for r in rows], np.float16))


# --------------------------------------------------------------------------------------------- low-pass fitting
def rolloff_model(f, la, beta, lfc, n, ln):
    return 10 * np.log10(10 ** la * f ** (-beta) / (1 + (f / 10 ** lfc) ** (2 * n)) + 10 ** ln)


def fit_rolloff(f: np.ndarray, d: np.ndarray, fs: float, fmin: float = 8.0) -> dict | None:
    """Fit dB-above-rounding = 10 log10(A f^-beta / (1 + (f/fc)^(2n)) + N) above fmin, mains lines excluded:
    a power-law signal through an n-th order low-pass at fc, over a white floor N. Returns fc, n, beta, the floor
    and the fit's RMS residual (dB), or None."""
    from scipy.optimize import least_squares
    sel = np.isfinite(d) & (f >= fmin) & (f <= 0.98 * fs / 2)
    for m in MAINS:
        sel &= np.abs(f - m) > 3
    if sel.sum() < 12:
        return None
    x, y = f[sel], d[sel]
    best = None
    for fc0 in (25.0, 40.0, 60.0, 100.0, 150.0):
        if fc0 >= fs / 2:
            continue
        p0 = [y[0] / 10 + 1, 1.0, np.log10(fc0), 2.0, np.median(y[-5:]) / 10]
        try:
            r = least_squares(lambda p: rolloff_model(x, *p) - y, p0,
                              bounds=([-5, -1, np.log10(5), 0.5, -3], [15, 4, np.log10(fs / 2), 8, 5]))
        except Exception:  # noqa: BLE001
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    la, beta, lfc, n, ln = best.x
    rms = float(np.sqrt(np.mean(best.fun ** 2)))
    return {"fc_hz": round(float(10 ** lfc), 2), "order": round(float(n), 2), "order_ok": bool(10 ** lfc < 0.3 * fs / 2),
            "beta": round(float(beta), 2),
            "floor_db": round(float(10 * ln), 2), "rms_db": round(rms, 2)}


def synth(a) -> None:
    """Check the low-pass fit, the notch measure and the channel delay on synthetic data."""
    rng = np.random.default_rng(1)
    out = {"rolloff": [], "notch": [], "delay": []}
    for fs, fc, n in ((500.0, 25.0, 2), (500.0, 40.0, 2), (500.0, 40.0, 4), (500.0, 100.0, 2), (500.0, 150.0, 3),
                      (128.0, 47.4, 4)):
        N = int(600 * fs)
        w = np.fft.rfft(rng.normal(0, 1, N))
        ff = np.fft.rfftfreq(N, 1 / fs)
        w[1:] *= ff[1:] ** -0.75                          # a power-law signal, steeper than white
        x = np.fft.irfft(w, N)
        sos = signal.butter(n, fc, fs=fs, output="sos")
        x = signal.sosfilt(sos, x)
        x = x / x.std() * 40 + rng.normal(0, 0.3, N)      # 40 steps of signal, 0.3 step of white noise after
        g = np.round(x).astype(np.int64)
        f, P = signal.welch(g.astype(float), fs, window="hann", nperseg=nseg(fs), detrend="constant")
        e = bins(fs)
        dbv = db(binned(f, P, e), 1 / (6 * fs))
        r = fit_rolloff(np.sqrt(e[:-1] * e[1:]), dbv, fs)
        out["rolloff"].append({"fs": fs, "fc_true": fc, "order_true": n, **(r or {})})
    fs = 500.0
    N = int(120 * fs)
    t = np.arange(N) / fs
    base = rng.normal(0, 3, N) + 5 * np.sin(2 * np.pi * 60 * t)
    for notch in (False, True):
        x = base.copy()
        if notch:
            b, aa = signal.iirnotch(60.0, 30.0, fs)
            x = signal.filtfilt(b, aa, rng.normal(0, 3, N))
        g = np.round(x).astype(np.int64)
        f, P = signal.welch(g.astype(float), fs, window="hann", nperseg=nseg(fs), detrend="constant")
        out["notch"].append({"notch": notch, "line": not notch, "m60_db": round(mains_level(f, P, 60.0), 2)})
    for lag_us in (0.0, 37.0, 370.0):                     # one 120 Hz line in two channels, the second sampled later
        line = lambda tt: 4 * np.sin(2 * np.pi * 120 * tt + 0.3)  # noqa: E731
        xa = np.round(line(t) + rng.normal(0, 2, N)).astype(np.int64)
        xb = np.round(line(t + lag_us * 1e-6) + rng.normal(0, 2, N)).astype(np.int64)
        ga = {k: xa[k * int(60 * fs):(k + 1) * int(60 * fs)] for k in range(2)}
        gb = {k: xb[k * int(60 * fs):(k + 1) * int(60 * fs)] for k in range(2)}
        p = pair_spectra(ga, gb, fs)
        out["delay"].append({"b_samples_later_us": lag_us, "later_us_120": p["later_us_120"], "coh_120": p["coh_120"]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    sc = sub.add_parser("scan")
    sc.add_argument("--root", required=True)
    sc.add_argument("--out", required=True, help="path prefix: OUT.jsonl and OUT_mNNN.npz")
    sc.add_argument("--part", type=int, default=0)
    sc.add_argument("--of", type=int, default=1)
    sc.add_argument("--time-budget", type=float, default=1e9)
    a = ap.parse_args()
    {"synth": synth, "scan": scan}[a.cmd](a)


if __name__ == "__main__":
    main()
