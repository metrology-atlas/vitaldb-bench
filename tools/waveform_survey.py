# SPDX-License-Identifier: MIT
"""Waveform survey: hidden sample rates (holds), the noise floor against the rounding floor, and the source of the
SNUADC's 120 Hz line (metrology-atlas studies track-value-resolution, primus-co2-update-grid, monitor-lines).
This machine only for the scan; the summary leaves.

    python tools/waveform_survey.py cases --root $R --n 300 --seed 20261106 --out $W/waveform_survey/cases.json
    python tools/waveform_survey.py synth --out $W/waveform_survey/synth.json
    python tools/waveform_survey.py scan --root $R --cases $W/waveform_survey/cases.json \
        --out $W/waveform_survey/part_K.jsonl --part K --of N [--time-budget S]
    python tools/waveform_survey.py agg $W/waveform_survey/part_*.jsonl --out $W/waveform_survey/summary.json

Every waveform track of at least 50 Hz, on its value grid: integer formats on their stored counts; float formats on
the grid of their smallest step between distinct values (snapped to 2, 3 or 4 significant digits first, as float32
storage blurs it), where at least 99.9% of a window's values sit on it (otherwise the window is skipped and
counted). Windows are gap-free: consecutive records within half a sample.

* **Holds** (two 120 s windows, at a third and two thirds of the longest gap-free run). A stored rate higher than the
  real one shows as identical consecutive samples while the signal is moving. Moving samples: the top 20% of
  |q[i+5] - q[i-5]|, where that is also at least 10 steps (a slope of a step per sample or more, at which a truly
  sampled signal rarely repeats). Per window: the share of repeats among all samples and among moving ones,
  the mean length of runs of identical values that start on a moving sample, and the autocorrelation of "value
  changed" at lags 1-15 (a held signal shows its update rhythm there).
* **Noise floor** (up to 30 one-minute gap-free epochs on an absolute 60 s grid, spread over the recording). Welch
  spectrum (Hann, 50% overlap, segment 2**round(log2(8 fs)) samples, mean removed per segment) of the values in
  steps, against the rounding floor q = 1 / (6 fs) steps^2/Hz (uniform error of +-1/2 step): floor_db = 10 log10 of
  the median PSD over 80-98% of Nyquist over q; mid_db over 5-15 Hz and lf_db over 0.2-1 Hz the same way (where
  below Nyquist). 0 dB means the converter's resolution sets the floor. Also, for tracks whose codes stay within
  0-1023 (the SNUADC's 10 bits), the share of samples at 0 or 1023 (the converter's rails).
* **120 Hz source** (SNUADC ECG II, ECG V5, ART, CVP; the longest gap-free run, its first 60 min): the mains lines at
  h x 60 Hz (h = 1..8) of tools/mains_hum.py, each with its frequency as ppm of the harmonic on the SNUADC count. A
  mains harmonic sits at the same ppm as the other mains lines; a line from the oximeter's frame clock would sit
  tens of ppm away (the frame clock runs tens of ppm from the grid). One line's ppm scatters by tens of ppm at these
  signal strengths, so agg pools: over lines with SNR >= 50 and a phase residual <= 10 rad, ppm(120 Hz) minus the
  median ppm of the track's other such lines, against the same difference for the other harmonics (the control:
  lines that are mains by construction). If 120 Hz is mains the two distributions match; if it is a frame line the
  first is shifted.

``synth`` checks the floor and hold measures on synthetic data: a slow signal plus white noise of known SD, rounded
(expected floor: the variance of round(u + n) averaged over the offset u, over 1/12; for SD of half a step or more
this is 10 log10(1 + 12 sd^2) dB, below that the noise does not dither the rounding and the floor is lower), noise
low-passed below the floor band (the floor then measures the stop band, not the noise), and a 25 Hz signal held
onto a 62.5 Hz grid against the same signal sampled at 62.5 Hz.
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
from mains_hum import lines as mains_lines  # noqa: E402

MIN_FS = 50.0
HOLD_WIN_S = 120.0
EPOCH_S = 60.0
MAX_EPOCHS = 30
HUM_TRACKS = ("SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/CVP")
ON_GRID = 0.999
MIN_SLOPE = 1.0          # steps per sample for a sample to count as moving
HUM_SNR, HUM_RES = 50.0, 10.0   # lines kept for the 120 Hz comparison


def nseg(fs: float) -> int:
    return int(2 ** round(np.log2(8 * fs)))


def runs(tr) -> list[tuple[int, int, float]]:
    """Gap-free runs of records as (first value index, end value index, start time)."""
    rt, rs, fs = tr.record_times.astype(float), tr.record_sizes, float(tr.srate)
    if rt.size == 0:
        return []
    ok = np.abs(np.diff(rt) - rs[:-1] / fs) < 0.5 / fs
    out, a = [], 0
    for i, g in enumerate(ok):
        if not g:
            out.append((a, i + 1))
            a = i + 1
    out.append((a, rt.size))
    off = tr.record_offsets
    return [(int(off[a]), int(off[b]), float(rt[a])) for a, b in out]


def on_grid(x: np.ndarray, integer: bool) -> tuple[np.ndarray | None, float | None]:
    """Values in steps of the window's grid, and the step (1 for integer counts)."""
    if integer:
        return x.astype(np.int64), 1.0
    x = x[np.isfinite(x)].astype(np.float64)
    u = np.unique(x)
    if u.size < 3:
        return None, None
    d = np.diff(u)
    if not np.any(d > 1e-12):
        return None, None
    d0 = float(d[d > 1e-12].min())
    # float32 storage blurs the step (0.05 stored as 0.049995...): try it snapped to fewer significant digits
    for step in [float(f"{d0:.{k}g}") for k in (2, 3, 4)] + [d0]:
        r = x / step
        if np.mean(np.abs(r - np.round(r)) < 1e-3) >= ON_GRID:
            return np.round(r).astype(np.int64), step
    return None, d0


def holds(q: np.ndarray) -> dict | None:
    if q.size < 200:
        return None
    d = np.diff(q)
    rep = d == 0
    m = np.zeros(q.size, bool)
    s = np.abs(q[10:] - q[:-10])
    thr = max(np.percentile(s, 80), 10 * MIN_SLOPE)
    m[5:-5] = s >= thr
    mov = m[1:]                                            # moving at sample i (d[i-1] = q[i] - q[i-1])
    if mov.sum() < 50:
        return {"repeat_all": round(float(rep.mean()), 4), "repeat_moving": None}
    starts = np.flatnonzero(np.r_[True, d != 0])
    lens = np.diff(np.r_[starts, q.size])
    ms = m[starts]
    ch = (~rep).astype(float) - (~rep).mean()
    den = float(np.dot(ch, ch)) or 1.0
    ac = [round(float(np.dot(ch[:-k], ch[k:]) / den), 3) for k in range(1, 16)]
    return {"repeat_all": round(float(rep.mean()), 4), "repeat_moving": round(float(rep[mov].mean()), 4),
            "run_moving": round(float(lens[ms].mean()), 3) if ms.any() else None, "change_acf_1_15": ac}


def band_db(f: np.ndarray, P: np.ndarray, lo: float, hi: float, q: float) -> float | None:
    sel = (f >= lo) & (f <= hi)
    if sel.sum() < 3:
        return None
    m = float(np.median(P[sel]))
    return round(float(10 * np.log10(m / q)), 2) if m > 0 else None


def spectrum_features(q: np.ndarray, fs: float) -> dict:
    f, P = signal.welch(q.astype(float), fs, window="hann", nperseg=min(nseg(fs), q.size), noverlap=None,
                        detrend="constant", scaling="density")
    rq = 1.0 / (6.0 * fs)
    ny = fs / 2
    return {"floor_db": band_db(f, P, 0.8 * ny, 0.98 * ny, rq), "mid_db": band_db(f, P, 5, 15, rq) if ny > 15 else None,
            "lf_db": band_db(f, P, 0.2, 1.0, rq)}


def track_survey(tr) -> dict:
    fs = float(tr.srate)
    integer = bool(tr.is_integer_format)
    vals = np.asarray(tr.values)
    R = runs(tr)
    out = {"fs": fs, "integer": integer, "hours": round(float(sum(b - a for a, b, _ in R) / fs / 3600), 3),
           "holds": [], "epochs": [], "off_grid_windows": 0}
    if not R:
        return out
    a, b, t0 = max(R, key=lambda r: r[1] - r[0])
    n = int(HOLD_WIN_S * fs)
    for frac in (1 / 3, 2 / 3):
        s = a + int(frac * (b - a)) - n // 2
        if s < a or s + n > b:
            continue
        g, step = on_grid(vals[s: s + n], integer)
        if g is None:
            out["off_grid_windows"] += 1
            continue
        h = holds(g)
        if h:
            out["holds"].append({**h, "step": step})
    # one-minute epochs on an absolute 60 s grid inside gap-free runs
    ep = []
    for a, b, t in R:
        first = int(np.ceil(t / EPOCH_S))
        k = first
        while True:
            s = a + int(round((k * EPOCH_S - t) * fs))
            e = s + int(EPOCH_S * fs)
            if e > b:
                break
            ep.append((s, e))
            k += 1
    if len(ep) > MAX_EPOCHS:
        ep = [ep[i] for i in np.linspace(0, len(ep) - 1, MAX_EPOCHS).round().astype(int)]
    # rails: the 10-bit converter's limits, where a track's codes stay within 0-1023 (SNUADC)
    tenbit = integer and vals.size and int(vals.min()) >= 0 and int(vals.max()) <= 1023
    for s, e in ep:
        g, step = on_grid(vals[s:e], integer)
        if g is None:
            out["off_grid_windows"] += 1
            continue
        rec = spectrum_features(g, fs)
        if tenbit:
            w = vals[s:e]
            rec["rail_share"] = round(float(np.mean((w == 0) | (w == 1023))), 4)
        rec["step"] = step
        out["epochs"].append(rec)
    return out


def hum(tr) -> list[dict]:
    R = runs(tr)
    if not R:
        return []
    a, b, _ = max(R, key=lambda r: r[1] - r[0])
    fs = float(tr.srate)
    b = min(b, a + int(3600 * fs))
    if (b - a) < 600 * fs:
        return []
    x = np.nan_to_num(tr.physical(np.asarray(tr.values)[a:b]).astype(float))
    return mains_lines(x, fs)


def scan(a) -> None:
    from vitaldb_reader.vital_file import TRACK_WAVE, read_vital
    cases = json.loads(Path(a.cases).read_text())["cases"]
    cases = [c for i, c in enumerate(cases) if i % a.of == a.part]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t_start = time.time()
    with out.open("a") as fh:
        for c in cases:
            if c in done:
                continue
            if time.time() - t_start > a.time_budget:
                break
            t0 = time.time()
            row = {"case": c, "tracks": {}, "hum": {}}
            try:
                v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital")
                for name, tr in v.tracks.items():
                    if tr.track_type != TRACK_WAVE or float(tr.srate) < MIN_FS or not tr.record_times.size:
                        continue
                    row["tracks"][name] = track_survey(tr)
                    if name in HUM_TRACKS:
                        row["hum"][name] = hum(tr)
            except Exception as ex:  # noqa: BLE001
                row["error"] = f"{type(ex).__name__}: {ex}"[:300]
            row["secs"] = round(time.time() - t0, 1)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    print(json.dumps({"part": a.part, "seconds": round(time.time() - t_start, 1)}))


def q5(v) -> list | None:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return [round(float(x), 3) for x in np.percentile(v, [5, 25, 50, 75, 95])] + [int(v.size)] if v.size else None


def agg(a) -> None:
    rows = list({r["case"]: r for p in a.parts for r in map(json.loads, Path(p).read_text().splitlines())}.values())
    by = {}
    for r in rows:
        for name, t in r.get("tracks", {}).items():
            by.setdefault(name, []).append(t)
    S = {"recordings": len(rows), "errors": sum("error" in r for r in rows), "tracks": {}, "hum_120": {}}
    for name, T in sorted(by.items()):
        fl = [np.median([e["floor_db"] for e in t["epochs"] if e["floor_db"] is not None]) for t in T
              if any(e["floor_db"] is not None for e in t["epochs"])]
        within = [float(np.subtract(*np.percentile([e["floor_db"] for e in t["epochs"] if e["floor_db"] is not None],
                                                   [75, 25])))
                  for t in T if sum(e["floor_db"] is not None for e in t["epochs"]) >= 5]
        mid = [np.median([e["mid_db"] for e in t["epochs"] if e.get("mid_db") is not None]) for t in T
               if any(e.get("mid_db") is not None for e in t["epochs"])]
        H = [h for t in T for h in t["holds"]]
        acf = np.array([h["change_acf_1_15"] for h in H if h.get("change_acf_1_15")], float)
        S["tracks"][name] = {
            "recordings": len(T), "fs": T[0]["fs"], "integer_share": round(float(np.mean([t["integer"] for t in T])), 3),
            "floor_db_recording_medians_p5_25_50_75_95_n": q5(fl),
            "floor_db_within_recording_iqr_median": round(float(np.median(within)), 2) if within else None,
            "mid_db_recording_medians_p5_25_50_75_95_n": q5(mid),
            "rail_share_median": round(float(np.median([e["rail_share"] for t in T for e in t["epochs"]
                                                        if "rail_share" in e])), 4)
            if any("rail_share" in e for t in T for e in t["epochs"]) else None,
            "minutes_with_rails_share": round(float(np.mean([e["rail_share"] > 0 for t in T for e in t["epochs"]
                                                             if "rail_share" in e])), 4)
            if any("rail_share" in e for t in T for e in t["epochs"]) else None,
            "off_grid_windows": int(sum(t["off_grid_windows"] for t in T)),
            "holds": {"windows": len(H), "repeat_all_median": round(float(np.median([h["repeat_all"] for h in H])), 3)
                      if H else None,
                      "repeat_moving_median": round(float(np.median([h["repeat_moving"] for h in H
                                                                     if h.get("repeat_moving") is not None])), 3)
                      if any(h.get("repeat_moving") is not None for h in H) else None,
                      "run_moving_median": round(float(np.median([h["run_moving"] for h in H if h.get("run_moving")])), 3)
                      if any(h.get("run_moving") for h in H) else None,
                      "change_acf_1_15_median": [round(float(x), 3) for x in np.median(acf, 0)] if acf.size else None}}
    pooled = {"h2": [], "control": []}
    for name in HUM_TRACKS:
        diffs, ctrl = [], []
        for r in rows:
            L = {x["h"]: x["ppm"] for x in r.get("hum", {}).get(name, []) if x["snr"] >= HUM_SNR and x["res"] <= HUM_RES}
            if 2 in L:
                others = [L[h] for h in L if h != 2]
                if others:
                    diffs.append(L[2] - float(np.median(others)))
            for h in L:
                if h == 2:
                    continue
                others = [L[k] for k in L if k not in (2, h)]
                if others:
                    ctrl.append(L[h] - float(np.median(others)))
        pooled["h2"] += diffs
        pooled["control"] += ctrl
        S["hum_120"][name] = {"ppm_120_minus_other_mains_p5_25_50_75_95_n": q5(diffs),
                              "control_other_harmonic_minus_rest_p5_25_50_75_95_n": q5(ctrl)}
    S["hum_120"]["pooled"] = {k: q5(v) for k, v in pooled.items()}
    if len(pooled["h2"]) >= 5 and len(pooled["control"]) >= 5:
        from scipy.stats import mannwhitneyu
        S["hum_120"]["pooled"]["mannwhitney_p"] = round(float(mannwhitneyu(pooled["h2"], pooled["control"]).pvalue), 4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1)[:4000])


def rounded_noise_var(sd: float, offsets: int = 64) -> float:
    """Variance (steps^2) of round(u + n), n ~ N(0, sd^2), averaged over offsets u uniform in [0, 1)."""
    from scipy.stats import norm
    k = np.arange(-int(8 * sd) - 2, int(8 * sd) + 3)
    v = []
    for u in (np.arange(offsets) + 0.5) / offsets:
        p = norm.cdf((k + 0.5 - u) / sd) - norm.cdf((k - 0.5 - u) / sd)
        m = np.sum(p * k)
        v.append(np.sum(p * k * k) - m * m)
    return float(np.mean(v))


def synth(a) -> None:
    rng = np.random.default_rng(0)
    out = {"white": [], "lowpassed": [], "holds": {}}
    for fs in (500.0, 128.0):
        t = np.arange(int(600 * fs)) / fs
        slow = 50 * np.sin(2 * np.pi * 0.01 * t)               # moves through every position between two steps
        for sd in (0.3, 0.5, 1.0, 2.0, 4.0):
            q = np.round(slow + rng.normal(0, sd, t.size)).astype(np.int64)
            expect = 10 * np.log10(rounded_noise_var(sd) * 12)
            fl = spectrum_features(q, fs)["floor_db"]
            out["white"].append({"fs": fs, "sd_steps": sd, "expected_db": round(float(expect), 2), "measured_db": fl})
        b, aa = signal.butter(4, 0.5 * (fs / 2) / (fs / 2))
        x = signal.lfilter(b, aa, rng.normal(0, 20, int(600 * fs)))
        q = np.round(x).astype(np.int64)
        out["lowpassed"].append({"fs": fs, "cutoff_hz": fs / 4, "noise_sd_steps": 20,
                                 "measured_db": spectrum_features(q, fs)["floor_db"]})
    fs, n = 62.5, int(120 * 62.5)
    t = np.arange(n) / fs
    sig = lambda tt: 40 * np.sin(2 * np.pi * 0.25 * tt) + 15 * np.sin(2 * np.pi * 1.3 * tt)  # noqa: E731
    upd = np.floor(t * 24.974) / 24.974                       # held at a 24.974 Hz update grid
    out["holds"]["held_25hz"] = holds(np.round(sig(upd)).astype(np.int64))
    out["holds"]["sampled_62_5hz"] = holds(np.round(sig(t) + rng.normal(0, 0.3, n)).astype(np.int64))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def cases(a) -> None:
    ids = sorted(int(p.stem) for p in (Path(a.root) / "vital_files").glob("*.vital"))
    pick = sorted(int(x) for x in np.random.default_rng(a.seed).choice(ids, size=min(a.n, len(ids)), replace=False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"seed": a.seed, "n": len(pick), "of": len(ids), "cases": pick}) + "\n")
    print(len(pick), "of", len(ids))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--root", required=True)
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--seed", type=int, default=20261106)
    c.add_argument("--out", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    sc = sub.add_parser("scan")
    sc.add_argument("--root", required=True)
    sc.add_argument("--cases", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--part", type=int, default=0)
    sc.add_argument("--of", type=int, default=1)
    sc.add_argument("--time-budget", type=float, default=1e9)
    g = sub.add_parser("agg")
    g.add_argument("parts", nargs="+")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    {"cases": cases, "synth": synth, "scan": scan, "agg": agg}[a.cmd](a)


if __name__ == "__main__":
    main()
