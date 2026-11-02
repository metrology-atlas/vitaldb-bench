# SPDX-License-Identifier: MIT
"""Monitor clocks in the SNUADC waveforms' spectral lines: the 240 Hz output line in ART and PLETH, the 30 Hz family
in PLETH, and the skew between ART and PLETH. Per-case numbers only.

    python tools/monitor_lines.py scan --root $R --scan $W/sawtooth_char/scan --out $W/monitor_lines/lines.jsonl \
        [--part 0 --of 3] [--time-budget 150]
    python tools/monitor_lines.py agg --lines $W/monitor_lines/lines.jsonl \
        --recordings $W/sawtooth_char/bench8/recordings.csv --out $W/monitor_lines

For each recording of a sawtooth_char scan (its stretch: the longest continuous ECG II and PLETH, 20-90 min), on the
raw converter counts of SNUADC ECG II, ART and PLETH:

* **Line signal.** The absolute sample-to-sample change, divided by its 25-sample running mean (the pattern of step
  sizes, which an output stage running on its own clock imprints; tools/output_cycle_fold.py).
* **Demodulation** at a nominal frequency f (240 Hz in 10 s windows; 30, 60, 90, 120, 150 Hz in 60 s windows, as
  their phase turns slowly) on the global sample index: each window's complex amplitude. The line's strength is the median window amplitude over that at a control
  frequency nearby (f x 0.98875: 237.3 Hz for 240 Hz) where no line is expected. Its frequency against the SNUADC
  sample clock is the slope of the unwrapped window phase (ppm), when the line is present (strength > 3).
* **Skew** of ART against PLETH at 240 Hz: the circular mean of the phase difference of their windows, as time
  (positive: ART's line comes earlier, so ART is read first).
* **Frame clock** (from the sawtooth_char scan: raw PLETH maxima locking to the 33.3 ms frame) for comparison with
  the 30 Hz family: a frame period longer by e ppm gives lines lower in frequency by e ppm.

``agg`` summarises over recordings: where each line is present, its rate, the 30 Hz family's rate against the frame
clock (30, 60 and 90 Hz; 120 Hz is compared with both the 240 Hz line and the frame clock, as it could belong to
either), the skew, and, from sawtooth_char's recordings.csv, which recordings have a PPG sawtooth.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pat import sample_times  # noqa: E402
from vitaldb_reader.vital_file import read_vital  # noqa: E402

FS = 500.0
WIN_S = 10.0               # windows for the 240 Hz line (its phase turns fastest)
WIN_LOW_S = 60.0           # windows for the 30 Hz family (slow phase; longer windows, more signal)
CH = {"ECG": "SNUADC/ECG_II", "ART": "SNUADC/ART", "PLETH": "SNUADC/PLETH"}
LINES = {"240": 240.0, "30": 30.0, "60": 60.0, "90": 90.0, "120": 120.0, "150": 150.0}
FAMILY = ("30", "60", "90")   # the frame family (120 Hz is checked separately: it may be half the output line)
CTRL = 0.98875              # control frequency = f x CTRL (240 -> 237.3 Hz)
PRESENT = 3.0               # strength above which a line counts as present


def line_signal(x: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(x.astype(np.float64)))
    m = np.convolve(d, np.ones(25) / 25, "same") + 1e-9
    r = d / m
    return r - r.mean()


def win_s(f: float) -> float:
    return WIN_S if f > 200 else WIN_LOW_S


def demod(u: np.ndarray, n0: int, f: float, win: float | None = None) -> np.ndarray:
    """Complex amplitude of u at f in consecutive windows (10 s at 240 Hz, 60 s below; global sample index n0 + i)."""
    W = int((win or win_s(f)) * FS)
    k = u.size // W
    if k < 3:
        return np.empty(0, complex)
    n = n0 + np.arange(k * W)
    z = (u[: k * W] * np.exp(-2j * np.pi * f * n / FS)).reshape(k, W).mean(1)
    return z


def line(u: np.ndarray, n0: int, f: float) -> dict:
    z = demod(u, n0, f)
    zc = demod(u, n0, f * CTRL, win_s(f))
    if not z.size:
        return {}
    strength = float(np.median(np.abs(z)) / max(np.median(np.abs(zc)), 1e-12))
    out = {"strength": round(strength, 2)}
    if strength > PRESENT:
        ph = np.unwrap(np.angle(z))
        tt = np.arange(z.size) * win_s(f)
        slope = np.polyfit(tt, ph, 1)[0]              # rad/s: the line is at f + slope / 2 pi
        out["ppm"] = round(float(slope / (2 * np.pi) / f * 1e6), 2)
        out["resid_rad"] = round(float(np.std(ph - np.polyval(np.polyfit(tt, ph, 1), tt))), 3)
    return out


def scan(a) -> None:
    files = [f for i, f in enumerate(sorted(Path(a.scan).glob("*.npz"))) if i % a.of == a.part]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t0 = time.time()
    with out.open("a") as fh:
        for f in files:
            c = int(f.stem)
            if c in done:
                continue
            if time.time() - t0 > a.time_budget:
                break
            res = {"case": c}
            try:
                z = np.load(f)
                fc = json.loads(str(z["frame_clock"]))
                res["frame_ppm"] = fc.get("ppm")
                res["frame_sharpness"] = fc.get("sharpness")
                t_a, t_b = float(z["ppg_t0"]), float(np.nanmax(z["up"]))
                v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names=set(CH.values()))
                sig = {}
                for name, tn in CH.items():
                    tr = v.tracks.get(tn)
                    if tr is None or not tr.record_times.size or float(tr.srate) != FS:
                        continue
                    ts = sample_times(tr)
                    i0, i1 = int(np.searchsorted(ts, t_a - 1e-6)), int(np.searchsorted(ts, t_b))
                    if i1 - i0 < 600 * FS or np.max(np.abs(np.diff(ts[i0:i1]) - 1 / FS)) > 0.01 / FS:
                        continue                                    # this channel is not continuous over the stretch
                    x = np.asarray(tr.values[i0:i1])
                    if np.ptp(x) == 0:
                        continue
                    sig[name] = (line_signal(x), i0)
                res["channels"] = sorted(sig)
                res["minutes"] = round((t_b - t_a) / 60, 1)
                for name, (u, i0) in sig.items():
                    res[name] = {k: line(u, i0, fq) for k, fq in LINES.items()}
                if "ART" in sig and "PLETH" in sig:
                    za = demod(sig["ART"][0], sig["ART"][1], 240.0)
                    zp = demod(sig["PLETH"][0], sig["PLETH"][1], 240.0)
                    m = min(za.size, zp.size)
                    if m >= 3 and res["ART"]["240"].get("strength", 0) > PRESENT and res["PLETH"]["240"].get("strength", 0) > PRESENT:
                        d = np.angle(np.mean(za[:m] * np.conj(zp[:m]) / np.abs(za[:m] * zp[:m])))
                        # phase of a signal leading by tau is advanced: angle(za conj zp) = -2 pi f tau with this
                        # demodulation (exp(-i...)), so tau_ART_first = -d / (2 pi f)
                        res["skew_ms"] = round(float(-d / (2 * np.pi * 240.0) * 1000), 3)
                        res["skew_coherence"] = round(float(np.abs(np.mean(za[:m] * np.conj(zp[:m]) / np.abs(za[:m] * zp[:m])))), 3)
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(res) + "\n")
    print(json.dumps({"part": a.part, "done": sum(1 for _ in out.open()), "seconds": round(time.time() - t0, 1)}))


def q(v, ps=(5, 25, 50, 75, 95)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return ({f"p{p}": round(float(np.percentile(v, p)), 3) for p in ps} | {"n": int(v.size)}) if v.size else None


def agg(a) -> None:
    rows = [json.loads(x) for x in Path(a.lines).read_text().splitlines() if x.strip()]
    ok = [r for r in rows if "error" not in r]
    recs = {int(d["case"]): d for d in csv.DictReader(open(a.recordings))} if a.recordings else {}
    S = {"recordings": len(rows), "errors": sum(1 for r in rows if "error" in r),
         "with_channel": {ch: sum(1 for r in ok if ch in r.get("channels", [])) for ch in CH}}
    pres = {}
    for ch in CH:
        for k in LINES:
            v = [r[ch][k] for r in ok if ch in r and r[ch].get(k)]
            pres[f"{ch} {k} Hz"] = {"of": len(v), "present": sum(1 for x in v if x.get("strength", 0) > PRESENT),
                                    "strength": q([x.get("strength") for x in v]),
                                    "ppm": q([x.get("ppm") for x in v])}
    S["lines"] = pres
    # 240 Hz: ART and PLETH agree?
    both = [(r["ART"]["240"]["ppm"], r["PLETH"]["240"]["ppm"]) for r in ok if "ART" in r and "PLETH" in r
            and r["ART"]["240"].get("ppm") is not None and r["PLETH"]["240"].get("ppm") is not None]
    S["240_art_minus_pleth_ppm"] = q([x - y for x, y in both])
    S["skew_ms"] = q([r.get("skew_ms") for r in ok])
    S["skew_coherence"] = q([r.get("skew_coherence") for r in ok])
    S["skew_positive"] = sum(1 for r in ok if (r.get("skew_ms") or 0) > 0)
    # the 30 Hz family against the frame clock: line ppm should be minus the frame period's ppm
    fam = []
    for r in ok:
        if (r.get("frame_sharpness") or 0) < 10 or "PLETH" not in r:
            continue
        v = [r["PLETH"][k].get("ppm") for k in FAMILY if r["PLETH"].get(k, {}).get("ppm") is not None]
        if v:
            fam.append((float(np.median(v)), r["frame_ppm"], r["case"]))
    S["family_vs_frame"] = {"n": len(fam), "family_plus_frame_ppm": q([x + y for x, y, _ in fam]),
                            "within_3ppm": sum(1 for x, y, _ in fam if abs(x + y) <= 3),
                            "corr": round(float(np.corrcoef([x for x, _, _ in fam], [y for _, y, _ in fam])[0, 1]), 3) if len(fam) > 3 else None}
    # 120 Hz: half the output line's frequency, or a member of the frame family? Its rate against each
    r120 = []
    for r in ok:
        for ch in CH:
            if ch in r and r[ch].get("120", {}).get("ppm") is not None:
                l240 = r[ch].get("240", {}).get("ppm")
                if l240 is None and "PLETH" in r:
                    l240 = r["PLETH"].get("240", {}).get("ppm")
                fr = r.get("frame_ppm") if (r.get("frame_sharpness") or 0) >= 10 else None
                r120.append((ch, r[ch]["120"]["ppm"], l240, fr))
    S["120_hz"] = {"n": len(r120), "by_channel": {ch: sum(1 for x in r120 if x[0] == ch) for ch in CH},
                   "minus_240_ppm": q([x[1] - x[2] for x in r120 if x[2] is not None]),
                   "within_3ppm_of_240": sum(1 for x in r120 if x[2] is not None and abs(x[1] - x[2]) <= 3),
                   "with_240": sum(1 for x in r120 if x[2] is not None),
                   "plus_frame_ppm": q([x[1] + x[3] for x in r120 if x[3] is not None]),
                   "within_3ppm_of_frame": sum(1 for x in r120 if x[3] is not None and abs(x[1] + x[3]) <= 3),
                   "with_frame": sum(1 for x in r120 if x[3] is not None)}
    # 240 Hz against the frame clock (a different clock?)
    s240 = [(r["PLETH"]["240"]["ppm"], r["frame_ppm"]) for r in ok if "PLETH" in r and r["PLETH"]["240"].get("ppm") is not None
            and (r.get("frame_sharpness") or 0) >= 10]
    S["240_vs_frame"] = {"n": len(s240), "line240_plus_frame_ppm": q([x + y for x, y in s240]),
                         "corr": round(float(np.corrcoef(*zip(*s240))[0, 1]), 3) if len(s240) > 3 else None}
    # sawtooth in recordings with both lines (the open conflict)
    if recs:
        w = [r for r in ok if "skew_ms" in r and int(r["case"]) in recs]
        S["sawtooth_where_both_240"] = {"n": len(w),
                                        "up": sum(1 for r in w if recs[r["case"]]["direction"] == "up"),
                                        "down": sum(1 for r in w if recs[r["case"]]["direction"] == "down"),
                                        "slips_any": sum(1 for r in w if int(recs[r["case"]]["up"] or 0) + int(recs[r["case"]]["down"] or 0) > 0)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: S[k] for k in ("recordings", "errors", "with_channel")}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--root", required=True)
    p.add_argument("--scan", required=True, help="a sawtooth_char scan folder (stretches and frame clocks)")
    p.add_argument("--out", required=True)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--time-budget", type=float, default=150)
    p = sub.add_parser("agg")
    p.add_argument("--lines", required=True)
    p.add_argument("--recordings", help="sawtooth_char recordings.csv")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"scan": scan, "agg": agg}[a.cmd](a)


if __name__ == "__main__":
    main()
