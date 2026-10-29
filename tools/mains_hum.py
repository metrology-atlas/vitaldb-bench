# SPDX-License-Identifier: MIT
"""Mains hum in the waveforms, on each device's own sample count. This machine only for the scan; aggregates leave.

    python tools/mains_hum.py scan --root R --out part.jsonl [--cases 1,2,3 | --part P --nparts N] [--block-min 0]
    python tools/mains_hum.py agg PARTS... --rates RATES... --ports ports.jsonl --table per_recording.csv \
        --summary summary.json

Scan. Per track (SNUADC ECG II, ECG V5, ART, CVP; BIS EEG1, EEG2), on the longest run of contiguous records (gap
< 0.5 sample): the mains lines at h x 60 Hz (h = 1..8), at their aliased positions for the track's rate. Strength =
peak of a 0.01 Hz Welch spectrum over the median of +-1 Hz; for strong lines (SNR > 20) the frequency by complex
demodulation (phase slope over 1 s blocks), as ppm of the harmonic on the device's count. A line at the true grid
frequency appears at -(device fast) ppm, so line = f_grid - f_device. With --block-min 0 (default) one window, the
first 60 min of the run; with --block-min B, every B-minute block of the whole run (for stability over a recording).
Resumable.

Agg. Per recording: the SNUADC line (median of h = 2 or 6 lines with SNR > 50 over the SNUADC tracks), the BIS line
(h = 1), and the SNUADC and BIS rates against the recorder PC from tools/clock_correction_periods.py
(physiosync.stamp_clock; r = f_PC - f_device, negative = device fast) with their number of corrections. g = r - line
= f_PC - f_grid, the PC's rate against the grid, where the rate is measured (one correction or more). The table
stays on this machine; the summary (coverage, g by number of corrections, by SNUADC port, by setup) leaves.

Whole release, 27-29 Oct 2026: SNUADC hum in 2,455 of 6,348 recordings, mostly 120 and 360 Hz (seen at 140 Hz).
g: +1.6 ppm (median) in recordings with one correction, -13.8 ppm with two or more: which recordings carry
corrections is selected by their rate, so neither is a population value. The BIS 60 Hz line is not mains.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import signal

SNU = ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ART", "SNUADC/CVP"]
BIS = ["BIS/EEG1_WAV", "BIS/EEG2_WAV"]


def alias(f, fs):
    g = f % fs
    return (g, +1) if g <= fs / 2 else (fs - g, -1)


def demod_ppm(x, fs, fa, sgn, h):
    """Frequency of the line at fa (Hz on the device's count) as ppm of the harmonic h x 60 Hz, and phase residual."""
    n = np.arange(x.size)
    z = x * np.exp(-2j * np.pi * fa * n / fs)
    blk = int(fs)
    m = z[: z.size // blk * blk].reshape(-1, blk).mean(1)
    if m.size < 60:
        return None, None
    ph = np.unwrap(np.angle(m)); t = np.arange(m.size)
    p = np.polyfit(t, ph, 1)
    return sgn * p[0] / (2 * np.pi) / (h * 60) * 1e6, float(np.std(ph - np.polyval(p, t)))


def longest_run(tr):
    rt, rs, fs = tr.record_times, tr.record_sizes, float(tr.srate)
    if rt.size < 2:
        return None
    ok = np.abs(np.diff(rt) - rs[:-1] / fs) < 0.5 / fs
    best, cur, bs, cs = 0, 0, 0, 0
    for i, g in enumerate(ok):
        cur = cur + 1 if g else 0
        if cur == 1:
            cs = i
        if cur > best:
            best, bs = cur, cs
    a, b = bs, bs + best + 1
    starts = np.r_[0, np.cumsum(rs)[:-1]]
    x = np.nan_to_num(tr.physical()[starts[a]: starts[b - 1] + rs[b - 1]].astype(float))
    return x, fs, float(rt[a])


def lines(x, fs, snr_min=20.0):
    x = signal.detrend(x)
    f, P = signal.welch(x, fs, nperseg=int(100 * fs), noverlap=0, detrend="linear")
    out = []
    for h in range(1, 9):
        fa, sgn = alias(60.0 * h, fs)
        if fa < 1:
            continue
        near = np.abs(f - fa) < 1.0; pk = np.abs(f - fa) < 0.05
        if not pk.any():
            continue
        snr = float(P[pk].max() / np.median(P[near]))
        if snr > snr_min:
            ppm, res = demod_ppm(x, fs, fa, sgn, h)
            if ppm is not None:
                out.append({"h": h, "snr": round(snr, 1), "ppm": round(ppm, 2), "res": round(res, 2)})
    return out


def one(root, case, block_min):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{case:04d}.vital", track_names=set(SNU + BIS))
    out = {"case": case, "tracks": {}}
    for k in SNU + BIS:
        tr = v.tracks.get(k)
        if tr is None or not tr.record_times.size or not tr.srate:
            continue
        r = longest_run(tr)
        if r is None:
            continue
        x, fs, t0 = r
        if x.size < fs * 600:
            continue
        if block_min <= 0:
            x = x[: int(3600 * fs)]
            out["tracks"][k] = {"min": round(x.size / fs / 60, 1), "t0": t0, "lines": lines(x, fs)}
        else:
            n = int(block_min * 60 * fs)
            out["tracks"][k] = {"t0": t0, "block_min": block_min,
                                "blocks": [lines(x[i * n:(i + 1) * n], fs) for i in range(x.size // n)]}
    return out


def scan(a):
    root = Path(a.root)
    if a.cases:
        cases = [int(c) for c in a.cases.split(",")]
    else:
        files = sorted(int(p.stem) for p in (root / "vital_files").glob("*.vital"))
        cases = [c for i, c in enumerate(files) if i % a.nparts == a.part]
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    for c in cases:
        if c in done:
            continue
        t = time.time()
        try:
            r = one(root, c, a.block_min)
        except Exception as e:  # keep going; the error is recorded
            r = {"case": c, "error": repr(e)[:200]}
        r["s"] = round(time.time() - t, 1)
        with out.open("a") as f:
            f.write(json.dumps(r) + "\n")


def q(v):
    v = np.asarray(v, float)
    return {"n": int(v.size), "p10_25_50_75_90": np.round(np.percentile(v, [10, 25, 50, 75, 90]), 1).tolist()} if v.size else None


def agg(a):
    rates = defaultdict(dict)
    for f in a.rates:
        for l in open(f):
            d = json.loads(l)
            for k in ("SNUADC/ECG_II", "BIS/EEG1_WAV"):
                if isinstance(d.get(k), dict) and d[k].get("rate_ppm") is not None:
                    rates[d["case"]][k] = (d[k]["rate_ppm"], d[k].get("events", 0), d[k].get("hours"))
    ports = {}
    for l in open(a.ports):
        d = json.loads(l); ports[d["case"]] = d["ports"]
    rows, cov = [], Counter()
    for f in a.parts:
        for l in open(f):
            d = json.loads(l); c = d["case"]; T = d.get("tracks", {})
            cov["recordings"] += 1
            if "error" in d:
                cov["errors"] += 1
                continue
            s = [e["ppm"] for k in SNU for e in T.get(k, {}).get("lines", []) if e["h"] in (2, 6) and e["snr"] > 50]
            b = [e["ppm"] for k in BIS for e in T.get(k, {}).get("lines", []) if e["h"] == 1 and e["snr"] > 50]
            cov["with SNUADC"] += any(k in T for k in SNU); cov["SNUADC hum"] += bool(s)
            cov["with BIS"] += any(k in T for k in BIS); cov["BIS 60 Hz line"] += bool(b)
            rs = rates.get(c, {}).get("SNUADC/ECG_II"); rb = rates.get(c, {}).get("BIS/EEG1_WAV"); p = ports.get(c, {})
            rows.append(dict(case=c, snu_line=statistics.median(s) if s else None,
                             snu_rate=rs[0] if rs else None, snu_events=rs[1] if rs else None, hours=rs[2] if rs else None,
                             bis_line=statistics.median(b) if b else None, bis_rate=rb[0] if rb else None,
                             snu_port=p.get("SNUADC"), bis_port=p.get("BIS"),
                             key=" ".join(f"{k}:{v}" for k, v in sorted(p.items()))))
    import csv
    with open(a.table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    g = [(r["snu_rate"] - r["snu_line"], r) for r in rows if r["snu_line"] is not None and r["snu_events"]]
    by_port, by_key = defaultdict(list), defaultdict(list)
    for v, r in g:
        by_port[r["snu_port"]].append(v); by_key[r["key"]].append(v)
    summ = {"coverage": dict(cov),
            "snuadc_line_ppm": q([r["snu_line"] for r in rows if r["snu_line"] is not None]),
            "pc_vs_grid_ppm": {"one correction": q([v for v, r in g if r["snu_events"] == 1]),
                               "two or more": q([v for v, r in g if r["snu_events"] >= 2]),
                               "all with corrections": q([v for v, r in g])},
            "pc_vs_grid_by_snuadc_port": {k: q(v) for k, v in sorted(by_port.items(), key=lambda kv: -len(kv[1])) if len(v) >= 20},
            "pc_vs_grid_by_setup_medians": sorted(round(float(np.median(v)), 1) for v in by_key.values() if len(v) >= 30),
            "bis_line_ppm": q([r["bis_line"] for r in rows if r["bis_line"] is not None])}
    json.dump(summ, open(a.summary, "w"), indent=1)
    print(json.dumps(summ["coverage"]), json.dumps(summ["pc_vs_grid_ppm"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("--root", required=True); s.add_argument("--out", required=True)
    s.add_argument("--cases"); s.add_argument("--part", type=int, default=0); s.add_argument("--nparts", type=int, default=1)
    s.add_argument("--block-min", type=float, default=0.0)
    g = sp.add_parser("agg"); g.add_argument("parts", nargs="+"); g.add_argument("--rates", nargs="+", required=True)
    g.add_argument("--ports", required=True); g.add_argument("--table", required=True); g.add_argument("--summary", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
