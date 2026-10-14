# SPDX-License-Identifier: MIT
"""Reverse-engineer two hidden properties of .vital tracks, reporting aggregates only.

1. The TRUE sampling period of each waveform track. Every waveform record carries a float timestamp;
   within a gap-free run, a straight-line fit of record start time against the cumulative sample index
   gives the period the device actually ran at. The tool reports, per track name, the deviation of
   that period from the nominal one (ppm), how well a straight line explains the timestamps (residual,
   in microseconds), how many sudden steps the residual has (clock corrections), and the worst timing
   error each storage choice would leave behind:
     - nominal rate, one regular axis per run (what a naive import does)
     - fitted period rounded to whole nanoseconds (AtriumDB period_ns)
     - fitted frequency rounded to whole nanohertz (AtriumDB freq_nhz)
     - fitted period as a Fraction of microseconds with a bounded denominator, origin rounded to
       whole microseconds (TimeNet RegularAxis)
2. The INTEGER lattice under float-valued tracks. For float waveform and numeric tracks it looks for a
   step q and offset o such that every value equals o + k*q for integer k, then checks whether the
   float values are reproduced bit-exactly from (k, q, o). It reports q, o, the share reproduced and
   the bits that k needs. Steps and offsets are device calibration constants, not patient data.

    python vital_timing_quant.py scan ROOT --state tq_state.jsonl [--max-files N] [--seed 0] [--time-budget 120]
    python vital_timing_quant.py summary --state tq_state.jsonl --out timing_quant.json

The state file (per-file aggregates keyed by path) stays on this computer; the summary is file-anonymous.
Needs Python >= 3.9 and numpy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import struct
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

FMT = {1: ("<f4", 4), 2: ("<f8", 8), 3: ("<i1", 1), 4: ("<u1", 1), 5: ("<i2", 2), 6: ("<u2", 2), 7: ("<i4", 4), 8: ("<u4", 4)}
_u16 = struct.Struct("<H").unpack_from
_u32 = struct.Struct("<L").unpack_from
_f32 = struct.Struct("<f").unpack_from
_f64 = struct.Struct("<d").unpack_from
MIN_RUN_S = 60.0  # runs shorter than this give a poor period estimate and are skipped
MAX_VALUES = 400_000  # per track per file, for the lattice search


def _str(buf, pos):
    n = _u32(buf, pos)[0]
    return buf[pos + 4 : pos + 4 + n].decode("utf-8", "replace"), pos + 4 + n


def read_tracks(path: Path) -> dict:
    """Parse one .vital file into per-track record lists (times as float64, values as arrays)."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"VITA":
        raise ValueError("bad signature")
    hlen = _u16(data, 8)[0]
    pos, n = 10 + hlen, len(data)
    devs, trk = {}, {}
    while pos + 5 <= n:
        ptype, plen = data[pos], _u32(data, pos + 1)[0]
        body, pos = pos + 5, pos + 5 + plen
        if pos > n:
            break
        if ptype == 9:
            did = _u32(data, body)[0]
            dtype_, p = _str(data, body + 4)
            name, p = _str(data, p)
            devs[did] = name or dtype_
        elif ptype == 0:
            b = data[body:pos]
            tid, ttype, fmt = _u16(b, 0)[0], b[2], b[3]
            name, p = _str(b, 4)
            unit, p = _str(b, p)
            p += 12
            srate = _f32(b, p)[0]
            gain = _f64(b, p + 4)[0]
            offset = _f64(b, p + 12)[0]
            did = _u32(b, p + 21)[0] if p + 25 <= len(b) else 0
            dname = devs.get(did, "")
            trk[tid] = {"name": f"{dname}/{name}" if dname else name, "type": ttype, "fmt": fmt, "srate": srate,
                        "gain": gain, "offset": offset, "dt": [], "ns": [], "vals": []}
        elif ptype == 1 and plen >= 12:
            infolen = _u16(data, body)[0]
            dt = _f64(data, body + 2)[0]
            t = trk.get(_u16(data, body + 10)[0])
            if t is None or t["fmt"] not in FMT:
                continue
            p = body + 2 + infolen
            code, w = FMT[t["fmt"]]
            if t["type"] == 1:
                ns = _u32(data, p)[0]
                t["dt"].append(dt)
                t["ns"].append(ns)
                if t["fmt"] in (1, 2):
                    t["vals"].append(np.frombuffer(data, dtype=code, count=ns, offset=p + 4))
            elif t["type"] == 2:
                t["dt"].append(dt)
                if t["fmt"] in (1, 2):
                    t["vals"].append(np.frombuffer(data, dtype=code, count=1, offset=p))
    return {t["name"]: t for t in trk.values()}


def _bucket(x: float, edges, fmt="{}") -> str:
    for e in edges:
        if x <= e:
            return "<=" + fmt.format(e)
    return ">" + fmt.format(edges[-1])


PPM_EDGES = (-1000, -100, -50, -20, -10, -5, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 1000)
US_EDGES = (0.5, 1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000)


def timing(t: dict) -> dict:
    """Fit the true period within each gap-free run of one waveform track."""
    sr = float(t["srate"])
    if sr <= 0 or len(t["dt"]) < 2:
        return {}
    dt = np.asarray(t["dt"], dtype=np.float64)
    ns = np.asarray(t["ns"], dtype=np.int64)
    nominal = 1.0 / sr
    order_ok = bool(np.all(np.diff(dt) > 0))
    ends = dt + ns * nominal
    d = (dt[1:] - ends[:-1]) * sr  # boundary offsets in samples
    cuts = np.flatnonzero(np.abs(d) >= 0.5) + 1
    out = {"runs": 0, "runs_fitted": 0, "ppm": [], "fit_resid_us": [], "nominal_err_us": [], "ns_err_us": [],
           "nhz_err_us": [], "tn_err_us": {}, "steps": [], "inc_quanta": Counter(), "backward": not order_ok,
           "ppm_spread_in_case": None}
    tn_dens = (1, 10, 100, 1000, 10**6)
    for den in tn_dens:
        out["tn_err_us"][den] = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(dt)]):
        out["runs"] += 1
        if b - a < 3:
            continue
        rel = dt[a:b] - dt[a]  # exact: doubles of similar magnitude subtract exactly
        k = np.r_[0, np.cumsum(ns[a : b - 1])].astype(np.float64)
        if k[-1] / sr < MIN_RUN_S:
            continue
        out["runs_fitted"] += 1
        A = np.vstack([np.ones_like(k), k]).T
        (c0, period), *_ = np.linalg.lstsq(A, rel, rcond=None)
        resid = rel - (c0 + k * period)
        out["ppm"].append((nominal / period - 1.0) * 1e6)  # positive: device runs faster than nominal
        out["fit_resid_us"].append(float(np.abs(resid).max() * 1e6))
        # steps: a jump in residual between neighbouring records bigger than 5 us
        out["steps"].append(int((np.abs(np.diff(resid)) > 5e-6).sum()))
        # per-record increments minus nominal, in units of 2^-20 s (the float grid at ~4e9 s)
        inc = (np.diff(rel) - ns[a : b - 1] * nominal) * 2**20
        for v, c in zip(*np.unique(np.round(inc).astype(np.int64), return_counts=True)):
            if len(out["inc_quanta"]) < 40:
                out["inc_quanta"][int(v)] += int(c)
        out["nominal_err_us"].append(float(np.abs(rel - k * nominal).max() * 1e6))
        # AtriumDB: start time in whole ns, period in whole ns
        p_ns = round(period * 1e9)
        start_ns = round(c0 * 1e9)
        out["ns_err_us"].append(float(np.abs(rel * 1e9 - (start_ns + k * p_ns)).max() / 1e3))
        # AtriumDB: frequency in whole nHz
        f_nhz = round(1e9 / period)
        out["nhz_err_us"].append(float(np.abs(rel - (c0 + k * 1e9 / f_nhz)).max() * 1e6))
        # TimeNet: period as a Fraction of us with bounded denominator, origin in whole us
        for den in tn_dens:
            p_us = Fraction(period * 1e6).limit_denominator(den)
            off = round(c0 * 1e6)
            err = np.abs(rel * 1e6 - (off + k * float(p_us)))
            out["tn_err_us"][den].append(float(err.max()))
    if len(out["ppm"]) > 1:
        out["ppm_spread_in_case"] = float(max(out["ppm"]) - min(out["ppm"]))
    return out


def lattice(values: np.ndarray, is32: bool) -> dict:
    """Find step q and offset o with values == o + k*q (k integer); report reproducibility."""
    v = values[np.isfinite(values)].astype(np.float64)
    if v.size == 0:
        return {"status": "no-finite"}
    u = np.unique(v)
    if u.size == 1:
        return {"status": "constant"}
    ulp = np.spacing(np.float32(np.abs(u).max())) if is32 else np.spacing(np.abs(u).max())
    tol = 2.5 * float(ulp)
    diffs = np.diff(u)
    cands = []
    small = np.sort(diffs)[: min(50, diffs.size)]
    for d in np.unique(small):
        for div in (1, 2, 3, 4, 5, 8, 10):
            cands.append(d / div)
    cands += [10.0**-e for e in range(0, 7)]
    best = None
    for q in sorted(set(cands), reverse=True):
        if q <= tol:
            continue
        k = np.round((u - u[0]) / q)
        if np.unique(k).size != u.size:  # two distinct values on one lattice point: step too coarse
            continue
        # refine q and o by least squares over the integer indices, then test the residual
        A = np.vstack([np.ones_like(k), k]).T
        (o, qq), *_ = np.linalg.lstsq(A, u, rcond=None)
        if np.abs(u - (o + k * qq)).max() > max(tol, 1e-12):
            continue
        # express the origin as the lattice offset nearest zero, so k is the stored integer
        shift = np.round(o / qq)
        o -= shift * qq
        kk = k + shift
        # the fitted step is close to, not equal to, the constant the device used: try the nearby
        # "nice" spellings (float32 of it, short decimals) and keep whichever reproduces most values
        def score(q_, o_):
            f64 = np.mean(np.float32(o_ + kk * q_) == np.float32(u)) if is32 else np.mean(o_ + kk * q_ == u)
            f32 = np.mean((np.float32(kk) * np.float32(q_) + np.float32(o_)) == np.float32(u)) if is32 else None
            return f64, f32

        qs = {float(qq), float(np.float32(qq))} | {float(f"{qq:.{d}g}") for d in range(1, 9)}
        os_ = {float(o), 0.0, float(np.float32(o))} | {float(f"{o:.{d}g}") for d in range(1, 9)}
        (exact, exact32), qq, o = max(((score(q_, o_), q_, o_) for q_ in qs for o_ in os_),
                                      key=lambda x: (max(x[0][0], x[0][1] or 0), -len(repr(x[1]))))
        best = {"status": "lattice", "step": float(qq), "offset": float(o), "exact_f64_math": float(exact),
                "exact_f32_math": None if exact32 is None else float(exact32),
                "k_bits": int(math.ceil(math.log2(kk.max() - kk.min() + 1))) if kk.max() > kk.min() else 0,
                "k_range": [int(kk.min()), int(kk.max())],
                "distinct": int(u.size), "frac_values_integer": float(np.mean(u == np.round(u)))}
        break
    return best or {"status": "no-lattice", "distinct": int(u.size), "min_diff": float(diffs.min())}


def scan_file(path: Path) -> dict:
    tracks = read_tracks(path)
    out = {}
    for name, t in tracks.items():
        r = {"type": t["type"], "fmt": t["fmt"], "srate": t["srate"]}
        if t["type"] == 1:
            r["timing"] = timing(t)
        if t["fmt"] in (1, 2) and t["vals"]:
            v = np.concatenate(t["vals"])
            if v.size > MAX_VALUES:
                v = v[np.random.default_rng(0).choice(v.size, MAX_VALUES, replace=False)]
            r["lattice"] = lattice(v, t["fmt"] == 1)
        out[name] = r
    return out


def cmd_scan(a) -> None:
    fresh = time.time() - a.min_age
    files = sorted(p for p in Path(a.root).rglob("*.vital") if p.stat().st_mtime < fresh)
    random.Random(a.seed).shuffle(files)
    files = files[: a.max_files] if a.max_files else files
    state = Path(a.state)
    done = {json.loads(x)["path"] for x in state.read_text().splitlines() if x.strip()} if state.exists() else set()
    t0, k = time.time(), 0
    todo = [p for p in files if str(p) not in done]
    with state.open("a") as fh:
        for p in todo:
            if time.time() - t0 > a.time_budget:
                break
            try:
                rec = {"tracks": scan_file(p)}
            except Exception as e:  # noqa: BLE001
                rec = {"error": type(e).__name__}
            rec["path"] = str(p)
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            k += 1
    print(json.dumps({"done": len(done) + k, "remaining": len(todo) - k, "seconds": round(time.time() - t0, 1)}))


def cmd_summary(a) -> None:
    recs = [json.loads(x) for x in Path(a.state).read_text().splitlines() if x.strip()]
    T: dict = defaultdict(lambda: {"files": 0, "runs": 0, "runs_fitted": 0, "ppm": Counter(), "ppm_values": [],
                                   "fit_resid_us": Counter(), "steps": Counter(), "nominal_err_us": Counter(),
                                   "ns_err_us": Counter(), "nhz_err_us": Counter(), "tn_err_us": defaultdict(Counter),
                                   "ppm_spread_in_case": Counter(), "inc_quanta": Counter(), "backward_files": 0,
                                   "lattice": Counter(), "lattice_steps": Counter(), "lattice_exact": Counter(), "k_bits": Counter()})
    errors = Counter()
    for r in recs:
        if "error" in r:
            errors[r["error"]] += 1
            continue
        for name, t in r["tracks"].items():
            g = T[name]
            g["files"] += 1
            tm = t.get("timing") or {}
            if tm:
                g["runs"] += tm["runs"]
                g["runs_fitted"] += tm["runs_fitted"]
                g["backward_files"] += bool(tm["backward"])
                for x in tm["ppm"]:
                    g["ppm"][_bucket(x, PPM_EDGES)] += 1
                    g["ppm_values"].append(round(x, 3))
                for key in ("fit_resid_us", "nominal_err_us", "ns_err_us", "nhz_err_us"):
                    for x in tm[key]:
                        g[key][_bucket(x, US_EDGES)] += 1
                for den, xs in tm["tn_err_us"].items():
                    for x in xs:
                        g["tn_err_us"][f"den<={den}"][_bucket(x, US_EDGES)] += 1
                for x in tm["steps"]:
                    g["steps"][_bucket(x, (0, 1, 5, 20, 100, 1000))] += 1
                if tm["ppm_spread_in_case"] is not None:
                    g["ppm_spread_in_case"][_bucket(tm["ppm_spread_in_case"], (0.1, 0.5, 1, 2, 5, 10, 50, 100))] += 1
                g["inc_quanta"].update({str(k): v for k, v in tm["inc_quanta"].items()})
            lt = t.get("lattice")
            if lt:
                g["lattice"][lt["status"]] += 1
                if lt["status"] == "lattice":
                    g["lattice_steps"][f"step={lt['step']:.9g} offset={lt['offset']:.9g}"] += 1
                    g["lattice_exact"][f"f64math={lt['exact_f64_math']:.4f} f32math={lt['exact_f32_math']}"] += 1
                    g["k_bits"][lt["k_bits"]] += 1
                elif lt["status"] == "no-lattice":
                    g["lattice_steps"][f"none; min diff {lt['min_diff']:.3g}, distinct {lt['distinct']}"] += 1
    out = {"files": len(recs), "errors": dict(errors), "tracks": {}}
    for name, g in sorted(T.items(), key=lambda kv: -kv[1]["files"]):
        pv = sorted(g.pop("ppm_values"))
        g["ppm_quantiles"] = {q: pv[min(len(pv) - 1, int(q / 100 * len(pv)))] for q in (0, 10, 50, 90, 100)} if pv else None
        g["tn_err_us"] = {k: dict(v) for k, v in g["tn_err_us"].items()}
        g["inc_quanta"] = dict(g["inc_quanta"].most_common(12))
        out["tracks"][name] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in g.items()}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}: {len(recs)} files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("root")
    s.add_argument("--state", required=True)
    s.add_argument("--max-files", type=int, default=0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--time-budget", type=float, default=120)
    s.add_argument("--min-age", type=float, default=300)
    m = sub.add_parser("summary")
    m.add_argument("--state", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    {"scan": cmd_scan, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
