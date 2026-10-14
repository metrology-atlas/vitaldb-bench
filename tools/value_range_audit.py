# SPDX-License-Identifier: MIT
"""Full audit of the value range each integer waveform uses, where it clips, and per-file configuration
markers (to test whether resolution changes mark epochs or machines). Per case, this machine only.

    python tools/value_range_audit.py scan --root R --cases 1,2 --out part.jsonl
    python tools/value_range_audit.py agg part*.jsonl --units units.csv --out summary.json

scan, per integer waveform track (counts as stored; physical = count * gain + offset):
  gain, offset, number of samples, min / max count, p0.1 / p99.9, distinct counts used, effective bits
  (log2 of the p0.1-p99.9 span + 1), and clipping: the share of samples at the minimum and at the maximum
  count, and the longest run of identical samples there (a run of >= 25 samples = 50 ms at 500 Hz is a
  plateau, not a signal crossing); plus every count where runs of >= 250 samples (0.5 s) occur, which finds
  clip levels inside the converter range (an upstream output stage that saturates first).
per case markers: recorder version, numeric stamp grid (share of Solar8000/HR stamps on whole ms), device
  type names and ports, Primus waveform gains, BIS EEG value step, Solar GAS2 value grid, pump rate grids,
  SNUADC rate.
agg: per track, the distribution of the above over cases; per marker pair, contingency tables (Primus gain
  configuration against stamp grid, recorder version, device type names, ports and the clock-unit identity
  from units.csv); within-file changes (tracks that appear twice, '#2').
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ORIGIN = 4102444800.0


def longest_run_at(x: np.ndarray, level: int) -> int:
    m = np.r_[False, x == level, False]
    d = np.diff(m.astype(np.int8))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((en - st).max()) if st.size else 0


def plateau_levels(x: np.ndarray, min_run: int) -> dict[int, int]:
    """Counts at which runs of >= min_run identical samples occur -> number of such runs."""
    if x.size < min_run:
        return {}
    ch = np.flatnonzero(np.diff(x) != 0)
    st = np.r_[0, ch + 1]
    ln = np.diff(np.r_[st, x.size])
    k = ln >= min_run
    return dict(Counter(x[st[k]].tolist()))


def grid_of(x):
    u = np.unique(np.round(np.asarray(x, float)[np.isfinite(x)], 6))
    if u.size < 3:
        return None
    best = None
    for g in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        if np.mean(np.abs(u / g - np.round(u / g)) < 1e-3) > 0.999:
            best = g
    return best


def scan_case(root, c):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital")
    res = {"case": c, "version": int(v.version), "devices": {k: [d.type_name, d.port] for k, d in v.devices.items()},
           "tracks": {}, "markers": {}}
    for n, tr in v.tracks.items():
        if tr.track_type == 1 and tr.is_integer_format and len(tr.values):
            x = np.asarray(tr.values).astype(np.int64)
            fs = float(tr.srate)
            bc = np.bincount(x - x.min())
            lo, hi = int(x.min()), int(x.max())
            cs = np.cumsum(bc) / x.size
            p001 = lo + int(np.searchsorted(cs, 0.001))
            p999 = lo + int(np.searchsorted(cs, 0.999))
            res["tracks"][n] = {
                "gain": float(tr.gain), "offset": float(tr.offset), "srate": fs, "n": int(x.size),
                "min": lo, "max": hi, "p001": p001, "p999": p999, "distinct": int((bc > 0).sum()),
                "eff_bits": round(math.log2(p999 - p001 + 1), 2),
                "at_min": round(float(bc[0] / x.size), 6), "at_max": round(float(bc[-1] / x.size), 6),
                "run_min": longest_run_at(x, lo), "run_max": longest_run_at(x, hi),
                "plateaus": {str(k): v_ for k, v_ in sorted(plateau_levels(x, int(0.5 * fs) or 1).items())[:20]}}
        elif tr.track_type == 1 and len(tr.values):
            x = np.asarray(tr.values, float)
            u = np.unique(np.round(x[np.isfinite(x)][:300000], 6))
            d = np.diff(u)
            d = d[d > 1e-7]
            res["markers"][f"{n}:min_step"] = round(float(d.min()), 6) if d.size else None
    mk = res["markers"]
    hr = v.tracks.get("Solar8000/HR")
    if hr is not None and hr.record_times.size:
        rel = np.asarray(hr.record_times) - ORIGIN
        mk["stamp_on_1ms"] = round(float(np.mean(np.abs(rel * 1000 - np.round(rel * 1000)) < 0.02)), 3)
    for n in ("Solar8000/GAS2_EXPIRED", "Orchestra/PPF20_CT", "Orchestra/RFTN20_CT", "Orchestra/PHEN_RATE", "Orchestra/NEPI_RATE"):
        tr = v.tracks.get(n)
        if tr is not None and len(tr.values):
            mk[f"{n}:grid"] = grid_of(tr.values)
    dup = [n for n in v.tracks if "#" in n]
    if dup:
        mk["duplicate_tracks"] = dup
    return res


def scan(root, cases, out):
    done = set()
    if Path(out).exists():
        done = {json.loads(l)["case"] for l in open(out)}
    for c in cases:
        if c in done or c > 6388:
            continue
        try:
            r = scan_case(root, c)
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:200]}
        with open(out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


def q(x, p=(1, 10, 50, 90, 99)):
    x = np.asarray([v for v in x if v is not None], float)
    return [round(float(v), 4) for v in np.percentile(x, p)] if x.size else None


def agg(paths, units, out):
    rows, seen = [], set()
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            if r["case"] not in seen and "error" not in r:
                seen.add(r["case"])
                rows.append(r)
    U = {}
    if units and Path(units).exists():
        U = {int(r["case"]): r for r in csv.DictReader(open(units))}
    S = {"cases": len(rows), "tracks": {}, "contingency": {}}
    by = defaultdict(list)
    for r in rows:
        for n, t in r["tracks"].items():
            by[n.split("#")[0]].append(t)
    for n, T in sorted(by.items()):
        cfg = Counter((round(t["gain"], 6), round(t["offset"], 4)) for t in T)
        lim_hi = Counter(t["max"] for t in T).most_common(5)
        lim_lo = Counter(t["min"] for t in T).most_common(5)
        pl = Counter()
        for t in T:
            for k, v_ in t["plateaus"].items():
                pl[int(k)] += 1
        clip_hi = [t for t in T if t["run_max"] >= 25]
        clip_lo = [t for t in T if t["run_min"] >= 25]
        S["tracks"][n] = {
            "cases": len(T), "configs": [[list(k), v_] for k, v_ in cfg.most_common(5)],
            "max_count_most_common": lim_hi, "min_count_most_common": lim_lo,
            "cases_clipping_high": len(clip_hi), "cases_clipping_low": len(clip_lo),
            "share_samples_at_max_p50_p99": q([t["at_max"] for t in clip_hi], (50, 99)) if clip_hi else None,
            "share_samples_at_min_p50_p99": q([t["at_min"] for t in clip_lo], (50, 99)) if clip_lo else None,
            "plateau_levels_cases": pl.most_common(12),
            "eff_bits": q([t["eff_bits"] for t in T]), "distinct_counts": q([t["distinct"] for t in T]),
            "p001_p50": float(np.median([t["p001"] for t in T])), "p999_p50": float(np.median([t["p999"] for t in T])),
            "physical_range_used_p50": [round(float(np.median([t["p001"] * t["gain"] + t["offset"] for t in T])), 3),
                                        round(float(np.median([t["p999"] * t["gain"] + t["offset"] for t in T])), 3)]}

    # markers and contingency
    def primus_cfg(r):
        t = r["tracks"].get("Primus/CO2")
        return None if t is None else f"CO2 gain {round(t['gain'], 4)}"

    def bis_cfg(r):
        s = r["markers"].get("BIS/EEG1_WAV:min_step")
        return None if s is None else ("0.04993" if abs(s - 0.049927) < 2e-5 else ("0.04999" if abs(s - 0.049988) < 2e-5 else f"{s}"))

    feats = {
        "primus": primus_cfg,
        "bis_eeg_step": bis_cfg,
        "stamp_grid": lambda r: None if "stamp_on_1ms" not in r["markers"] else ("1 ms" if r["markers"]["stamp_on_1ms"] > 0.9 else "0.1 ms"),
        "version": lambda r: r["version"],
        "solar_type": lambda r: (r["devices"].get("Solar8000") or [None])[0],
        "gas2_grid": lambda r: r["markers"].get("Solar8000/GAS2_EXPIRED:grid"),
        "snuadc_port": lambda r: (r["devices"].get("SNUADC") or [None, None])[1],
        "primus_port": lambda r: (r["devices"].get("Primus") or [None, None])[1],
        "solar_port": lambda r: (r["devices"].get("Solar8000") or [None, None])[1],
        "bis_unit": lambda r: (U.get(r["case"]) or {}).get("bis_unit") or None,
        "snuadc_unit": lambda r: (U.get(r["case"]) or {}).get("snuadc_unit") or None,
        "phen_rate_grid": lambda r: r["markers"].get("Orchestra/PHEN_RATE:grid"),
        "ppf_ct_grid": lambda r: r["markers"].get("Orchestra/PPF20_CT:grid"),
    }
    F = {k: [f(r) for r in rows] for k, f in feats.items()}
    for a in ("primus", "bis_eeg_step", "gas2_grid", "stamp_grid", "phen_rate_grid"):
        for b in feats:
            if a == b:
                continue
            ct = Counter((str(x), str(y)) for x, y in zip(F[a], F[b]) if x is not None and y is not None)
            if not ct:
                continue
            xs = sorted({k[0] for k in ct})
            ys = Counter()
            for (x, y), v_ in ct.items():
                ys[y] += v_
            top = [y for y, _ in ys.most_common(12)]
            S["contingency"][f"{a} x {b}"] = {"rows": xs, "cols": top,
                                              "table": [[ct.get((x, y), 0) for y in top] for x in xs],
                                              "n": int(sum(ct.values())), "cramers_v": cramers_v(ct)}
    S["marker_counts"] = {k: Counter(str(x) for x in v_).most_common(8) for k, v_ in F.items()}
    S["within_file_changes"] = Counter(t for r in rows for t in r["markers"].get("duplicate_tracks", [])).most_common(20)
    json.dump(S, open(out, "w"), indent=1)


def cramers_v(ct):
    xs = sorted({k[0] for k in ct})
    ys = sorted({k[1] for k in ct})
    if len(xs) < 2 or len(ys) < 2:
        return None
    M = np.array([[ct.get((x, y), 0) for y in ys] for x in xs], float)
    n = M.sum()
    E = M.sum(1, keepdims=True) * M.sum(0, keepdims=True) / n
    chi2 = float(((M - E) ** 2 / np.where(E > 0, E, 1)).sum())
    return round(math.sqrt(chi2 / (n * (min(M.shape) - 1))), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan", "agg"])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--units")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "scan":
        scan(a.root, [int(x) for x in a.cases.split(",") if x], a.out)
    else:
        agg(a.inputs, a.units, a.out)


if __name__ == "__main__":
    main()
