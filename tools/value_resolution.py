# SPDX-License-Identifier: MIT
"""Value (amplitude) resolution of every VitalDB track and lab test. Aggregates only leave this machine.

    python tools/value_resolution.py tracks --root R --cases 1,2 --out part.json
    python tools/value_resolution.py merge part*.json --out tracks.json
    python tools/value_resolution.py labs --root R --out labs.json

tracks: per track, from the file header and the stored values:
  storage format (dtype, bits), gain and offset (physical = count * gain + offset for integer formats),
  the step between stored counts actually used (gcd of count differences: 1 = every count used, 16 = a 12-bit
  converter stored in 16 bits), the count range used (effective bits = log2 of p0.1-p99.9 span), the
  resulting physical resolution (gain * step) and full-scale range (gain * 2^bits). Float tracks: the
  smallest step between distinct values (the value grid, e.g. 0.1 or 0.01), and whether all values sit on it.
labs: per test, the number of decimal places. The CSV drops trailing zeros, so a test reported to d places
  shows fewer places only when its last digit is 0 (about 1 in 10). Short values are compared with that
  expectation (the mean count of each non-zero last digit); a ratio well above 1 means mixed precision.
  Short-value share is split by whether the draw included a blood-gas panel (ph, po2, pco2, hco3, be, sao2
  at the same case and time), which separates the blood-gas analyser from the central lab.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from functools import reduce
from pathlib import Path

import numpy as np

ABGA = {"ph", "po2", "pco2", "hco3", "be", "sao2"}


def gcd_step(x):
    d = np.abs(np.diff(np.unique(x.astype(np.int64))))
    d = d[d > 0][:200000]
    return int(reduce(math.gcd, d.tolist())) if d.size else None


def float_grid(x):
    """(smallest step between distinct values, coarsest decimal grid every value sits on)."""
    u = np.unique(np.round(x[np.isfinite(x)].astype(np.float64), 9))
    if u.size < 3:
        return None, None
    d = np.diff(u)
    d = d[d > 1e-9]
    best = None
    for g in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        if np.mean(np.abs(u / g - np.round(u / g)) < 1e-3) > 0.999:
            best = g
    return float(np.min(d)), best


def scan(root, cases):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import FORMAT_DTYPES, read_vital
    A = defaultdict(lambda: {"cases": 0, "fmt": Counter(), "gain": Counter(), "offset": Counter(), "unit": Counter(),
                             "step": Counter(), "eff_bits": [], "lo": [], "hi": [], "fq": Counter(), "fgrid": Counter(),
                             "type": Counter()})
    for c in cases:
        try:
            v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital")
        except Exception:
            continue
        for n, tr in v.tracks.items():
            if tr.track_type not in (1, 2) or not len(tr.values):
                continue
            a = A[n]
            a["cases"] += 1
            a["type"][int(tr.track_type)] += 1
            a["fmt"][f"{tr.fmt}:{FORMAT_DTYPES[tr.fmt].name}"] += 1
            a["gain"][repr(float(tr.gain))] += 1
            a["offset"][repr(float(tr.offset))] += 1
            a["unit"][str(tr.unit)] += 1
            x = np.asarray(tr.values)
            if tr.is_integer_format:
                x = x[: 2_000_000]
                st = gcd_step(x)
                a["step"][str(st)] += 1
                lo, hi = np.percentile(x, [0.1, 99.9])
                a["lo"].append(float(lo))
                a["hi"].append(float(hi))
                if hi > lo:
                    a["eff_bits"].append(float(np.log2((hi - lo) / (st or 1) + 1)))
            else:
                q, g = float_grid(x[: 500_000])
                a["fq"][repr(None if q is None else round(q, 6))] += 1
                a["fgrid"][repr(g)] += 1
    out = {}
    for n, a in A.items():
        out[n] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in a.items()}
    return out


def merge(paths):
    M = {}
    for p in paths:
        for n, a in json.load(open(p)).items():
            if n not in M:
                M[n] = a
                continue
            m = M[n]
            m["cases"] += a["cases"]
            for k in ("fmt", "gain", "offset", "unit", "step", "fq", "fgrid", "type"):
                c = Counter(m[k])
                c.update(a[k])
                m[k] = dict(c)
            for k in ("eff_bits", "lo", "hi"):
                m[k] += a[k]
    S = {}
    for n, m in sorted(M.items()):
        g = Counter(m["gain"]).most_common(1)[0][0]
        fmt = Counter(m["fmt"]).most_common(1)[0][0]
        bits = int(fmt.split(":")[1].lstrip("uintfloat") or 0) if ("int" in fmt) else None
        st = Counter(m["step"]).most_common(1)[0][0] if m["step"] else None
        res = {"cases": m["cases"], "type": m["type"], "fmt": m["fmt"], "unit": m["unit"],
               "gain": dict(Counter(m["gain"]).most_common(4)), "offset": dict(Counter(m["offset"]).most_common(3))}
        if bits:
            gain = float(g)
            stp = int(st) if st not in (None, "None") else 1
            res.update({"bits": bits, "count_step": dict(Counter(m["step"]).most_common(4)),
                        "physical_resolution": gain * stp, "full_scale": gain * 2 ** bits,
                        "effective_bits_p10_50_90": np.percentile(m["eff_bits"], [10, 50, 90]).round(1).tolist() if m["eff_bits"] else None,
                        "count_range_used_p50": [float(np.median(m["lo"])), float(np.median(m["hi"]))] if m["lo"] else None})
        else:
            res.update({"min_step": dict(Counter(m["fq"]).most_common(4)), "decimal_grid": dict(Counter(m["fgrid"]).most_common(4))})
        S[n] = res
    return S


def labs(root):
    root = Path(root)
    rows = list(csv.DictReader(open(root / "lab_data.csv", encoding="utf-8-sig")))
    panel = defaultdict(set)
    for r in rows:
        panel[(r["caseid"], r["dt"])].add(r["name"])

    def dec(s):
        s = s.strip()
        return len(s.split(".")[1]) if "." in s else 0

    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)
    out = {}
    for n, rs in by.items():
        d = np.array([dec(r["result"]) for r in rs])
        mx = int(np.percentile(d, 99.5))
        o = {"n": len(rs), "decimals": dict(Counter(d.tolist())), "reported_places": mx}
        if mx > 0:
            last = Counter(r["result"].strip()[-1] for r, x in zip(rs, d) if x == mx)
            exp = float(np.mean([last[str(k)] for k in range(1, 10)]))
            short = int((d < mx).sum())
            o["short_observed"], o["short_expected"] = short, round(exp, 1)
            o["short_ratio"] = round(short / exp, 2) if exp else None
        else:
            o["values_with_decimals"] = int((d > 0).sum())
        for key, sel in (("with_bloodgas_panel", True), ("without_bloodgas_panel", False)):
            s = [dec(r["result"]) for r in rs if bool(panel[(r["caseid"], r["dt"])] & ABGA) == sel]
            if s:
                s = np.array(s)
                o[key] = {"n": int(s.size), "share_below_reported_places": round(float(np.mean(s < mx)), 3) if mx else None,
                          "share_with_decimals": round(float(np.mean(s > 0)), 3)}
        out[n] = o
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tracks", "merge", "labs"])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "tracks":
        J = scan(a.root, [int(x) for x in a.cases.split(",") if x and int(x) <= 6388])
    elif a.mode == "merge":
        J = merge(a.inputs)
    else:
        J = labs(a.root)
    json.dump(J, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
