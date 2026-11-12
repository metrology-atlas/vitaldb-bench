# SPDX-License-Identifier: MIT
"""At which ages does the recomputed HR equal the monitor's? The share of monitor HR values that the atlas's rule
reproduces exactly when read at a fixed age a before the stamp, as a function of a - d (d the recording's fitted HR
delay).

    python examples/hr_match_profile.py scan --root R --cases 1,2,3 --out part.jsonl [--every 1]
    python examples/hr_match_profile.py agg part*.jsonl --groups hr_delay_group_labels.csv --out DIR

Follows up metrology-atlas study numeric-delay-check. Under the delay convention a value stamped at t was computed at
t - d - phi, phi uniform over the gap (about 2 s), so read at a fixed age a the match share should be high and flat for
a - d in [0, 2) and fall outside it; the middle of that plateau is where the recompute test places each value (d + 1 s).
If the plateau sits later, the fitted d is early. Per recording (first 4 hours, loaded as examples/hr_change_age.py):
the exact-match share at a - d from -3 to 6 s in 0.1 s steps; agg pools them by HR delay group (mean of recordings) and
reports the profile's peak and its half-maximum width. Aggregates only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hr_change_age import load_case  # noqa: E402
from fallback_rate import load_step_rates  # noqa: E402

OFF = np.round(np.arange(-3.0, 6.0001, 0.1), 2)


def run_case(root, c, u, step_rates, d):
    L = load_case(root, c, u, step_rates)
    if L is None:
        return {"case": c, "skip": "needs ECG and HR, long enough"}
    tc, y, ht, hv, _ = L
    prof = []
    for o in OFF:
        j = np.searchsorted(tc, ht - d - o, side="right") - 1
        ok = j >= 0
        prof.append(round(float(np.mean(y[j[ok]] == hv[ok])), 4))
    return {"case": c, "d": d, "stamps": int(ht.size), "profile": prof}


def scan(a) -> None:
    w = os.environ.get("VITALDB_WORK") or os.environ.get("W") or ""
    U = {int(r["case"]): r for r in csv.DictReader(open(f"{w}/results/sync_full/units.csv"))}
    step_rates = load_step_rates(f"{w}/results/pat/snuadc_step_rates.json")
    D = {int(r["case"]): float(r["delay_from_window_end_s"]) for r in csv.DictReader(open(f"{w}/results/pat/numeric_delay_labels.csv"))
         if r["track"] == "Solar8000/HR" and r.get("quality") == "good"}
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t0 = time.time()
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done or c % a.every or c not in D:
            continue
        if a.time_budget and time.time() - t0 > a.time_budget:
            break
        t1 = time.time()
        try:
            r = run_case(a.root, c, U.get(c, {}), step_rates.for_case(c), D[c])
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:300]}
        r["secs"] = round(time.time() - t1, 1)
        with open(out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


def agg(a) -> None:
    rows = list({r["case"]: r for p in a.parts for l in Path(p).read_text().splitlines() if l.strip()
                 for r in [json.loads(l)]}.values())
    grp = {int(r["case"]): r["group"] for r in csv.DictReader(open(a.groups))}
    S = {"recordings": len(rows), "offsets_s": [float(o) for o in OFF], "by_group": {}}
    for g in ("early", "late", "all"):
        P = np.array([r["profile"] for r in rows if "profile" in r and (g == "all" or grp.get(r["case"]) == g)])
        if not len(P):
            continue
        m = P.mean(axis=0)
        k = int(np.argmax(m))
        half = m.min() + (m[k] - m.min()) / 2
        above = np.flatnonzero(m >= half)
        S["by_group"][g] = {"recordings": int(len(P)), "peak_offset_s": float(OFF[k]), "peak_share": round(float(m[k]), 4),
                           "half_max_from_s": float(OFF[above[0]]), "half_max_to_s": float(OFF[above[-1]]),
                           "half_max_centre_s": round(float((OFF[above[0]] + OFF[above[-1]]) / 2), 2),
                           "profile_mean": [round(float(v), 4) for v in m]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({g: {k: v for k, v in d.items() if k != "profile_mean"} for g, d in S["by_group"].items()}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--root", required=True)
    s.add_argument("--cases", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--every", type=int, default=1)
    s.add_argument("--time-budget", type=float, default=0)
    g = sub.add_parser("agg")
    g.add_argument("parts", nargs="+")
    g.add_argument("--groups", required=True)
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
