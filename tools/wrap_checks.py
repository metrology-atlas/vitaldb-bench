# SPDX-License-Identifier: MIT
"""Checks on the whole-release wrap test (metrology-atlas study snuadc-firmware), from tools/wrap_test.py's output.

    python tools/wrap_checks.py --wrap $W/wrap_full/run/wrap_test.json --units $W/results/sync_full/units.csv \
        --modes $W/results/sync_full/snuadc_mode_label.json --ports $W/pat_census/aggf/cases.csv --out $W/wrap_full/checks

From the shared ART+PLETH steps of every tested recording:

1. **Paired against unpaired.** Steps that belong to a pair 4,295 +/- 30 s apart against the other shared steps:
   size and sign.
2. **Multiples of the wrap.** Pairs at 2 x and 3 x 4,295 s (8,590 and 12,885 s) against control spacings, as the
   wrap test counts them (shared steps, +/- 30 s; controls every 10 s from 2,000 to 13,000 s, more than 120 s from
   any multiple of the wrap).
3. **By setup.** Per SNUADC port (--ports: a CSV with case and snuadc_port, e.g. the PAT census's cases.csv,
   which covers more recordings than units.csv) and correction mode: recordings that span two wraps, and the share of them with a
   pair; and the pairs per wrap due.

4. **Phase.** Where the run recorded where each stretch starts (stretch_start_index, stretch_start_after_art_s):
   the time of each paired step since the ART track's first sample (sample count / 500 Hz), modulo 4,295 s. A
   counter that starts with the stream (a board reset when recording starts) puts every pair at one phase; a board
   that runs on between recordings spreads them evenly. Rayleigh test of uniformity, with the unpaired shared steps
   as a comparison, and the phases in ten bins.

Writes summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WRAP_S = 2 ** 32 / 1e6
TOL = 30.0


def pairs_at(shared_t: list[float], D: float) -> list[tuple[float, float]]:
    t = sorted(shared_t)
    return [(a, b) for i, a in enumerate(t) for b in t[i + 1:] if abs(b - a - D) <= TOL]


def q(x) -> dict:
    x = np.asarray(x, float)
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), **{f"p{p}": round(float(np.percentile(x, p)), 3) for p in (5, 25, 50, 75, 95)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wrap", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--ports", help="CSV with case and snuadc_port")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W = json.loads(Path(a.wrap).read_text())
    cases = W["cases"]
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    if a.ports:
        for r in csv.DictReader(open(a.ports)):
            if r.get("snuadc_port"):
                U.setdefault(int(r["case"]), {})["snuadc_port"] = r["snuadc_port"]
    modes = json.loads(Path(a.modes).read_text())
    # 1. paired against unpaired
    paired, unpaired = [], []
    for r in cases:
        ts = [x["t_s"] for x in r["shared"]]
        inp = {t for p in pairs_at(ts, WRAP_S) for t in p}
        for x in r["shared"]:
            (paired if x["t_s"] in inp else unpaired).append(0.5 * (x["art_ms"] + x["pleth_ms"]))
    P, Q = np.array(paired), np.array(unpaired)
    S = {"wrap": a.wrap, "recordings": len(cases),
         "paired_steps": {"size_ms": q(np.abs(P)), "signed_ms": q(P), "share_negative": round(float(np.mean(P < 0)), 3) if P.size else None},
         "unpaired_steps": {"size_ms": q(np.abs(Q)), "signed_ms": q(Q), "share_negative": round(float(np.mean(Q < 0)), 3) if Q.size else None}}
    # 2. multiples of the wrap
    shared = [[x["t_s"] for x in r["shared"]] for r in cases]
    def count(D):
        return sum(len(pairs_at(ts, D)) for ts in shared)
    ctrl_D = [D for D in np.arange(2000, 13001, 10) if min(abs(D - k * WRAP_S) for k in (1, 2, 3)) > 120]
    ctrl = np.array([count(D) for D in ctrl_D])
    S["multiples"] = {}
    for k in (1, 2, 3):
        D = k * WRAP_S
        n = count(D)
        # control spacings near D (within 2,000 s), as the pair count falls with spacing (fewer recordings that long)
        near = np.array([c for d, c in zip(ctrl_D, ctrl) if abs(d - D) <= 2000])
        S["multiples"][f"{k}x"] = {"spacing_s": round(D, 1), "pairs": n, "control_near_mean": round(float(near.mean()), 2) if near.size else None,
                                   "control_near_max": int(near.max()) if near.size else None,
                                   "recordings_long_enough": sum(1 for r in cases if r["stretch_s"] >= D)}
    # 3. by setup
    grp = defaultdict(lambda: {"recordings": 0, "two_wraps": 0, "two_wraps_with_pair": 0, "wraps_due": 0, "pairs": 0})
    for r in cases:
        c = r["case"]
        key = f"{U.get(c, {}).get('snuadc_port') or '?'} {modes.get(str(c), '?')}"
        g = grp[key]
        ts = [x["t_s"] for x in r["shared"]]
        np_ = len(pairs_at(ts, WRAP_S))
        g["recordings"] += 1
        g["wraps_due"] += r["wraps_due"]
        g["pairs"] += np_
        if r["stretch_s"] >= 2 * WRAP_S:
            g["two_wraps"] += 1
            g["two_wraps_with_pair"] += int(np_ > 0)
    S["by_setup"] = {k: {**v, "pairs_per_wrap_due": round(v["pairs"] / v["wraps_due"], 3) if v["wraps_due"] else None,
                         "share_two_wraps_with_pair": round(v["two_wraps_with_pair"] / v["two_wraps"], 3) if v["two_wraps"] else None}
                     for k, v in sorted(grp.items(), key=lambda kv: -kv[1]["recordings"]) if v["recordings"] >= 10}
    by_mode = defaultdict(lambda: [0, 0, 0, 0])
    for k, v in grp.items():
        m = k.split(" ", 1)[1]
        by_mode[m] = [by_mode[m][0] + v["recordings"], by_mode[m][1] + v["wraps_due"], by_mode[m][2] + v["pairs"],
                      by_mode[m][3] + v["two_wraps"]]
    S["by_mode"] = {m: {"recordings": r_, "wraps_due": w_, "pairs": p_, "pairs_per_wrap_due": round(p_ / w_, 3) if w_ else None}
                    for m, (r_, w_, p_, _) in by_mode.items()}
    ph_p, ph_u, first_u = [], [], []
    for r in cases:
        if "stretch_start_index" not in r:
            continue
        base = r["stretch_start_index"] / 500.0
        ts = [x["t_s"] for x in r["shared"]]
        pp = pairs_at(ts, WRAP_S)
        inp = {t for p_ in pp for t in p_}
        for t in ts:
            (ph_p if t in inp else ph_u).append((base + t) % WRAP_S)
        if pp:
            first_u.append(base + min(p_[0] for p_ in pp))
    if ph_p:
        def rayleigh(ph):
            z = np.exp(2j * np.pi * np.asarray(ph) / WRAP_S)
            R = abs(z.mean())
            n = z.size
            return {"n": n, "R": round(float(R), 3), "p": float(np.exp(-n * R * R)),
                    "mean_phase_s": round(float((np.angle(z.mean()) % (2 * np.pi)) / (2 * np.pi) * WRAP_S), 1)}
        S["phase"] = {"paired": rayleigh(ph_p), "unpaired": rayleigh(ph_u),
                      "paired_bins": np.histogram(ph_p, np.linspace(0, WRAP_S, 11))[0].tolist(),
                      "unpaired_bins": np.histogram(ph_u, np.linspace(0, WRAP_S, 11))[0].tolist(),
                      "first_pair_since_art_start_s": q(first_u)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
