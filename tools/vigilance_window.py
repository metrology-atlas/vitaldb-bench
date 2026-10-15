# SPDX-License-Identifier: MIT
"""Vigilance HR_AVG: infer its averaging window and delay against ECG R-R intervals (SNUADC stamp-model time),
with the uniform time/value model of tools/numeric_derivation_pilot.py over longer windows. Per case, this
machine only.

    python tools/vigilance_window.py --root R --cases 1,2 --units units.csv --step-rates rates.json --out out.jsonl
    python tools/vigilance_window.py --agg out.jsonl --summary summary.json

Candidates: 60/mean(R-R), mean(60/R-R), median and EWMA over N beats (N = 32..4096) or T seconds (T = 30..3600);
delay d from 0 to 900 s. The delay is not allowed below zero (the availability rule: a value cannot be emitted
before its window ends). An earlier grid from -120 s, with coarser windows, let a longer window paired with a
negative delay stand in for a shorter one with a positive delay (window and delay trade off; 19 of 52 cases),
so the window grid is also finer. Polls are taken from the record stamps (2 s). The fitted delay is the clock offset
plus the device's processing delay, measured from the window's close (see numeric_window_diagram).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import numeric_derivation_pilot as ndp  # noqa: E402
from pat import r_peaks  # noqa: E402
from vent_sync_pilot import snu_times  # noqa: E402
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: E402

ndp.N_BEATS = [32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096]
ndp.T_BEATS = [30, 60, 120, 180, 300, 450, 600, 750, 900, 1050, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3600]


def run_case(root, c, u, sr):
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital",
                   track_names={"Vigilance/HR_AVG", "SNUADC/ECG_II", "SNUADC/ECG_V5"})
    tr = v.tracks.get("Vigilance/HR_AVG")
    ecg = v.tracks.get("SNUADC/ECG_II") or v.tracks.get("SNUADC/ECG_V5")
    if tr is None or ecg is None or not ecg.record_times.size:
        return {"case": c, "skip": "no HR_AVG or ECG"}
    te, _ = snu_times(ecg, u, sr)
    xe = np.asarray(ecg.physical(), float)[: te.size]
    ok = np.isfinite(xe)
    rp = r_peaks(np.where(ok, xe, np.nanmedian(xe)), float(ecg.srate))
    R = te[rp]
    rr = np.diff(R)
    g = (rr >= 0.25) & (rr <= 2.5)
    tc, x = R[1:][g], rr[g]
    ot, ov, og = ndp.numeric_obs(tr, 25, 250)
    m = (ot > tc[0] + 1800) & (ot < tc[-1])
    ot, ov, og = ot[m], ov[m], og[m]
    if ot.size < 500:
        return {"case": c, "skip": "too little overlap"}
    cands = ndp.candidates("rate_beat", tc, x)
    D = np.arange(0.0, 900.001, 2.0)
    s_t, s_v, s_g = ndp.subsample(ot, ov, og, 3000)
    q = ndp.step_res(ov)
    rows = []
    for cn, (t_, y) in cands.items():
        okc = np.isfinite(y)
        if okc.sum() < 100:
            continue
        for quant in ("round", "floor"):
            ll, b, s = ndp.fit(s_t, s_v, s_g, t_[okc], y[okc], q, quant, False, 225.0, D, 2.0)
            k, L = ndp.best_of(ll)
            rows.append({"cand": cn, "quant": quant, "ll": round(L, 2), "d": float(D[k]), "s": round(float(s[k]), 3)})
    rows.sort(key=lambda r: -r["ll"])
    best = rows[0]
    tc_, y_ = cands[best["cand"]]
    okc = np.isfinite(y_)
    Dr = np.arange(max(0.0, best["d"] - 2), best["d"] + 2.001, 0.1)  # never below zero (availability rule)
    ll, _, s = ndp.fit(*ndp.subsample(ot, ov, og, 8000), tc_[okc], y_[okc], q, best["quant"], False, 225.0, Dr, 0.1)
    k, L = ndp.best_of(ll)
    best = {**best, "d": round(float(Dr[k]), 2), "ll_all": round(L, 2), "s": round(float(s[k]), 3)}
    fam = {}
    for r in rows:
        key = "|".join(r["cand"].split("|")[:2])
        fam.setdefault(key, r)
    n_fit = int(s_t.size)
    return {"case": c, "n": int(ot.size), "q": q, "best": best, "top": rows[:5],
            "families": {k: {"cand": v_["cand"], "d": v_["d"], "dll_per_rec": round((rows[0]["ll"] - v_["ll"]) / n_fit, 4)}
                         for k, v_ in fam.items()}}


def agg(path, out):
    from collections import Counter
    R = [json.loads(l) for l in open(path)]
    G = [r for r in R if "best" in r]
    fam = Counter("|".join(r["best"]["cand"].split("|")[:2]) for r in G)
    win = {}
    for r in G:
        f, wt, w = r["best"]["cand"].split("|")
        win.setdefault(f"{f}|{wt}", []).append(float(w))
    d = [r["best"]["d"] for r in G]
    s = [r["best"]["s"] for r in G]
    S = {"cases": len(R), "fitted": len(G), "skipped": Counter(r.get("skip") for r in R if "skip" in r).most_common(),
         "best_family": fam.most_common(), "windows": {k: sorted(v) for k, v in win.items()},
         "delay_s_p10_25_50_75_90": np.percentile(d, [10, 25, 50, 75, 90]).round(1).tolist() if d else None,
         "sigma_p50": float(np.median(s)) if s else None,
         "floor_better": sum(r["best"]["quant"] == "floor" for r in G)}
    json.dump(S, open(out, "w"), indent=1)
    print(json.dumps(S, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--units")
    ap.add_argument("--step-rates")
    ap.add_argument("--out")
    ap.add_argument("--agg")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.agg:
        agg(a.agg, a.summary)
        return
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    sr = json.loads(Path(a.step_rates).read_text())
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in open(a.out)}
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        try:
            r = run_case(a.root, c, U.get(c, {}), sr)
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:300]}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
