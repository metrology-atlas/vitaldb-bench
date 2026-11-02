# SPDX-License-Identifier: MIT
"""How the Solar8000 computes its heart rate (HR): the window length and the outlier rule, settled.

    bash examples/run_scan.sh hr_derivation                   (every tenth recording, the pilot's 639)
    python tools/hr_derivation.py agg $W/examples/hr_derivation/part_*.jsonl --out $W/examples/hr_derivation

Why: the numeric derivation pilot (tools/numeric_derivation_pilot.py) found HR best reproduced by a mean of the
R-R intervals that leaves out intervals far from the window's median (406 of 633 recordings; 80% of values
exact, truncated), not the median the atlas assumed. Its window grid skipped 7 and 9 beats, and the winners split
between 8 and 10, and it tried only one outlier band (25%, the band PLETH_HR uses). This rerun settles both.

Reference and model as the pilot: R peaks of SNUADC/ECG_II (or V5) on the stamp-model line, R-R intervals
0.25-2.5 s; the value stamped at t was computed at t - d - phi, phi uniform over the gap since the previous
record; truncated to the step (and rounded, for the main rule, as a check). Candidates, 60 over:
  rmean|N|b   the mean of the last N intervals within b of their median, b in 10, 15, 20, 25, 30, 50%
  median|N    the median of the last N intervals
  mean|N      the plain mean of the last N intervals
  tmean|N     the mean without the longest and the shortest
for N = 4..16. Each is fitted for its delay (-3..15 s) on up to 3,000 records and scored by the share of up
to 8,000 records it reproduces exactly at its best delay (refined to 0.05 s).

scan writes per recording the delay, log-likelihood and exact share of every candidate, and the median R-R
interval; agg ranks them (wins, pooled exact), gives the winning N and band per recording, the delay and window
centre (delay + N x median R-R / 2) by HR delay group (results/pat/hr_delay_group_labels.csv), and how far the
winner beats its neighbours (N - 1, N + 1; the next band). Aggregates only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).parent))
from numeric_derivation_pilot import beat_series, fit, best_of, lookup, numeric_obs, step_res, subsample  # noqa: E402
from vent_sync_pilot import snu_times  # noqa: E402
from vitaldb_reader.vital_file import read_vital  # noqa: E402

warnings.filterwarnings("ignore")
NS = list(range(4, 17))
BANDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
SAMPLE_MOD, SAMPLE_REM = 10, 3
D1 = np.round(np.arange(-3.0, 15.0001, 0.2), 3)


def roll(x, N, how, band=0.25):
    if x.size < N:
        return np.full(x.size, np.nan)
    w = sliding_window_view(x, N)
    if how == "mean":
        y = w.mean(axis=1)
    elif how == "median":
        y = np.median(w, axis=1)
    elif how == "tmean":
        y = (w.sum(axis=1) - w.max(axis=1) - w.min(axis=1)) / (N - 2)
    else:
        med = np.median(w, axis=1)[:, None]
        keep = np.abs(w - med) <= band * med
        y = np.where(keep, w, 0).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)
    return np.r_[np.full(N - 1, np.nan), 60.0 / y]


def candidates(rr):
    out = {}
    for N in NS:
        for b in BANDS:
            out[f"rmean|{N}|{int(b * 100)}"] = roll(rr, N, "rmean", b)
        out[f"median|{N}"] = roll(rr, N, "median")
        out[f"mean|{N}"] = roll(rr, N, "mean")
        out[f"tmean|{N}"] = roll(rr, N, "tmean")
    return out


def run_case(root, c, u, step_rates):
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital",
                   track_names={"SNUADC/ECG_II", "SNUADC/ECG_V5", "Solar8000/HR"})
    tr = v.tracks.get("SNUADC/ECG_II")
    if tr is None or not tr.record_times.size:
        tr = v.tracks.get("SNUADC/ECG_V5")
    hr = v.tracks.get("Solar8000/HR")
    if tr is None or hr is None or not tr.record_times.size:
        return {"case": c, "skip": "no ECG with HR"}
    t, _ = snu_times(tr, u, step_rates)
    x = tr.physical().astype(np.float64)[: t.size]
    bs = beat_series(t, x, float(tr.srate), None, None)
    del x, t
    tc, rr = bs["rr"]
    if tc.size < 200:
        return {"case": c, "skip": "too few beats"}
    ot, ov, og = numeric_obs(hr, 25, 250)
    m = (ot > tc[0] + 60) & (ot < tc[-1])
    ot, ov, og = ot[m], ov[m], og[m]
    if ot.size < 300:
        return {"case": c, "skip": "too few HR records over the ECG"}
    q = step_res(ov)
    span = 225.0
    s_t, s_v, s_g = subsample(ot, ov, og, 3000)
    a_t, a_v, a_g = subsample(ot, ov, og, 8000)
    res = {"case": c, "records": int(ot.size), "q": q, "median_rr_s": round(float(np.median(rr)), 4), "cands": {}}
    for name, y in candidates(rr).items():
        ok = np.isfinite(y)
        if ok.sum() < 50:
            continue
        quants = ("floor", "round") if name == "rmean|8|25" or name == "rmean|10|25" else ("floor",)
        for qm in quants:
            ll, _, _ = fit(s_t, s_v, s_g, tc[ok], y[ok], q, qm, False, span, D1, 0.2)
            k, _ = best_of(ll)
            Dr = np.round(np.arange(D1[k] - 0.4, D1[k] + 0.4001, 0.05), 3)
            ll2, _, _ = fit(a_t, a_v, a_g, tc[ok], y[ok], q, qm, False, span, Dr, 0.05)
            k2, L = best_of(ll2)
            d = float(Dr[k2])
            pred = lookup(tc[ok], y[ok], a_t - d - a_g / 2)
            pq = np.floor(pred / q + 1e-9) * q if qm == "floor" else np.round(pred / q) * q
            f = np.isfinite(pq)
            ex = float(np.mean(np.abs(pq[f] - a_v[f]) < q / 2)) if f.any() else 0.0
            key = name if qm == "floor" else name + "|round"
            res["cands"][key] = [d, round(L / a_t.size, 5), round(ex, 4)]
    return res


def scan(a):
    w = os.environ.get("VITALDB_WORK") or os.environ.get("W") or ""
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units or f"{w}/results/sync_full/units.csv"))}
    step_rates = json.loads(Path(a.step_rates or f"{w}/results/pat/snuadc_step_rates.json").read_text())
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done or (not a.all and c % SAMPLE_MOD != SAMPLE_REM):
            continue
        t1 = time.time()
        try:
            r = run_case(a.root, c, U.get(c, {}), step_rates)
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:300]}
        r["secs"] = round(time.time() - t1, 1)
        with open(out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


def agg(a):
    rows = [json.loads(l) for p in a.parts for l in Path(p).read_text().splitlines() if l.strip()]
    rows = list({r["case"]: r for r in rows}.values())
    ok = [r for r in rows if r.get("cands")]
    w = os.environ.get("VITALDB_WORK") or os.environ.get("W") or ""
    gp = Path(a.groups or f"{w}/results/pat/hr_delay_group_labels.csv")
    group = {int(r["case"]): r["group"] for r in csv.DictReader(open(gp))} if gp.exists() else {}
    names = sorted({n for r in ok for n in r["cands"]})
    pooled = {n: [0.0, 0] for n in names}
    wins_ex, wins_ll = Counter(), Counter()
    for r in ok:
        sc = r["cands"]
        floor = {n: v for n, v in sc.items() if not n.endswith("|round")}
        wins_ex[max(floor, key=lambda n: floor[n][2])] += 1
        wins_ll[max(floor, key=lambda n: floor[n][1])] += 1
        for n, v in sc.items():
            pooled[n][0] += v[2] * r["records"]; pooled[n][1] += r["records"]
    pe = {n: pooled[n][0] / pooled[n][1] for n in names if pooled[n][1]}
    rank = sorted(pe, key=lambda n: -pe[n])
    top = rank[0]
    fam = lambda n: n.split("|")[0]  # noqa: E731
    best_n = Counter(int(max(((n, v) for n, v in r["cands"].items() if n.startswith("rmean|") and n.endswith("|25")),
                             key=lambda kv: kv[1][2])[0].split("|")[1]) for r in ok)
    best_band = Counter(int(max(((n, v) for n, v in r["cands"].items() if n.startswith("rmean|")
                                 and not n.endswith("round")), key=lambda kv: kv[1][2])[0].split("|")[2]) for r in ok)
    grid = {f"N{N}": {f"b{int(b * 100)}": round(pe.get(f"rmean|{N}|{int(b * 100)}", float('nan')), 4) for b in BANDS}
            for N in NS}
    others = {f: {f"N{N}": round(pe.get(f"{f}|{N}", float('nan')), 4) for N in NS} for f in ("median", "mean", "tmean")}
    by_group = {}
    for g in ("early", "late", "uncertain", None):
        sel = [r for r in ok if group.get(r["case"]) == g] if g else ok
        if not sel or top not in sel[0]["cands"]:
            continue
        N = int(top.split("|")[1]) if fam(top) != "x" else 0
        d = np.array([r["cands"][top][0] for r in sel if top in r["cands"]])
        cen = np.array([r["cands"][top][0] + N * r["median_rr_s"] / 2 for r in sel if top in r["cands"]])
        by_group[g or "all"] = {"recordings": len(sel), "delay_median_s": round(float(np.median(d)), 2),
                                "delay_p10_p90_s": [round(float(np.percentile(d, p)), 2) for p in (10, 90)],
                                "centre_median_s": round(float(np.median(cen)), 2)}
    # how much the winner beats its neighbours, per recording (exact share, points)
    margin = {}
    if fam(top) == "rmean":
        _, N, b = top.split("|")
        N = int(N)
        for lab, nb in (("N-1", f"rmean|{N - 1}|{b}"), ("N+1", f"rmean|{N + 1}|{b}"),
                        ("median same N", f"median|{N}"), ("plain mean same N", f"mean|{N}")):
            dlt = [r["cands"][top][2] - r["cands"][nb][2] for r in ok if nb in r["cands"] and top in r["cands"]]
            if dlt:
                margin[lab] = {"median_points": round(100 * float(np.median(dlt)), 2),
                               "recordings_winner_better": int(sum(x > 0 for x in dlt)), "of": len(dlt)}
    rf = {k: [round(100 * float(np.median([r["cands"][k][2] - r["cands"][k + "|round"][2]
                                            for r in ok if k + "|round" in r["cands"]])), 2)]
          for k in ("rmean|8|25", "rmean|10|25")}
    summ = {"recordings_read": len(rows), "analysed": len(ok),
            "skips": Counter(r.get("skip") or ("error" if r.get("error") else "ok") for r in rows),
            "ranking": [{"candidate": n, "pooled_exact": round(pe[n], 4), "wins_exact": wins_ex[n],
                         "wins_ll": wins_ll[n]} for n in rank[:20]],
            "best": top, "best_N_at_25pct": dict(sorted(best_n.items())), "best_band": dict(sorted(best_band.items())),
            "rmean_pooled_exact_by_N_and_band": grid, "other_families_pooled_exact": others,
            "winner_margin": margin, "floor_minus_round_points": rf, "by_delay_group": by_group}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    for k in ("--root", "--cases", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--units"); s.add_argument("--step-rates")
    s.add_argument("--all", action="store_true", help="every recording, not every tenth")
    s.add_argument("--time-budget", type=float, default=0)
    g = sub.add_parser("agg")
    g.add_argument("parts", nargs="+"); g.add_argument("--out", required=True); g.add_argument("--groups")
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
