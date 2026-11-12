# SPDX-License-Identifier: MIT
"""When the monitor's HR changes, how old is the beat that changed it? The delay distribution seen at value changes,
against the recording's fitted HR delay.

    python examples/hr_change_age.py scan --root R --cases 1,2,3 --out part.jsonl [--every 1]
    python examples/hr_change_age.py agg part*.jsonl --groups hr_delay_group_labels.csv --out DIR

Follows up metrology-atlas study numeric-delay-check (the monitor's HR changes about 1 s later than its fitted delay
plus half the poll gap predicts). Per recording (first 4 hours), as tools/numeric_recompute_lag.py: R peaks on the
SNUADC stamp-model times, the atlas's HR rule recomputed at every beat (60 / mean of the last 8 R-R intervals within
15% of their median, truncated), and the recording's fitted HR delay d (numeric_delay_labels.csv, quality good). At
every monitor HR stamp t whose value x differs from the previous stamp's (a change), the transition beat is the latest
beat with close c <= t at which the recomputed value becomes x (the beat before gives another value), searched over
the 12 s before t. The age t - c is recorded relative to d. Under the delay convention (a value stamped at t was
computed at t - d - phi, phi uniform over the gap since the previous stamp) the age minus d lies in [0, gap) with mean
about half the gap. scan writes per recording a histogram of age - d (0.1 s bins, -4 to 8 s) and counts; agg pools
them by HR delay group. Aggregates only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fallback_rate import load_step_rates  # noqa: E402
from numeric_derivation_pilot import beat_series, numeric_obs  # noqa: E402
from numeric_recompute_lag import hr_rule  # noqa: E402
from pleth_hr_derivation import MAX_HOURS  # noqa: E402
from vent_sync_pilot import snu_times  # noqa: E402

warnings.filterwarnings("ignore")
EDGES = np.round(np.arange(-4.0, 8.0001, 0.1), 2)
SEARCH_S = 12.0


def load_case(root, c, u, step_rates):
    """Beat closes and the recomputed HR (tc, y), and the monitor's HR stamps, values and gaps; None if missing."""
    from vitaldb_reader.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital",
                   track_names={"SNUADC/ECG_II", "SNUADC/ECG_V5", "Solar8000/HR"})
    ecg = v.tracks.get("SNUADC/ECG_II")
    if ecg is None or not ecg.record_times.size:
        ecg = v.tracks.get("SNUADC/ECG_V5")
    hr = v.tracks.get("Solar8000/HR")
    if ecg is None or hr is None or not ecg.record_times.size or not hr.record_times.size:
        return None
    te, _ = snu_times(ecg, u, step_rates)
    xe = ecg.physical().astype(np.float64)[: te.size]
    keep = te < te[0] + MAX_HOURS * 3600
    bs = beat_series(te[keep], xe[keep], float(ecg.srate), None, None)
    del xe, te
    tc, rr = bs["rr"]
    y = hr_rule(rr)
    ok = np.isfinite(y)
    tc, y = tc[ok], y[ok]
    ht, hv, hg = numeric_obs(hr, 30, 200)
    m = (ht > tc[0] + 30) & (ht < tc[-1])
    ht, hv, hg = ht[m], hv[m], hg[m]
    if tc.size < 300 or ht.size < 300:
        return None
    return tc, y, ht, hv, hg


def run_case(root, c, u, step_rates, d):
    L = load_case(root, c, u, step_rates)
    if L is None:
        return {"case": c, "skip": "needs ECG and HR, long enough"}
    tc, y, ht, hv, hg = L
    # transition beats: where the recomputed value changes
    trans = np.r_[False, y[1:] != y[:-1]]
    ch = np.r_[False, hv[1:] != hv[:-1]] & (hg < 3)
    ages = []
    found = 0
    for i in np.flatnonzero(ch):
        t, x = ht[i], hv[i]
        j1 = np.searchsorted(tc, t, side="right") - 1
        j0 = np.searchsorted(tc, t - SEARCH_S, side="left")
        if j1 <= j0:
            continue
        js = np.arange(j1, j0 - 1, -1)
        hit = js[(y[js] == x) & trans[js]]
        if hit.size:
            found += 1
            ages.append(t - tc[hit[0]])
    ages = np.array(ages)
    h = np.histogram(np.clip(ages - d, EDGES[0], EDGES[-1] - 1e-9), bins=EDGES)[0] if ages.size else np.zeros(EDGES.size - 1)
    return {"case": c, "d": d, "changes": int(ch.sum()), "found": found,
            "age_minus_d_median": round(float(np.median(ages - d)), 3) if ages.size else None,
            "age_minus_d_mean": round(float(np.mean(ages - d)), 3) if ages.size else None,
            "hist": [int(k) for k in h]}


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
    S = {"recordings": len(rows), "edges": [float(e) for e in EDGES], "by_group": {}}
    for g in ("early", "late", "all"):
        sel = [r for r in rows if "hist" in r and (g == "all" or grp.get(r["case"]) == g)]
        if not sel:
            continue
        H = np.sum([r["hist"] for r in sel], axis=0)
        mids = (EDGES[:-1] + EDGES[1:]) / 2
        tot = H.sum()
        cdf = np.cumsum(H) / max(tot, 1)
        S["by_group"][g] = {"recordings": len(sel), "changes_found": int(tot),
                           "median_age_minus_d": round(float(mids[np.searchsorted(cdf, 0.5)]), 2),
                           "mean_age_minus_d": round(float((H * mids).sum() / max(tot, 1)), 3),
                           "share_in_0_2s": round(float(H[(mids >= 0) & (mids < 2)].sum() / max(tot, 1)), 3),
                           "share_2_4s": round(float(H[(mids >= 2) & (mids < 4)].sum() / max(tot, 1)), 3),
                           "share_below_0": round(float(H[mids < 0].sum() / max(tot, 1)), 3),
                           "hist_0.5s": [int(H[(mids >= lo) & (mids < lo + 0.5)].sum()) for lo in np.arange(-4, 8, 0.5)],
                           "per_recording_median_of_means": round(float(np.median([r["age_minus_d_mean"] for r in sel
                                                                                   if r.get("age_minus_d_mean") is not None])), 3)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S["by_group"], indent=1))


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
