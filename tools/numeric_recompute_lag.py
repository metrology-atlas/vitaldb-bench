# SPDX-License-Identifier: MIT
"""Does each route to PLETH_HR's window get its own numeric right? The monitor's HR and PLETH_HR against the same
numerics recomputed from the waveforms, per recording; a correct window and delay give a lag of zero.

    P=4 O=$W/examples/numeric_recompute_lag_v1 bash examples/run_scan.sh numeric_recompute_lag
    python tools/numeric_recompute_lag.py agg $W/examples/numeric_recompute_lag_v1/part_*.jsonl --out DIR

Why: the two routes to where PLETH_HR's window sits differ by 1.0 s, 0.75 s beyond the pulse-arrival time, more at
fast heart rates (examples/pr_window_routes.py), and the comparison's method is unbiased on synthetic data
(examples/sim_pr_window_routes.py). This finds which route's own numeric is off, by the comparison's own machinery.
It also puts both waveforms on one time base: tools/hr_derivation.py read the ECG on the stamp model, but
tools/pleth_hr_derivation.py read the plethysmogram on its raw record stamps, which run late of the stamp model in
step-mode recordings.

Per recording (every --every-th case; first 4 hours of plethysmogram), on the SNUADC stamp-model times
(tools/vent_sync_pilot.snu_times) for both ECG_II (V5 if absent) and PLETH:
  HR       recomputed as the atlas's rule (60 / mean of the last 8 R-R intervals within 15% of their median,
           truncated), each value ready at its last beat + the recording's HR label delay (results/pat/
           numeric_delay_labels.csv, quality good), and stamped at the monitor's own HR stamps (the latest value
           ready by then);
  PLETH_HR recomputed as rmean|n|10 (within 25%, rounded) from pulses (tools/pleth_hr_derivation.pulses), its delay
           fitted here in the uniform-phase convention (pleth_hr_derivation.phase_delay) on the stamp model, and
           also on the raw stamps, as the derivation did; stamped at the monitor's PLETH_HR stamps.
Each recomputed value is placed at its last beat + d + WAIT, WAIT = 1 s: the delay convention has a value stamped at
t computed at t - d - phi, phi uniform over the 2 s gap, so on average the monitor's value is half a gap older than a
value recomputed at every beat (a pilot without WAIT read both lags about one record negative). With the right
window and delay each lag is then zero on average; what matters for the two routes is PLETH_HR's lag minus HR's.
Then, as examples/hr_vs_pulse_rate.py does for HR against PLETH_HR: the recomputed value held and read at the
monitor's stamps + L (L = -6..6 s, 0.5 s), changes over 10 s blocks, sufficient statistics per lag; and the median of
raw minus stamp-model time at the pulses. agg reads each pair's lag of best agreement (record by record at the plateau
midpoints, 2k + 1 s, by a parabola), pooled, by HR delay group and by thirds of heart rate; the delays; the raw-model offset.
Aggregates only.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from hr_vs_pulse_rate import BLOCK, RECORD_S, _corr, _held  # noqa: E402
from numeric_derivation_pilot import beat_series, numeric_obs  # noqa: E402
from pat import sample_times  # noqa: E402
from pleth_hr_derivation import MAX_HOURS, phase_delay, pulses, roll_n  # noqa: E402
from vent_sync_pilot import snu_times  # noqa: E402

warnings.filterwarnings("ignore")
LAGS = np.round(np.arange(-6, 6.01, 0.5), 1)
KEY = "rmean|n|10"
WAIT = 1.0          # half the 2 s gap: the mean wait for the poll in the delay convention


def hr_rule(rr):
    """60 / mean of the last 8 R-R intervals within 15% of their median, truncated (NaN before 8 are available)."""
    out = np.full(rr.size, np.nan)
    if rr.size < 8:
        return out
    w = np.lib.stride_tricks.sliding_window_view(rr, 8)
    med = np.median(w, axis=1)[:, None]
    keep = np.abs(w - med) <= 0.15 * med
    m = np.where(keep, w, 0).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)
    out[7:] = np.floor(60.0 / m)
    return out


def stamp_at(ready, v, at):
    """The latest recomputed value ready by each monitor stamp (NaN before the first)."""
    o = np.argsort(ready)
    ready, v = ready[o], v[o]
    i = np.searchsorted(ready, at, side="right") - 1
    out = np.full(at.size, np.nan)
    ok = i >= 0
    out[ok] = v[i[ok]]
    return out


def lag_stats(t, x_mon, x_rec):
    """Per lag, the change statistics of the monitor's series against the recomputed one held at stamp + L."""
    t0 = t[0]
    blk = np.floor((t - t0) / BLOCK).astype(int)
    fin = np.isfinite(x_rec)
    out = {}
    for L in LAGS:
        p = _held(t[fin], x_rec[fin], t + L)
        ok = np.isfinite(p)
        if ok.sum() < 100:
            continue
        b = blk[ok]
        nb = np.bincount(b)
        mh = np.bincount(b, x_mon[ok]) / np.where(nb, nb, 1)
        mp = np.bincount(b, p[ok]) / np.where(nb, nb, 1)
        full = nb >= 3
        k = full[1:] & full[:-1]
        dh, dp = np.diff(mh)[k], np.diff(mp)[k]
        out[str(L)] = [int(dh.size), round(float(dh.sum()), 3), round(float(dp.sum()), 3), round(float((dh * dh).sum()), 3),
                       round(float((dp * dp).sum()), 3), round(float((dh * dp).sum()), 3)]
    return out


def run_case(root, c, u, step_rates, d_hr):
    from vitaldb_reader.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital",
                   track_names={"SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/PLETH", "Solar8000/HR", "Solar8000/PLETH_HR"})
    ecg = v.tracks.get("SNUADC/ECG_II")
    if ecg is None or not ecg.record_times.size:
        ecg = v.tracks.get("SNUADC/ECG_V5")
    pl, hr, pr = (v.tracks.get(n) for n in ("SNUADC/PLETH", "Solar8000/HR", "Solar8000/PLETH_HR"))
    if any(x is None or not x.record_times.size for x in (ecg, pl, hr, pr)):
        return {"case": c, "skip": "needs ECG, PLETH, HR and PLETH_HR"}
    res = {"case": c}
    # ---- HR from R peaks on the stamp model
    te, _ = snu_times(ecg, u, step_rates)
    xe = ecg.physical().astype(np.float64)[: te.size]
    keep = te < te[0] + MAX_HOURS * 3600
    bs = beat_series(te[keep], xe[keep], float(ecg.srate), None, None)
    del xe, te
    tc, rr = bs["rr"]
    ht, hv, _ = numeric_obs(hr, 30, 200)
    if d_hr is not None and tc.size > 200:
        y = hr_rule(rr)
        ok = np.isfinite(y)
        m = (ht > tc[0] + 30) & (ht < tc[-1])
        if ok.sum() > 200 and m.sum() > 300:
            rec = stamp_at(tc[ok] + d_hr + WAIT, y[ok], ht[m])
            res["hr"] = lag_stats(ht[m], hv[m], rec)
            res["hr_delay_s"] = d_hr
    # ---- PLETH_HR from pulses, on the stamp model and on the raw stamps
    tp_m, _ = snu_times(pl, u, step_rates)
    tp_r = sample_times(pl)[: tp_m.size]
    xp = pl.physical().astype(float)[: tp_m.size]
    keep = tp_m < tp_m[0] + MAX_HOURS * 3600
    tp_m, tp_r, xp = tp_m[keep], tp_r[keep], xp[keep]
    pk_m = pulses(tp_m, xp, float(pl.srate))
    if pk_m.size < 200:
        return res | {"skip_pr": "too few pulses"}
    # the same pulses on the raw stamps: the raw time at each stamp-model pulse time
    pk_r = np.interp(pk_m, tp_m, tp_r)
    res["raw_minus_model_s"] = round(float(np.median(pk_r - pk_m)), 4)
    pt, pv, pg = numeric_obs(pr, 30, 200)
    res["median_interval_s"] = round(float(np.median(np.diff(pk_m)[(np.diff(pk_m) > 0.25) & (np.diff(pk_m) < 2.5)])), 3)
    for base, pk in (("model", pk_m), ("raw", pk_r)):
        iv, te_ = np.diff(pk), pk[1:]
        stat = roll_n(iv, 10, "rmean")
        m = (pt > te_[0] + 30) & (pt < te_[-1])
        if m.sum() < 300:
            continue
        ts, yi = pt[m], np.round(pv[m]).astype(int)
        gap = np.where(np.isfinite(pg[m]) & (pg[m] > 0) & (pg[m] < 10), pg[m], 2.0)
        sel = np.linspace(0, ts.size - 1, min(ts.size, 3000)).astype(int)
        ph = phase_delay(KEY, ("n", stat, 10), ts[sel], yi[sel], gap[sel], te_)
        d, score = ph[f"{KEY}|round"]
        res[f"pr_delay_{base}_s"], res[f"pr_exact_{base}"] = d, score
        if base == "model":
            ok = np.isfinite(stat)
            val = np.round(60.0 / stat[ok])
            # a value stamped at t was ready by t - phi; the recomputed value ready at close + d is stamped at the next
            rec = stamp_at(te_[ok] + d + WAIT, val, pt[m])
            res["pr"] = lag_stats(pt[m], pv[m], rec)
    return res


def scan(a) -> None:
    w = os.environ.get("VITALDB_WORK") or os.environ.get("W") or ""
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units or f"{w}/results/sync_full/units.csv"))}
    step_rates = json.loads(Path(a.step_rates or f"{w}/results/pat/snuadc_step_rates.json").read_text())
    D = {}
    for r in csv.DictReader(open(a.labels or f"{w}/results/pat/numeric_delay_labels.csv")):
        if r["track"] == "Solar8000/HR" and r.get("quality") == "good":
            D[int(r["case"])] = float(r["delay_from_window_end_s"])
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t0 = time.time()
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done or c % a.every:
            continue
        if a.time_budget and time.time() - t0 > a.time_budget:
            break
        t1 = time.time()
        try:
            r = run_case(a.root, c, U.get(c, {}), step_rates, D.get(c))
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:300]}
        r["secs"] = round(time.time() - t1, 1)
        with open(out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


def peak(rows, key):
    """Lag of best agreement, record by record (k = -3..2), each read at its plateau midpoint 2k + 1 s, by a parabola,
    pooled over rows. A read at exactly 2k s falls on a plateau's edge (the recorder's gaps are 1.999-2.001 s) and
    gets record k or k - 1 by chance (examples/sim_pr_window_routes.py --jitter)."""
    rec = list(range(-3, 3))
    c = []
    for k in rec:
        t = np.zeros(6)
        for r in rows:
            x = r.get(key, {}).get(str(float(RECORD_S * k + 1)))
            if x:
                t += np.array(x, float)
        c.append(_corr(t))
    c = np.array(c)
    i = int(np.clip(np.nanargmax(c), 1, len(rec) - 2))
    y0, y1, y2 = c[i - 1], c[i], c[i + 1]
    return RECORD_S * (rec[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))


def agg(a) -> None:
    rows = list({r["case"]: r for p in a.parts for l in Path(p).read_text().splitlines() if l.strip()
                 for r in [json.loads(l)]}.values())
    w = os.environ.get("VITALDB_WORK") or os.environ.get("W") or ""
    grp = {int(r["case"]): r["group"] for r in csv.DictReader(open(a.groups or f"{w}/results/pat/hr_delay_group_labels.csv"))}
    rng = np.random.default_rng(0)

    def block(sub):
        o = {"recordings": len(sub)}
        for key in ("hr", "pr"):
            s = [r for r in sub if key in r]
            if len(s) >= 30:
                bt = [peak([s[i] for i in rng.integers(0, len(s), len(s))], key) for _ in range(30)]
                o[f"{key}_recordings"] = len(s)
                o[f"{key}_lag_s"] = round(peak(s, key), 3)
                o[f"{key}_lag_boot_p2.5_p97.5"] = [round(float(np.percentile(bt, 2.5)), 3), round(float(np.percentile(bt, 97.5)), 3)]
        both = [r for r in sub if "hr" in r and "pr" in r]
        if len(both) >= 30:
            diff = lambda rs: peak(rs, "pr") - peak(rs, "hr")
            bt = [diff([both[i] for i in rng.integers(0, len(both), len(both))]) for _ in range(30)]
            o["pr_minus_hr_lag_s"] = round(diff(both), 3)
            o["pr_minus_hr_lag_boot_p2.5_p97.5"] = [round(float(np.percentile(bt, 2.5)), 3), round(float(np.percentile(bt, 97.5)), 3)]
        for k in ("pr_delay_model_s", "pr_delay_raw_s", "raw_minus_model_s", "hr_delay_s", "pr_exact_model", "pr_exact_raw"):
            x = np.array([r[k] for r in sub if r.get(k) is not None])
            if x.size:
                o[k.replace("_s", "") + "_median" + ("_s" if k.endswith("_s") else "")] = round(float(np.median(x)), 3)
        return o

    ok = [r for r in rows if "hr" in r or "pr" in r]
    summ = {"read": len(rows), "used": len(ok), "errors": sum("error" in r for r in rows), "all": block(ok), "by_group": {},
            "by_rate": []}
    for g in ("early", "late"):
        summ["by_group"][g] = block([r for r in ok if grp.get(r["case"]) == g])
    iv = np.array([r.get("median_interval_s", np.nan) for r in ok])
    fin = np.isfinite(iv)
    for lo, hi in zip(np.quantile(iv[fin], [0, 1 / 3, 2 / 3]), np.quantile(iv[fin], [1 / 3, 2 / 3, 1])):
        sub = [r for r, x in zip(ok, iv) if np.isfinite(x) and lo <= x <= hi]
        summ["by_rate"].append({"interval_s": [round(float(lo), 3), round(float(hi), 3)], **block(sub)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    for k in ("--root", "--cases", "--out"):
        s.add_argument(k, required=True)
    for k in ("--units", "--step-rates", "--labels"):
        s.add_argument(k)
    s.add_argument("--every", type=int, default=1, help="every N-th case (case %% N == 0)")
    s.add_argument("--time-budget", type=float, default=0)
    g = sub.add_parser("agg")
    g.add_argument("parts", nargs="+")
    g.add_argument("--out", required=True)
    g.add_argument("--groups")
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
