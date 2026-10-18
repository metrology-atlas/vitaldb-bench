# SPDX-License-Identifier: MIT
"""EMR anaesthesia start/end against device markers (experiments 1 and 4). Per case, this machine only.

    python tools/emr_markers.py scan --root R --cases 1,2 --out part.jsonl
    python tools/emr_markers.py agg part*.jsonl --root R --summary summary.json
    python tools/emr_markers.py fig part*.jsonl --root R --out emr-rounding.png

scan: marker times (s from casestart, i.e. the file origin 2100-01-01 00:00:00 UTC) from numerics on recorder
stamps (Orchestra ~1 s, BIS 1 s, Primus ~7 s):
  induction  ppf_start  first Orchestra/PPF20_RATE > 0            rftn_start  first RFTN20_RATE > 0
             bis_loc    first BIS < 60 held 30 s, after BIS >= 80   vent_start  first PIP_MBAR >= 10 held 3 records
             co2_start  first Primus ETCO2 >= 20 held 3 records
  emergence  ppf_stop   last PPF20_RATE > 0 (then 0 or no record)   rftn_stop   last RFTN20_RATE > 0
             vent_stop  last PIP_MBAR >= 10 held 3 records           co2_stop    last ETCO2 >= 20 held 3 records
             bis_wake   first BIS >= 80 held 60 s after the last BIS < 60
             rec_end    end of the recording
agg: per EMR event (anestart: induction markers; aneend: emergence markers) and marker, D = marker - EMR time
  for 5-min entries (the event on an absolute 5-min clock mark) and minute entries (on the minute grid, off a
  5-min mark), with the case clock of vitaldb_bench.case_clock. Shift of the 5-min group's median and mean
  against the minute group's (bootstrap CI), and a convolution test: the minute group's D distribution
  (rounding blur 60 s) convolved with each 5-min rounding kernel - recorded = floor, nearest or ceiling of the
  true time to 5 min - scored by the log-likelihood of the 5-min group's D (KDE).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ORIGIN = 4102444800.0
TRACKS = {"Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "BIS/BIS", "Primus/PIP_MBAR", "Primus/ETCO2"}


def series(v, n):
    tr = v.tracks.get(n)
    if tr is None or not tr.record_times.size:
        return None
    t = np.asarray(tr.record_times, float) - ORIGIN
    x = np.asarray(tr.physical(), float)[: t.size]
    o = np.argsort(t, kind="stable")
    t, x = t[o], x[o]
    ok = np.isfinite(x)
    return t[ok], x[ok]


def held(mask, k):
    """Index of the first True run of length >= k (start), and of the last run's end."""
    if mask.size < k:
        return None, None
    run = np.convolve(mask.astype(int), np.ones(k, int), "valid") >= k
    idx = np.flatnonzero(run)
    if not idx.size:
        return None, None
    return int(idx[0]), int(idx[-1] + k - 1)


def held_time(t, mask, min_s):
    """First time a condition starts and holds for >= min_s seconds (records may be irregular)."""
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    for s, e in zip(starts, ends):
        if t[e] - t[s] >= min_s:
            return float(t[s])
    return None


def scan_case(root, c):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names=TRACKS)
    m = {"case": c}
    for n, key in (("Orchestra/PPF20_RATE", "ppf"), ("Orchestra/RFTN20_RATE", "rftn")):
        s = series(v, n)
        if s is None:
            continue
        t, x = s
        on = np.flatnonzero(x > 0)
        if on.size:
            m[f"{key}_start"] = float(t[on[0]])
            m[f"{key}_stop"] = float(t[on[-1]])
            m[f"{key}_first_record"] = float(t[0])
    s = series(v, "BIS/BIS")
    if s is not None:
        t, x = s
        ok = x > 0
        t, x = t[ok], x[ok]
        if t.size > 100:
            awake = np.flatnonzero(x >= 80)
            if awake.size:
                a0 = t[awake[0]]
                k = t >= a0
                loc = held_time(t[k], x[k] < 60, 30)
                if loc is not None:
                    m["bis_loc"] = loc
            low = np.flatnonzero(x < 60)
            if low.size:
                k = t > t[low[-1]]
                w = held_time(t[k], x[k] >= 80, 60)
                if w is not None:
                    m["bis_wake"] = w
            m["bis_first_record"] = float(t[0])
    for n, key, thr in (("Primus/PIP_MBAR", "vent", 10), ("Primus/ETCO2", "co2", 20)):
        s = series(v, n)
        if s is None:
            continue
        t, x = s
        a, b = held(x >= thr, 3)
        if a is not None:
            m[f"{key}_start"] = float(t[a])
            m[f"{key}_stop"] = float(t[b])
            m[f"{key}_first_record"] = float(t[0])
    m["rec_end"] = max(float(tr.record_times.max()) for tr in v.tracks.values() if tr.record_times.size) - ORIGIN
    return m


INDUCTION = ["ppf_start", "rftn_start", "bis_loc", "vent_start", "co2_start"]
EMERGENCE = ["ppf_stop", "rftn_stop", "vent_stop", "co2_stop", "bis_wake", "rec_end"]


def kde_ll(x, sample, bw):
    x = np.asarray(x, float)[:, None]
    s = np.asarray(sample, float)[None, :]
    p = np.exp(-0.5 * ((x - s) / bw) ** 2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return float(np.log(p + 1e-12).sum())


def agg(paths, root, out):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vitaldb_bench.case_clock import time_of_day  # noqa: PLC0415

    M = {}
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            if "error" not in r:
                M[r["case"]] = r
    cl = {int(r["caseid"]): r for r in csv.DictReader(open(Path(root) / "clinical_data.csv", encoding="utf-8-sig"))}
    rng = np.random.default_rng(0)
    S = {"cases_with_markers": len(M)}
    for ev, markers in (("anestart", INDUCTION), ("aneend", EMERGENCE)):
        S[ev] = {}
        for mk in markers:
            g5, g1 = [], []
            for c, m in M.items():
                if mk not in m:
                    continue
                r = cl.get(c)
                if r is None:
                    continue
                clk = time_of_day(r)
                if clk.get("status") != "ok":
                    continue
                x = float(r[ev])
                # the marker must be observable: the device recorded before it (start markers) - otherwise a
                # marker 'first value' may be truncated by the recording start
                first = m.get(mk.split("_")[0] + "_first_record")
                if mk.endswith("_start") and first is not None and m[mk] - first < 30:
                    continue
                minute = (clk["tod_s"] + x) / 60
                if abs(minute - round(minute)) > 1e-3:
                    continue
                d = m[mk] - x
                if abs(d) > 3600:
                    continue
                (g5 if int(round(minute)) % 5 == 0 else g1).append(d)
            g5, g1 = np.array(g5), np.array(g1)
            o = {"n5": int(g5.size), "n1": int(g1.size)}
            if g5.size >= 20 and g1.size >= 20:
                bs = [np.median(rng.choice(g5, g5.size)) - np.median(rng.choice(g1, g1.size)) for _ in range(1000)]
                o.update({"D_5min_p10_50_90": np.percentile(g5, [10, 50, 90]).round(0).tolist(),
                          "D_1min_p10_50_90": np.percentile(g1, [10, 50, 90]).round(0).tolist(),
                          "shift_median_s": round(float(np.median(g5) - np.median(g1)), 1),
                          "shift_ci95": np.percentile(bs, [2.5, 97.5]).round(1).tolist()})
                # convolution test. D = marker - recorded = lag - (recorded - true).
                # floor: recorded - true in (-300, 0] -> D = lag + U[0, 300); nearest: lag + U(-150, 150];
                # ceiling: lag + U(-300, 0]. The minute group gives lag (+ its own small blur).
                lag = g1 - np.median(g1) * 0 + 0.0
                bw = max(1.06 * np.std(lag) * lag.size ** -0.2, 15.0)
                lls = {}
                for name, (a, b) in {"floor": (0, 300), "nearest": (-150, 150), "ceiling": (-300, 0)}.items():
                    sim = (rng.choice(lag, 4000) + rng.uniform(a, b, 4000))
                    lls[name] = round(kde_ll(g5, sim, bw), 1)
                best = max(lls, key=lls.get)
                o["convolution_ll"] = lls
                o["best_kernel"] = best
                o["ll_margin_vs_next"] = round(lls[best] - sorted(lls.values())[-2], 1)
            S[ev][mk] = o
    json.dump(S, open(out, "w"), indent=1)
    print(json.dumps(S, indent=1))


def groups(M, cl, ev, mk, time_of_day):
    """D = marker - EMR time for 5-minute entries and minute entries (same rules as agg)."""
    g5, g1 = [], []
    for c, m in M.items():
        if mk not in m or c not in cl:
            continue
        r = cl[c]
        clk = time_of_day(r)
        if clk.get("status") != "ok":
            continue
        x = float(r[ev])
        first = m.get(mk.split("_")[0] + "_first_record")
        if mk.endswith("_start") and first is not None and m[mk] - first < 30:
            continue
        minute = (clk["tod_s"] + x) / 60
        if abs(minute - round(minute)) > 1e-3:
            continue
        d = m[mk] - x
        if abs(d) > 3600:
            continue
        (g5 if int(round(minute)) % 5 == 0 else g1).append(d)
    return np.array(g5), np.array(g1)


LABEL = {"ppf_start": "propofol infusion start", "rftn_start": "remifentanil start", "bis_loc": "BIS below 60",
         "vent_start": "ventilation start", "co2_start": "ETCO2 start", "ppf_stop": "propofol infusion stop",
         "rftn_stop": "remifentanil stop", "vent_stop": "ventilation stop", "co2_stop": "ETCO2 stop",
         "bis_wake": "BIS above 80", "rec_end": "recording end"}


def fig(paths, root, out, example=("aneend", "co2_stop")):
    """Left: one marker's D for both groups, with the 5-minute group predicted under each rounding kernel.
    Right: the shift of the 5-minute group against the minute group for every marker (bootstrap 95% CI)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vitaldb_bench.case_clock import time_of_day  # noqa: PLC0415

    M = {}
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            if "error" not in r:
                M[r["case"]] = r
    cl = {int(r["caseid"]): r for r in csv.DictReader(open(Path(root) / "clinical_data.csv", encoding="utf-8-sig"))}
    rng = np.random.default_rng(0)
    f, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ev, mk = example
    g5, g1 = groups(M, cl, ev, mk, time_of_day)
    both = np.r_[g1, g5]
    lo_, hi_ = np.floor(np.percentile(both, 1) / 60) * 60 - 300, np.ceil(np.percentile(both, 99) / 60) * 60 + 300
    bins = np.arange(lo_, hi_ + 1, 60)
    ax[0].hist(g1, bins=bins, density=True, color="#bdbdbd", label=f"minute entries ({g1.size:,})")
    ax[0].hist(g5, bins=bins, density=True, histtype="step", lw=2, color="#08519c", label=f"5-minute entries ({g5.size:,})")
    centres = (bins[:-1] + bins[1:]) / 2
    for name, (a, b), col, ls in (("rounded up", (-300, 0), "#d62728", "-"), ("rounded to nearest", (-150, 150), "#636363", "--"),
                                   ("rounded down", (0, 300), "#636363", ":")):
        sim = rng.choice(g1, 200000) + rng.uniform(a, b, 200000)
        h, _ = np.histogram(sim, bins=bins, density=True)
        ax[0].plot(centres, h, color=col, ls=ls, lw=1.4, label=f"predicted if {name}")
    ax[0].set(xlabel=f"{LABEL[mk]} minus the EMR {'anaesthesia end' if ev == 'aneend' else 'anaesthesia start'} (s)",
              ylabel="share of cases (per s)", xlim=(lo_, hi_))
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    rows = []
    for ev2, markers in (("anestart", INDUCTION), ("aneend", EMERGENCE)):
        for m2 in markers:
            a5, a1 = groups(M, cl, ev2, m2, time_of_day)
            if a5.size < 20 or a1.size < 20:
                continue
            bs = [np.median(rng.choice(a5, a5.size)) - np.median(rng.choice(a1, a1.size)) for _ in range(1000)]
            rows.append((f"{'start' if ev2 == 'anestart' else 'end'}: {LABEL[m2]}", np.median(a5) - np.median(a1),
                         *np.percentile(bs, [2.5, 97.5])))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, med, lo, hi) in zip(y, rows):
        ax[1].plot([lo, hi], [yy, yy], color="#08519c", lw=2)
        ax[1].plot(med, yy, "o", color="#08519c", ms=5)
    for x, lab, col in ((-150, "rounded up", "#d62728"), (0, "nearest", "#636363"), (150, "down", "#636363")):
        ax[1].axvline(x, color=col, lw=.9, ls="--")
        ax[1].text(x, len(rows) - .3, lab, ha="center", fontsize=7.5, color=col)
    ax[1].set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax[1].set(xlabel="5-minute minus minute entries: shift of the median (s, 95% CI)", xlim=(-400, 200),
              ylim=(-.7, len(rows) + .2))
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    f.tight_layout()
    f.savefig(out, dpi=150)
    print(json.dumps({"example": {"n5": int(g5.size), "n1": int(g1.size)}, "markers": len(rows),
                      "shifts": [[r[0], round(float(r[1]), 0)] for r in rows]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan", "agg", "fig"])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--out")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.mode == "agg":
        agg(a.inputs, a.root, a.summary)
        return
    if a.mode == "fig":
        fig(a.inputs, a.root, a.out)
        return
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in open(a.out)}
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        try:
            r = scan_case(a.root, c)
        except Exception as e:  # noqa: BLE001
            r = {"case": c, "error": repr(e)[:200]}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
