# SPDX-License-Identifier: MIT
"""Temporal resolution of every kind of time stamp in VitalDB. Aggregates only leave this machine.

    python tools/timestamp_resolution.py tables --root R --out tables.json
    python tools/timestamp_resolution.py tracks --root R --cases 1,2 --out part.json      # per-track stamps
    python tools/timestamp_resolution.py merge part*.json --out tracks.json
    python tools/timestamp_resolution.py cautery --root R --cases 1,2 --out cautery.jsonl  # per case, stays here
    python tools/timestamp_resolution.py rounding --root R --cautery cautery.jsonl --out rounding.json

tables:  clinical_data.csv and lab_data.csv. For each time column: integer fraction and the fraction on
         60/300/3600/86400 s multiples. EMR event times (anestart, aneend, opstart, opend): the per-case
         seconds phase (all four on one wall-clock minute grid?), the largest common grid of their
         differences, and - with the case's time of day from adm (admission date at midnight, relative to
         casestart truncated to the minute) - whether each falls on an absolute 5-minute mark. adm/dis: whole
         days? Lab dt: integer seconds? on the case's minute grid?
tracks:  every track of every file: type, sample rate, record sizes; record stamps on a 1 ms / 10 ms grid;
         record interval (log histogram, 5 ms jitter histogram near the mode); interval between value
         changes; value step resolution; duplicate stamps; EVENT strings and their stamps.
cautery: SNUADC ECG 30-60 Hz bursts (tools/shared_artifacts.py envelope); the first and the last sustained
         burst (>= 10 bursts in the following / preceding 5 min) - an independent clock for incision and
         closure.
rounding: direction of the EMR rounding. Per event, cases where it falls on an absolute 5-min mark (mostly
         5-min entries) against cases where it is on the minute grid but off a 5-min mark (minute entries), each
         measured against a reference with its own clock: anestart vs recording start, aneend vs recording
         end, opstart vs first sustained cautery, opend vs last sustained cautery. Floor rounding makes the
         5-min group earlier by ~120 s (mean of U[0,300) less U[0,60)), ceiling later by ~120 s, nearest ~0.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ORIGIN = 4102444800.0          # 2100-01-01 00:00:00 UTC: every file's casestart
EMR = ["anestart", "aneend", "opstart", "opend"]
LOG_EDGES = np.logspace(-3, np.log10(86400), 321)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def on_grid(v, g, tol=1e-3):
    v = np.asarray(v, float)
    return np.abs(v / g - np.round(v / g)) * g < tol


# ------------------------------------------------------------------------------------------------ tables

def case_clock(r):
    """(time of day of casestart truncated to the minute, seconds past that minute) or None."""
    v = [fnum(r[k]) for k in EMR]
    adm = fnum(r["adm"])
    if any(x is None for x in v) or adm is None:
        return None
    ph = Counter(round(x % 60, 3) for x in v)
    p, n = ph.most_common(1)[0]
    if n < 3:
        return None
    s = (60 - p) % 60
    return ((-adm) % 86400, s)


def abs_minute(r, k, clk):
    """Wall-clock minute (of the day) of EMR event k, or None if off the case's minute grid."""
    tod, s = clk
    x = fnum(r[k])
    t = tod + s + x
    if abs(t / 60 - round(t / 60)) > 1e-3:
        return None
    return int(round(t / 60)) % 1440


def tables(root):
    root = Path(root)
    cl = list(csv.DictReader(open(root / "clinical_data.csv", encoding="utf-8-sig")))
    out = {"cases": len(cl), "columns": {}}
    for c in ["casestart", "caseend"] + EMR + ["adm", "dis"]:
        v = np.array([x for x in (fnum(r[c]) for r in cl) if x is not None])
        out["columns"][c] = {"n": int(v.size), "integer": round(float(on_grid(v, 1, 1e-4).mean()), 4),
                             **{f"on_{g}s": round(float(on_grid(v, g).mean()), 4) for g in (60, 300, 3600, 86400)}}
    # per-case phase, common grid of differences
    same, grid, ph = 0, Counter(), []
    off = Counter()
    nn = 0
    for r in cl:
        v = [fnum(r[k]) for k in EMR]
        if any(x is None for x in v):
            continue
        nn += 1
        m = Counter(round(x % 60, 3) for x in v)
        if len(m) == 1:
            same += 1
            ph.append(v[0] % 60)
        else:
            maj, cnt = m.most_common(1)[0]
            for k, x in zip(EMR, v):
                if cnt >= 2 and round(x % 60, 3) != maj:
                    off[k] += 1
        d = [v[i] - v[0] for i in range(1, 4)]
        gs = [g for g in (1, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600) if all(on_grid([x], g)[0] for x in d)]
        grid[max(gs) if gs else 0] += 1
    out["emr"] = {"cases": nn, "all_four_same_minute_phase": same, "event_off_phase": off.most_common(),
                  "largest_common_grid_of_differences": sorted(grid.items()),
                  "phase_hist_10s": np.histogram(ph, bins=6, range=(0, 60))[0].tolist()}
    # absolute 5-min marks, per event, by the casestart minute convention
    abs5 = {}
    for mode in ("floor", "nearest", "ceil"):
        res = {k: [0, 0] for k in EMR}
        for r in cl:
            clk = case_clock(r)
            if clk is None:
                continue
            tod, s = clk
            base = {"floor": tod + s, "nearest": tod + s - 60 * (s >= 30), "ceil": tod + s - 60 * (s > 0)}[mode]
            for k in EMR:
                x = fnum(r[k])
                t = base + x
                if abs(t / 60 - round(t / 60)) > 1e-3:
                    continue
                res[k][0] += 1
                res[k][1] += int(round(t / 60)) % 5 == 0
        abs5[mode] = {k: round(b / a, 3) if a else None for k, (a, b) in res.items()}
    out["emr_on_absolute_5min_mark"] = abs5
    mh = {k: Counter() for k in EMR}
    for r in cl:
        clk = case_clock(r)
        if clk is None:
            continue
        for k in EMR:
            m = abs_minute(r, k, clk)
            if m is not None:
                mh[k][m % 60] += 1
    out["emr_minute_of_hour"] = {k: [mh[k][i] for i in range(60)] for k in EMR}
    tod = np.array([(-fnum(r["adm"])) % 86400 / 3600 for r in cl if fnum(r["adm"]) is not None])
    dd = np.array([fnum(r["dis"]) - fnum(r["adm"]) for r in cl if fnum(r["dis"]) is not None and fnum(r["adm"]) is not None])
    out["adm_dis"] = {"dis_minus_adm_whole_days": round(float(on_grid(dd, 86400).mean()), 4),
                      "casestart_hour_of_day_hist": np.histogram(tod, bins=24, range=(0, 24))[0].tolist()}
    # lab
    clk = {int(r["caseid"]): case_clock(r) for r in cl}
    tot = integ = on60 = onphase = 0
    for r in csv.DictReader(open(root / "lab_data.csv", encoding="utf-8-sig")):
        dt = fnum(r["dt"])
        if dt is None:
            continue
        tot += 1
        integ += abs(dt - round(dt)) < 1e-4
        on60 += abs(dt / 60 - round(dt / 60)) < 1e-4
        k = clk.get(int(r["caseid"]))
        if k is not None:
            onphase += abs(((dt + k[1]) / 60) - round((dt + k[1]) / 60)) < 1e-4
    out["lab_dt"] = {"rows": tot, "integer": round(integ / tot, 4), "on_60s_from_casestart": round(on60 / tot, 4),
                     "on_wall_clock_minute": round(onphase / tot, 4)}
    return out


# ------------------------------------------------------------------------------------------------ tracks

def empty_track():
    return {"cases": 0, "type": Counter(), "srate": Counter(), "sizes": Counter(), "records": 0, "on1ms": 0,
            "on10ms": 0, "on1s": 0, "dup": 0, "rec": np.zeros(LOG_EDGES.size - 1, np.int64),
            "chg": np.zeros(LOG_EDGES.size - 1, np.int64), "step": Counter(), "case_median_rec": []}


def scan_tracks(root, cases):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    A = defaultdict(empty_track)
    ev_labels, ev_res = Counter(), defaultdict(Counter)
    for c in cases:
        try:
            v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital")
        except Exception:
            continue
        for n, tr in v.tracks.items():
            t = np.asarray(tr.record_times, float)
            if not t.size:
                continue
            a = A[n]
            a["cases"] += 1
            a["type"][int(tr.track_type)] += 1
            a["srate"][float(tr.srate)] += 1
            a["sizes"].update(Counter(int(s) for s in np.asarray(tr.record_sizes)[:5000]))
            a["records"] += int(t.size)
            rel = t - ORIGIN
            a["on1ms"] += int(on_grid(rel, 0.001, 2e-5).sum())
            a["on10ms"] += int(on_grid(rel, 0.01, 2e-5).sum())
            a["on1s"] += int(on_grid(rel, 1.0, 2e-5).sum())
            o = np.sort(t)
            d = np.diff(o)
            a["dup"] += int((d < 0.005).sum())
            dd = d[(d >= 0.005) & (d < 3600)]
            a["rec"] += np.histogram(dd, LOG_EDGES)[0]
            if dd.size:
                a["case_median_rec"].append(float(np.median(dd)))
            if tr.track_type == 2:
                x = np.asarray(tr.physical(), float)[: t.size]
                oo = np.argsort(t, kind="stable")
                tt, xx = t[oo], x[oo]
                k = np.r_[True, np.diff(tt) >= 0.005]
                tt, xx = tt[k], xx[k]
                ch = np.flatnonzero(np.r_[False, np.diff(xx) != 0])
                if ch.size > 1:
                    a["chg"] += np.histogram(np.diff(tt[ch]), LOG_EDGES)[0]
                s = np.abs(np.diff(xx))
                s = s[(s > 0) & np.isfinite(s)]
                a["step"].update(Counter(np.round(s, 4).tolist()[:20000]))
            if n == "EVENT":
                vals = tr.values if not hasattr(tr, "physical") else None
                try:
                    vals = list(tr.physical())
                except Exception:
                    vals = list(getattr(tr, "values", []))
                for tv, lab in zip(rel, vals):
                    lab = str(lab)[:40]
                    ev_labels[lab] += 1
                    ev_res[lab]["int_s" if abs(tv - round(tv)) < 1e-4 else "frac_s"] += 1
    out = {"tracks": {}, "event_labels": ev_labels.most_common(40),
           "event_stamp_kind": {k: dict(v) for k, v in ev_res.items() if ev_labels[k] >= 5}}
    for n, a in A.items():
        out["tracks"][n] = {"cases": a["cases"], "type": dict(a["type"]), "srate": {str(k): v for k, v in a["srate"].items()},
                            "sizes": dict(a["sizes"].most_common(5)), "records": a["records"], "on1ms": a["on1ms"],
                            "on10ms": a["on10ms"], "on1s": a["on1s"], "dup": a["dup"], "rec": a["rec"].tolist(),
                            "chg": a["chg"].tolist(), "step": {str(k): v for k, v in a["step"].most_common(8)},
                            "case_median_rec": a["case_median_rec"]}
    return out


def merge(paths):
    M = None
    for p in paths:
        J = json.load(open(p))
        if M is None:
            M = J
            continue
        for n, a in J["tracks"].items():
            if n not in M["tracks"]:
                M["tracks"][n] = a
                continue
            m = M["tracks"][n]
            for k in ("cases", "records", "on1ms", "on10ms", "on1s", "dup"):
                m[k] += a[k]
            for k in ("rec", "chg"):
                m[k] = (np.asarray(m[k]) + np.asarray(a[k])).tolist()
            for k in ("type", "srate", "sizes", "step"):
                c = Counter(m[k])
                c.update(a[k])
                m[k] = dict(c)
            m["case_median_rec"] += a["case_median_rec"]
        ev = Counter(dict(M["event_labels"]))
        ev.update(dict(J["event_labels"]))
        M["event_labels"] = ev.most_common(40)
        for k, v in J["event_stamp_kind"].items():
            c = Counter(M["event_stamp_kind"].get(k, {}))
            c.update(v)
            M["event_stamp_kind"][k] = dict(c)
    return M


def summarise_tracks(M):
    E = np.asarray(LOG_EDGES)
    C = np.sqrt(E[:-1] * E[1:])

    def q(h, p):
        h = np.asarray(h, float)
        if h.sum() == 0:
            return None
        cs = np.cumsum(h) / h.sum()
        return [round(float(C[min(np.searchsorted(cs, x), C.size - 1)]), 3) for x in p]

    rows = {}
    for n, a in sorted(M["tracks"].items()):
        r = a["records"] or 1
        rows[n] = {"cases": a["cases"], "type": a["type"], "srate": a["srate"], "sizes": a["sizes"],
                   "stamp_on_1ms": round(a["on1ms"] / r, 3), "stamp_on_10ms": round(a["on10ms"] / r, 3),
                   "stamp_on_1s": round(a["on1s"] / r, 3), "dup_frac": round(a["dup"] / r, 3),
                   "record_interval_p5_50_95": q(a["rec"], (0.05, 0.5, 0.95)),
                   "change_interval_p10_50_90": q(a["chg"], (0.1, 0.5, 0.9)),
                   "case_median_rec_p10_90": [round(float(x), 3) for x in np.percentile(a["case_median_rec"], [10, 90])]
                   if a["case_median_rec"] else None,
                   "value_steps": dict(list(a["step"].items())[:4])}
    return rows


# ------------------------------------------------------------------------------------------------ cautery

def cautery(root, cases, out):
    sys.path.insert(0, str(Path(__file__).parent))
    from shared_artifacts import envelope  # noqa: E402
    from pat import sample_times  # noqa: E402
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    done = set()
    if Path(out).exists():
        done = {json.loads(l)["case"] for l in open(out)}
    for c in cases:
        if c in done:
            continue
        res = {"case": c}
        try:
            v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names={"SNUADC/ECG_II", "SNUADC/ECG_V5"})
            tr = v.tracks.get("SNUADC/ECG_II") or v.tracks.get("SNUADC/ECG_V5")
            if tr is None or not tr.record_times.size:
                res["skip"] = "no ECG"
            else:
                t = sample_times(tr)
                x = tr.physical().astype(np.float64)[: t.size]
                ok = np.isfinite(x)
                x = np.where(ok, x, 0.0)
                n = int((t[-1] - ORIGIN) / 0.05) + 1
                e = envelope(t, x, float(tr.srate), (30, 60), ORIGIN, n)
                m = np.nan_to_num(e, nan=0) > 1.0
                dm = np.diff(np.r_[0, m.astype(int), 0])
                st, en = np.flatnonzero(dm == 1), np.flatnonzero(dm == -1)
                keep = (en - st) >= 6
                on = st[keep] * 0.05
                res["bursts"] = int(on.size)
                if on.size >= 10:
                    nxt = np.searchsorted(on, on + 300) - np.arange(on.size)
                    prv = np.arange(on.size) + 1 - np.searchsorted(on, on - 300)
                    first = on[np.flatnonzero(nxt >= 10)]
                    last = on[np.flatnonzero(prv >= 10)]
                    res["first_sustained"] = round(float(first[0]), 2) if first.size else None
                    res["last_sustained"] = round(float(last[-1]), 2) if last.size else None
                res["rec_end"] = round(float(t[-1] - ORIGIN), 2)
        except Exception as ex:  # noqa: BLE001
            res["error"] = repr(ex)[:200]
        with open(out, "a") as fh:
            fh.write(json.dumps(res) + "\n")


def rounding(root, caut_path):
    root = Path(root)
    cl = {int(r["caseid"]): r for r in csv.DictReader(open(root / "clinical_data.csv", encoding="utf-8-sig"))}
    CA = {}
    if caut_path:
        for l in open(caut_path):
            r = json.loads(l)
            CA[r["case"]] = r
    out = {}
    for k in EMR:
        g5, g1 = [], []
        for c, r in cl.items():
            clk = case_clock(r)
            if clk is None:
                continue
            mk = abs_minute(r, k, clk)
            if mk is None:
                continue
            # on an absolute 5-min mark: mostly 5-min entries (plus the 1 in 5 minute entries that land there);
            # off a mark: certainly a minute entry
            grp = "5" if mk % 5 == 0 else "1"
            x = fnum(r[k])
            if k == "anestart":
                ref = 0.0
            elif k == "aneend":
                ref = fnum(r["caseend"])
            else:
                a = CA.get(c)
                if not a:
                    continue
                ref = a.get("first_sustained") if k == "opstart" else a.get("last_sustained")
                if ref is None:
                    continue
            d = x - ref
            if abs(d) > 3600:
                continue
            (g5 if grp == "5" else g1).append(d)
        g5, g1 = np.array(g5), np.array(g1)
        if g5.size < 20 or g1.size < 20:
            out[k] = {"n5": int(g5.size), "n1": int(g1.size)}
            continue
        rng = np.random.default_rng(0)
        bs = [np.median(rng.choice(g5, g5.size)) - np.median(rng.choice(g1, g1.size)) for _ in range(2000)]
        bm = [np.mean(rng.choice(g5, g5.size)) - np.mean(rng.choice(g1, g1.size)) for _ in range(2000)]
        out[k] = {"reference": {"anestart": "recording start", "aneend": "recording end (caseend)",
                                "opstart": "first sustained cautery", "opend": "last sustained cautery"}[k],
                  "n5": int(g5.size), "n1": int(g1.size),
                  "event_minus_ref_5min_p10_50_90": np.percentile(g5, [10, 50, 90]).round(0).tolist(),
                  "event_minus_ref_1min_p10_50_90": np.percentile(g1, [10, 50, 90]).round(0).tolist(),
                  "shift_median_s": round(float(np.median(g5) - np.median(g1)), 1),
                  "shift_median_ci95": np.percentile(bs, [2.5, 97.5]).round(1).tolist(),
                  "shift_mean_s": round(float(g5.mean() - g1.mean()), 1),
                  "shift_mean_ci95": np.percentile(bm, [2.5, 97.5]).round(1).tolist(),
                  "expected": {"floor": -120, "nearest": 0, "ceil": 120}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tables", "tracks", "merge", "summary", "cautery", "rounding"])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--root")
    ap.add_argument("--cases")
    ap.add_argument("--cautery")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cases = [int(x) for x in a.cases.split(",") if x] if a.cases else []
    if a.mode == "tables":
        J = tables(a.root)
    elif a.mode == "tracks":
        J = scan_tracks(a.root, cases)
    elif a.mode == "merge":
        J = merge(a.inputs)
    elif a.mode == "summary":
        M = json.load(open(a.inputs[0]))
        J = {"tracks": summarise_tracks(M), "event_labels": M["event_labels"], "event_stamp_kind": M["event_stamp_kind"]}
    elif a.mode == "cautery":
        cautery(a.root, cases, a.out)
        return
    else:
        J = rounding(a.root, a.cautery)
    json.dump(J, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
