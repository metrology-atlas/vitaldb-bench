# SPDX-License-Identifier: MIT
"""Aggregate the numeric tracks in the record cache (tools/record_scan.py): intervals, gaps, frozen runs, value grids.
This machine only.

    python tools/numeric_agg.py parts --records DIR --out AGG [--time-budget S]   # resumable, 400 cases per part
    python tools/numeric_agg.py merge --out AGG                                   # numerics.pkl, numerics.csv
    python tools/numeric_agg.py figures --out AGG

Per track (re-declared '#2' tracks folded into the original): recordings, records, every interval between stamps
(1 ms), every gap (an interval over twice the track's median interval in that recording), every run of an unchanged
value over 3 or more records (duration, records, and the value for runs of 60 s or more), and every distinct value
with its count.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def new():
    return dict(cases=0, records=0, dup=0, iv=Counter(), gap_d=[], gap_rel=[], run_d=[], run_n=[], run_long_v=Counter(),
                vals=Counter(), med=[], unit=Counter(), fmt=Counter(), device=Counter(), disp=Counter())


def parts(a):
    rec, out = Path(a.records), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((f for f in rec.glob("*.npz") if f.stem.isdigit()), key=lambda f: int(f.stem))
    t0 = time.time()
    for p in range(0, len(files), 400):
        dst = out / f"part_{p // 400:02d}.pkl"
        if dst.exists():
            continue
        acc = defaultdict(new)
        for f in files[p:p + 400]:
            try:
                z = np.load(f, allow_pickle=False)
            except Exception:  # noqa: BLE001
                continue
            keys = set(z.files)
            for k in keys:
                if not k.endswith("|iv_ms"):
                    continue
                tn = k[:-6]
                m = json.loads(str(z[tn + "|meta"]))
                A = acc[tn.replace("__", "/").split("#")[0]]
                A["cases"] += 1
                A["records"] += m.get("records", 0)
                A["dup"] += m.get("duplicate_stamps", 0)
                med = m.get("median_interval")
                A["med"].append(med)
                A["unit"][m.get("unit")] += 1
                A["fmt"][m.get("format")] += 1
                A["device"][(m.get("device") or [None])[0]] += 1
                A["disp"][(m.get("mindisp"), m.get("maxdisp"))] += 1
                for v, n in zip(z[k], z[tn + "|iv_n"]):
                    A["iv"][int(v)] += int(n)
                if tn + "|gap_d" in keys:
                    g = z[tn + "|gap_d"]
                    A["gap_d"].append(g.astype(np.float32))
                    if med:
                        A["gap_rel"].append((g / med).astype(np.float32))
                if tn + "|run_d" in keys:
                    d, n, v = z[tn + "|run_d"], z[tn + "|run_n"], z[tn + "|run_v"]
                    A["run_d"].append(d.astype(np.float32)); A["run_n"].append(n.astype(np.int32))
                    for vv in v[d >= 60]:
                        A["run_long_v"][float(vv)] += 1
                if tn + "|val_v" in keys:
                    for v, n in zip(z[tn + "|val_v"], z[tn + "|val_n"]):
                        A["vals"][float(v)] += int(n)
        for A in acc.values():
            for key in ("gap_d", "gap_rel", "run_d", "run_n"):
                A[key] = np.concatenate(A[key]) if A[key] else np.zeros(0)
        tmp = dst.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(dict(acc), fh)
        tmp.rename(dst)
        print(dst.name, len(acc), round(time.time() - t0), flush=True)
        if a.time_budget and time.time() - t0 > a.time_budget:
            print("time budget reached; rerun to continue")
            return


RUN_BINS = np.logspace(-1, 5, 121)


def merge(a):
    """Streams the parts: run arrays (hundreds of millions of runs) are reduced to counts and a log histogram."""
    out = Path(a.out)
    tot = {}
    for f in sorted(out.glob("part_*.pkl")):
        for tn, A in pickle.load(open(f, "rb")).items():
            T = tot.setdefault(tn, dict(new(), run_hist=np.zeros(len(RUN_BINS) - 1, np.int64), runs=0, runs60=0,
                                        runs600=0, frozen_s=0.0, gap_d=[], gap_rel=[]))
            for key in ("cases", "records", "dup"):
                T[key] += A[key]
            for key in ("iv", "run_long_v", "vals", "unit", "fmt", "device", "disp"):
                T[key].update(A[key])
            T["med"].extend(A["med"])
            T["gap_d"].append(A["gap_d"]); T["gap_rel"].append(A["gap_rel"])
            rd = A["run_d"]
            T["run_hist"] += np.histogram(rd, RUN_BINS)[0]
            T["runs"] += rd.size; T["runs60"] += int((rd >= 60).sum()); T["runs600"] += int((rd >= 600).sum())
            T["frozen_s"] += float(rd[rd >= 60].sum())
    for T in tot.values():
        T["gap_d"] = np.concatenate(T["gap_d"]); T["gap_rel"] = np.concatenate(T["gap_rel"])
        T.pop("run_d", None); T.pop("run_n", None)
    rows = []
    for tn, T in sorted(tot.items(), key=lambda kv: -kv[1]["cases"]):
        iv = T["iv"]; ntot = sum(iv.values()) or 1
        mode = max(iv, key=iv.get) if iv else None
        top = ", ".join(f"{k}:{v / ntot:.2f}" for k, v in iv.most_common(4))
        vals = np.array(sorted(T["vals"]))
        fin = vals[np.isfinite(vals)]
        steps = np.diff(fin)
        res = float(np.min(steps[steps > 0])) if steps.size and (steps > 0).any() else np.nan
        cnt = np.array([T["vals"][v] for v in fin], float)

        def on(g):
            return float(cnt[np.abs(fin / g - np.round(fin / g)) < 1e-3].sum() / max(cnt.sum(), 1))
        hours = sum(k * v for k, v in iv.items()) / 3.6e6
        rows.append(dict(track=tn, device=T["device"].most_common(1)[0][0], unit=T["unit"].most_common(1)[0][0],
                         format=T["fmt"].most_common(1)[0][0], cases=T["cases"], records=T["records"],
                         hours=round(hours), median_interval_s=float(np.median([m for m in T["med"] if m])) if any(T["med"]) else np.nan,
                         interval_mode_ms=mode, interval_top=top, dup_stamps=T["dup"],
                         gaps=len(T["gap_d"]), gaps_per_100h=round(100 * len(T["gap_d"]) / max(hours, 1e-9), 1),
                         gap_med_s=round(float(np.median(T["gap_d"])), 2) if len(T["gap_d"]) else np.nan,
                         runs_ge_60s=T["runs60"], runs_ge_600s=T["runs600"], frozen_h=round(T["frozen_s"] / 3600, 1),
                         distinct=len(fin), min_step=res, grid1=round(on(1), 3), grid01=round(on(0.1), 3),
                         grid001=round(on(0.01), 3), long_run_values=", ".join(f"{k:g}:{v}" for k, v in T["run_long_v"].most_common(3))))
    # the distinct-value counters are too large to keep together (Orchestra volumes); keep the 2,000 commonest
    for T in tot.values():
        T["vals"] = Counter(dict(T["vals"].most_common(2000)))
    pickle.dump(tot, open(out / "numerics.pkl", "wb"))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "numerics.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
    print(df[df.cases >= 50].drop(columns=["format", "interval_top", "long_run_values"]).to_string(index=False))


def figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(a.out)
    tot = pickle.load(open(out / "numerics.pkl", "rb"))
    tracks = ["Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/ART_SBP", "Solar8000/NIBP_SBP", "Solar8000/ST_II",
              "Primus/ETCO2", "BIS/BIS", "Orchestra/RFTN20_RATE", "EV1000/CO"]
    tracks = [t for t in tracks if t in tot]
    fig, axs = plt.subplots(len(tracks), 3, figsize=(15, 2.3 * len(tracks)))
    for r, t in enumerate(tracks):
        T = tot[t]
        iv = T["iv"]; ks = np.array(sorted(iv)); ns = np.array([iv[k] for k in ks], float)
        ax = axs[r, 0]
        ax.bar(ks, ns / ns.sum(), width=1.0, color="#1f5fa6")
        mode = ks[np.argmax(ns)]
        ax.set_xlim(max(-5, mode - 60), mode + 60); ax.set_yscale("log")
        ax.set_title(f"{t}: interval between stamps (1 ms bins)", loc="left", fontsize=9)
        ax = axs[r, 1]
        g = np.sort(T["gap_d"])
        if g.size:
            ax.step(g[::-1], np.arange(1, g.size + 1), where="post", color="#1f5fa6")
            ax.set_xscale("log"); ax.set_yscale("log")
        for x in (60, 3600):
            ax.axvline(x, color="0.7", ls=":", lw=0.8)
        ax.set_title(f"gaps (> 2x median interval) longer than x: {g.size:,}", loc="left", fontsize=9)
        ax = axs[r, 2]
        b = RUN_BINS
        ax.step(b[:-1], T["run_hist"], where="post", color="#e08a1e")
        ax.set_xscale("log"); ax.set_yscale("log")
        for x in (60, 600, 3600):
            ax.axvline(x, color="0.7", ls=":", lw=0.8)
        ax.set_title("runs of an unchanged value (3+ records), by duration", loc="left", fontsize=9)
        for x in axs[r]:
            x.spines[["top", "right"]].set_visible(False)
    axs[-1, 0].set_xlabel("interval (ms)"); axs[-1, 1].set_xlabel("gap (s)"); axs[-1, 2].set_xlabel("run duration (s)")
    fig.tight_layout()
    fig.savefig(out / "numerics_overview.png", dpi=100)


GRIDS = [100, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]


def on_grid(v: np.ndarray, g: float) -> np.ndarray:
    """True where v is a multiple of g as stored in float32 (allowing a few float32 ulps of representation error)."""
    ulp = np.spacing(np.abs(v).astype(np.float32)).astype(float)
    return np.abs(v - np.round(v / g) * g) <= 4 * ulp + 1e-12


def resolution(a):
    """Per numeric track: share of records on each decimal grid; the resolution is the coarsest grid holding 99% of
    records, and the share on the next coarser grid shows heaping or a mixed resolution."""
    out = Path(a.out)
    vals = defaultdict(Counter); cases = Counter(); dev = {}; unit = {}
    for f in sorted(out.glob("part_*.pkl")):
        for tn, A in pickle.load(open(f, "rb")).items():
            vals[tn].update(A["vals"]); cases[tn] += A["cases"]
            dev.setdefault(tn, A["device"].most_common(1)[0][0]); unit.setdefault(tn, A["unit"].most_common(1)[0][0])
    rows = []
    for tn, C in vals.items():
        v = np.array(list(C.keys()), float); n = np.array(list(C.values()), float)
        ok = np.isfinite(v); v, n = v[ok], n[ok]
        tot = n.sum()
        if tot == 0:
            continue
        share = {g: float(n[on_grid(v, g)].sum() / tot) for g in GRIDS}
        fine = [g for g in GRIDS if share[g] >= 0.99]
        res = max(fine) if fine else None            # the coarsest grid that holds 99% of records
        coarse = None
        if res is not None and GRIDS.index(res) > 0:
            coarse = share[GRIDS[GRIDS.index(res) - 1]]   # the next coarser grid: about 10-20% expected by chance
        rows.append(dict(track=tn, device=dev[tn], unit=unit[tn], cases=cases[tn], records=int(tot), distinct=int(v.size),
                         resolution=res, share_at_resolution=round(share[res], 4) if res else None,
                         share_one_step_coarser=round(coarse, 3) if coarse is not None else None,
                         min=float(v.min()), max=float(v.max()),
                         q_lo=float(v[np.argsort(v)][np.searchsorted(np.cumsum(n[np.argsort(v)]) / tot, 1e-5)]),
                         q_hi=float(v[np.argsort(v)][min(v.size - 1, np.searchsorted(np.cumsum(n[np.argsort(v)]) / tot, 1 - 1e-5))]), negative_share=round(float(n[v < 0].sum() / tot), 4),
                         zero_share=round(float(n[v == 0].sum() / tot), 4),
                         **{f"g{g:g}": round(share[g], 3) for g in GRIDS}))
    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["device", "track"])
    df.to_csv(out / "numeric_resolution.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    print(df[df.cases >= 20][["track", "unit", "cases", "distinct", "resolution", "share_at_resolution",
                              "share_one_step_coarser", "min", "max", "zero_share", "g1", "g0.1", "g0.01", "g0.001"]].to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("parts"); c.add_argument("--records", required=True); c.add_argument("--out", required=True)
    c.add_argument("--time-budget", type=float, default=0)
    m = sp.add_parser("merge"); m.add_argument("--out", required=True)
    f = sp.add_parser("figures"); f.add_argument("--out", required=True)
    r = sp.add_parser("resolution"); r.add_argument("--out", required=True)
    a = p.parse_args()
    {"parts": parts, "merge": merge, "figures": figures, "resolution": resolution}[a.cmd](a)


if __name__ == "__main__":
    main()
