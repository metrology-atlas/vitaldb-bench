# SPDX-License-Identifier: MIT
"""Worked example: post-induction hypotension, from the charted anaesthesia start and from the devices.

    python examples/induction_hypotension.py scan --root R --cases 1,2,3 --out part.jsonl [--time-budget 170]
    python examples/induction_hypotension.py agg part*.jsonl --root R --out DIR

Post-induction hypotension is usually defined from the charted anaesthesia start: MAP < 65 mmHg within the first
20 minutes. Anaesthesia start entered on a 5-minute mark is rounded up, and written a little late
([[study:emr-rounding]]), so the window opens after induction. Per case:

  induction   from the devices (tools/emr_markers.py): BIS loss of consciousness (BIS < 60 held 30 s after BIS >= 80),
              else the first propofol pump rate; kept when within -20 to +40 min of the charted start;
  pressure    cuff readings (Solar8000 NIBP_MBP, a new value or a value after a gap) and arterial-line dips (ART_MBP
              numeric < 65 for 60 s or more, start times), from 30 min before to 60 min after the charted start;
  windows     20 minutes from the charted start, and 20 minutes from the device induction; a sweep of window
              starts from -40 to +10 min around the device induction.
Cases are split, as in the EMR rounding study, into 5-minute entries and minute entries by the recovered case clock
(vitaldb_bench.case_clock). scan writes one JSON line per case (stays on this machine); agg writes DIR/summary.json
(aggregates only) and DIR/induction_hypotension.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emr_markers  # noqa: E402

warnings.filterwarnings("ignore")
LIM, WIN = 65.0, 1200.0
SHIFTS = np.arange(-2400, 601, 30)


def scan_case(root, c, anestart):
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
    m = emr_markers.scan_case(root, c)
    v = read_vital(Path(root) / "vital_files" / f"{c:04d}.vital", track_names={"Solar8000/NIBP_MBP", "Solar8000/ART_MBP"})
    lo, hi = anestart - 1800, anestart + 3600
    out = {"case": c, "markers": {k: m[k] for k in ("bis_loc", "ppf_start", "vent_start", "bis_first_record", "ppf_first_record") if k in m}}
    s = emr_markers.series(v, "Solar8000/NIBP_MBP")
    if s is not None:
        t, x = s
        new = np.r_[True, (np.diff(x) != 0) | (np.diff(t) > 10)]
        k = new & (t >= lo) & (t <= hi) & (x > 20) & (x < 200)
        out["cuff"] = [[round(float(a), 1), float(b)] for a, b in zip(t[k], x[k])]
    s = emr_markers.series(v, "Solar8000/ART_MBP")
    if s is not None:
        t, x = s
        k = (t >= lo) & (t <= hi)
        t, x = t[k], x[k]
        if t.size > 30:
            ok = (x > 20) & (x < 200)
            below = ok & (x < LIM)
            starts = np.flatnonzero(below & ~np.r_[False, below[:-1]])
            ends = np.flatnonzero(below & ~np.r_[below[1:], False])
            out["art_dips"] = [round(float(t[a]), 1) for a, b in zip(starts, ends) if t[b] - t[a] >= 60]
            out["art_span"] = [round(float(t[ok][0]), 1), round(float(t[ok][-1]), 1)] if ok.any() else None
    return out


def scan(a) -> None:
    clin = {int(r["caseid"]): r for r in csv.DictReader(open(Path(a.root) / "clinical_data.csv", encoding="utf-8-sig"))}
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t0 = time.time()
    with open(out, "a") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            if a.time_budget and time.time() - t0 > a.time_budget:
                break
            t1 = time.time()
            try:
                r = scan_case(a.root, c, float(clin[c]["anestart"]))
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 1)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def hypo(r, t0):
    """(monitored, hypotensive, minutes to the first low reading) in [t0, t0 + 20 min]."""
    cuff = [(t, x) for t, x in r.get("cuff", []) if t0 <= t <= t0 + WIN]
    span = r.get("art_span")
    art_on = span is not None and span[0] <= t0 + 300 and span[1] >= t0 + WIN
    if len(cuff) < 2 and not art_on:
        return False, False, None
    times = [t for t, x in cuff if x < LIM] + ([t for t in r.get("art_dips", []) if t0 <= t <= t0 + WIN] if art_on else [])
    return True, bool(times), (min(times) - t0) / 60 if times else None


def agg(a) -> None:
    from vitaldb_bench.case_clock import time_of_day
    clin = {int(r["caseid"]): r for r in csv.DictReader(open(Path(a.root) / "clinical_data.csv", encoding="utf-8-sig"))}
    rows = [json.loads(l) for p in a.parts for l in Path(p).read_text().splitlines() if l.strip()]
    rows = list({r["case"]: r for r in rows}.values())   # one line per recording (a resumed run may repeat one)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for r in rows:
        if "markers" not in r:
            continue
        cl = clin[r["case"]]
        ane = float(cl["anestart"])
        mk = r["markers"]
        dev = mk.get("bis_loc")
        src = "bis"
        if dev is None or not (-1200 <= dev - ane <= 2400) or dev - mk.get("bis_first_record", -1e9) < 30:
            dev, src = mk.get("ppf_start"), "ppf"
            if dev is None or not (-1200 <= dev - ane <= 2400) or dev - mk.get("ppf_first_record", -1e9) < 30:
                continue
        clk = time_of_day(cl)
        if clk.get("status") != "ok":
            continue
        minute = (clk["tod_s"] + ane) / 60
        on_minute = abs(minute - round(minute)) < 1e-3
        group = ("5-minute" if round(minute) % 5 == 0 else "minute") if on_minute else None
        if group is None:
            continue
        mc, hc, tc = hypo(r, ane)
        md, hd, td = hypo(r, dev)
        if not (mc and md):
            continue
        sweep = [hypo(r, dev + s)[1] if hypo(r, dev + s)[0] else None for s in SHIFTS]
        recs.append({"case": r["case"], "group": group, "src": src, "D": dev - ane, "hc": hc, "hd": hd, "tc": tc, "td": td,
                     "sweep": sweep, "art": r.get("art_span") is not None})
    summ = {"cases_read": len(rows), "cases_used": len(recs),
            "by_group": {}, "sweep_shift_s": SHIFTS.tolist()}
    rng = np.random.default_rng(0)
    for g in ("5-minute", "minute", "all"):
        R = [x for x in recs if g == "all" or x["group"] == g]
        if len(R) < 20:
            continue
        D = np.array([x["D"] for x in R]); hc = np.array([x["hc"] for x in R]); hd = np.array([x["hd"] for x in R])
        boot = [np.mean(hd[i]) - np.mean(hc[i]) for i in (rng.integers(0, len(R), len(R)) for _ in range(1000))]
        tc = np.array([x["tc"] for x in R if x["tc"] is not None]); td = np.array([x["td"] for x in R if x["td"] is not None])
        both = [(x["tc"], x["td"]) for x in R if x["tc"] is not None and x["td"] is not None]
        sw = np.array([[np.nan if s is None else float(s) for s in x["sweep"]] for x in R])
        summ["by_group"][g] = {
            "cases": len(R), "device_minus_charted_s_p10_25_50_75_90": [round(float(np.percentile(D, p))) for p in (10, 25, 50, 75, 90)],
            "incidence_charted": round(float(hc.mean()), 4), "incidence_device": round(float(hd.mean()), 4),
            "incidence_difference": round(float(hd.mean() - hc.mean()), 4),
            "incidence_difference_ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
            "relative_difference": round(float(hd.mean() / hc.mean() - 1), 3) if hc.mean() > 0 else None,
            "labels_flip": round(float(np.mean(hc != hd)), 4), "flip_to_hypotensive": int((hd & ~hc).sum()), "flip_to_not": int((hc & ~hd).sum()),
            "minutes_to_first_low_charted_median": round(float(np.median(tc)), 2) if tc.size else None,
            "minutes_to_first_low_device_median": round(float(np.median(td)), 2) if td.size else None,
            "minutes_to_first_low_diff_median": round(float(np.median([b - a for a, b in both])), 2) if both else None,
            "sweep_incidence": np.round(np.nanmean(sw, 0), 4).tolist(),
            "induction_source_bis_share": round(float(np.mean([x["src"] == "bis" for x in R])), 3),
            "with_arterial_line_share": round(float(np.mean([x["art"] for x in R])), 3)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    figure(recs, summ, out)


def figure(recs, summ, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"5-minute": "#e65100", "minute": "#3f51b5"}
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for g in ("minute", "5-minute"):
        D = np.array([x["D"] for x in recs if x["group"] == g]) / 60
        if D.size:
            ax[0].hist(np.clip(-D, -40, 10), bins=np.arange(-40, 10.5, 1), histtype="step", lw=1.3, color=col[g], density=True,
                       label=f"{g} entries ({D.size:,})")
    ax[0].axvline(0, color="0.4", lw=0.8)
    ax[0].set_xlabel("charted start after device induction (min)")
    ax[0].set_yticks([]); ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title("A. Where the charted start sits", fontsize=9, loc="left")
    for g in ("minute", "5-minute", "all"):
        if g in summ["by_group"]:
            ax[1].plot(np.array(summ["sweep_shift_s"]) / 60, np.array(summ["by_group"][g]["sweep_incidence"]) * 100,
                       color=col.get(g, "0.3"), lw=1.3 if g != "all" else 0.8, ls="-" if g != "all" else "--", label=g)
    ax[1].axvline(0, color="0.4", lw=0.8)
    med = -summ["by_group"]["all"]["device_minus_charted_s_p10_25_50_75_90"][2] / 60
    ax[1].axvline(med, color="0.6", lw=0.8, ls=":")
    ax[1].annotate("median charted start", (med, ax[1].get_ylim()[0]), xytext=(4, 6), textcoords="offset points", fontsize=7)
    ax[1].set_xlabel("20-min window opens this long after device induction (min)")
    ax[1].set_ylabel("post-induction hypotension (%)")
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_title("B. Incidence depends on where the window opens", fontsize=9, loc="left")
    gs = [g for g in ("minute", "5-minute") if g in summ["by_group"]]
    x = np.arange(len(gs))
    ax[2].bar(x - 0.2, [summ["by_group"][g]["incidence_charted"] * 100 for g in gs], 0.4, color="0.6", label="from charted start")
    ax[2].bar(x + 0.2, [summ["by_group"][g]["incidence_device"] * 100 for g in gs], 0.4, color="#3f51b5", label="from device induction")
    for xi, g in zip(x, gs):
        ax[2].text(xi, max(summ["by_group"][g]["incidence_charted"], summ["by_group"][g]["incidence_device"]) * 100 + 1,
                   f"{summ['by_group'][g]['labels_flip']:.0%} of labels flip", ha="center", fontsize=7)
    ax[2].set_xticks(x, [f"{g} entries" for g in gs])
    ax[2].set_ylabel("post-induction hypotension (%)")
    ax[2].legend(fontsize=7, frameon=True, loc="lower center", framealpha=0.9)
    ax[2].set_title("C. Cohort and patient", fontsize=9, loc="left")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Post-induction hypotension (MAP < 65 within 20 min): {summ['cases_used']:,} cases", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "induction_hypotension.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    for k in ("--root", "--cases", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--time-budget", type=float, default=0)
    g = sub.add_parser("agg")
    g.add_argument("parts", nargs="+")
    g.add_argument("--root", required=True)
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
