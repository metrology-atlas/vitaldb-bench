"""Why the Solar8000 route moved between its 19 Oct run and the stage-3 rerun (11 Nov 2026).

Two checks from saved outputs:

1. HR against mean arterial pressure on the sample-count timeline (tools/solar_snuadc_clock.py agg slopes.csv): the
   two numerics time the same SNUADC clock independently, so their per-recording rates should agree. Median, IQR and
   MAD of HR minus ART_MBP, the median HR standard error and the median |z|, for each run.
2. The loop BIS steps + bursts + solar (tools/clock_edges.py edge table), closed with each route separately: the
   combined solar edge, mean arterial pressure only, HR only (new and old runs). Misclosure per recording =
   bis_steps - bursts - route (x[bis]-x[pc] - (x[bis]-x[snuadc]) - (x[snuadc]-x[pc]); clock_reconcile reports the
   opposite sign).

The rule-swap test (today's tool with the 19 Oct numeric_delay_variability.py reproduces the Oct parts exactly) is
saved beside the output by the caller.

    python examples/solar_route_check.py --old W/solar_snuadc_clock --new W/solar_snuadc_clock_v4 \
        --edges W/clock_network/r6_inputs/edges_r6.csv --out W/clock_network/r6_inputs/solar_route_check.json
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path


def slopes(d):
    out = {}
    for r in csv.DictReader(open(Path(d) / "agg" / "slopes.csv")):
        if r["timeline"] == "counts":
            out.setdefault(int(r["case"]), {})[r["track"]] = (float(r["ppm"]), float(r["se"]))
    return out


def q(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))]


def agreement(S):
    both = [(v["HR"], v["ART_MBP"]) for v in S.values() if "HR" in v and "ART_MBP" in v]
    d = [h[0] - m[0] for h, m in both]
    med = st.median(d)
    return {"recordings": len(both), "hr_minus_mbp_median_ppm": round(med, 1),
            "iqr_ppm": [round(q(d, .25), 1), round(q(d, .75), 1)],
            "mad_ppm": round(st.median(abs(x - med) for x in d), 1),
            "hr_se_median_ppm": round(st.median(h[1] for h, _ in both), 1),
            "abs_z_median": round(st.median(abs(h[0] - m[0]) / (h[1] ** 2 + m[1] ** 2) ** .5 for h, m in both), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    So, Sn = slopes(a.old), slopes(a.new)
    E = {}
    for r in csv.DictReader(open(a.edges)):
        E.setdefault(int(r["recording"]), {})[r["method"]] = float(r["value"])
    loop = [c for c, m in E.items() if {"bis_steps", "bursts", "solar"} <= set(m)]

    def mis(route):
        v = [E[c]["bis_steps"] - E[c]["bursts"] - route(c) for c in loop if route(c) is not None]
        return {"recordings": len(v), "median_ppm": round(st.median(v), 1)}
    res = {"agreement_hr_vs_mbp": {"old": agreement(So), "new": agreement(Sn)},
           "loop_bis_bursts_solar": {"recordings": len(loop),
                                     "combined_solar_new": mis(lambda c: E[c]["solar"]),
                                     "mbp_only_new": mis(lambda c: (Sn.get(c, {}).get("ART_MBP") or [None])[0]),
                                     "mbp_only_old": mis(lambda c: (So.get(c, {}).get("ART_MBP") or [None])[0]),
                                     "hr_only_new": mis(lambda c: (Sn.get(c, {}).get("HR") or [None])[0]),
                                     "hr_only_old": mis(lambda c: (So.get(c, {}).get("HR") or [None])[0])},
           "sign": "bis_steps - bursts - route; clock_reconcile's summary reports the opposite sign"}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
