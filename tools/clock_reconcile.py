# SPDX-License-Identifier: MIT
"""Reconcile the VitalDB clock network recording by recording: loop misclosures and a least-squares adjustment.

    python tools/clock_reconcile.py EDGES.csv --out DIR

Reads the edge table of tools/clock_edges.py and, per recording, builds a physiosync.ClockNetwork with the recorder
PC as the reference. Writes:
  cycles.csv     every independent loop: its methods, misclosure (ppm), standard error, z, and shared sources
  adjusted.csv   every edge with its adjusted residual and normalised residual (flag |w| > 3)
  clocks.csv     every clock's adjusted rate against the PC, per recording
  summary.json   per loop type: count, misclosure percentiles, share with |z| > 3, median |z|; per method: pooled
                 variance factor (about 1 when its standard errors are right); the adjusted clock rates' medians
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from physiosync import ClockNetwork, Edge


def run(E: pd.DataFrame):
    cyc, adj_rows, clk = [], [], []
    chi, red = {}, {}
    for rec, g in E.groupby("recording"):
        edges = [Edge(r.a, r.b, float(r.value), float(r.se), r.method, set(str(r.sources).split(";")), str(r.label))
                 for r in g.itertuples()]
        if not any("pc" in (e.a, e.b) for e in edges):
            continue
        net = ClockNetwork(edges, "pc")
        for c in net.cycles():
            kind = " + ".join(sorted(set(c.methods) - {"arithmetic"}))
            cyc.append(dict(recording=rec, loop=kind, misclosure=c.misclosure, se=c.se, z=c.z,
                            shared=";".join(sorted(c.shared_sources))))
        A = net.adjust()
        for e, res, w in zip(edges, A.residuals, A.normalised):
            adj_rows.append(dict(recording=rec, a=e.a, b=e.b, method=e.method, value=e.value, se=e.se,
                                 residual=res, normalised=w))
        for n, v in A.values.items():
            if n != "pc":
                clk.append(dict(recording=rec, clock=n, rate_ppm=v, se=A.se[n], dof=A.dof))
        for m in A.method_chi2:
            chi[m] = chi.get(m, 0.0) + A.method_chi2[m]
            red[m] = red.get(m, 0.0) + A.method_redundancy[m]
    return pd.DataFrame(cyc), pd.DataFrame(adj_rows), pd.DataFrame(clk), chi, red


def summary(C, A, K, chi, red):
    pct = lambda s: [round(float(x), 2) for x in np.percentile(s, [5, 25, 50, 75, 95])] if len(s) else None
    out = {"recordings": int(K.recording.nunique()) if len(K) else 0, "loops": {}, "methods": {}, "clocks": {}}
    for kind, g in C.groupby("loop"):
        out["loops"][kind] = {"loops": len(g), "misclosure_ppm_p5_25_50_75_95": pct(g.misclosure),
                              "abs_z_median": round(float(g.z.abs().median()), 2),
                              "share_abs_z_over_3": round(float((g.z.abs() > 3).mean()), 3),
                              "shared_sources": sorted({s for s in g.shared if s})}
    for m in sorted(chi):
        f = A[A.method == m]
        out["methods"][m] = {"edges": len(f), "redundancy": round(red[m], 1),
                             "variance_factor": round(chi[m] / red[m], 2) if red[m] > 1e-6 else None,
                             "flagged_abs_w_over_3": int((f.normalised.abs() > 3).sum())}
    for n, g in K.groupby("clock"):
        out["clocks"][n] = {"recordings": len(g), "rate_ppm_p5_25_50_75_95": pct(g.rate_ppm)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edges")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C, A, K, chi, red = run(pd.read_csv(a.edges))
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    C.to_csv(o / "cycles.csv", index=False)
    A.to_csv(o / "adjusted.csv", index=False)
    K.to_csv(o / "clocks.csv", index=False)
    S = summary(C, A, K, chi, red)
    (o / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
