# SPDX-License-Identifier: MIT
"""Aggregate art_pre_event_trends output (preevents.jsonl): pre-event pulse-pressure slope and the jump at zeroings, flushes and random controls.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,numpy as np
E=[json.loads(l) for l in open('preevents.jsonl')]
def g(k,f): return [x for x in E if x["kind"]==k and f(x)]
print("== pre-event ART pulse pressure slope (%/h) and the jump at the event (1-min, mmHg)")
for k in ("zero","flush","random"):
    e=g(k,lambda x:"d_pp_1min" in x)
    s=np.array([x["pp_slope_pct_h"] for x in e]); d=np.array([x["d_pp_1min"] for x in e])
    fall=s<-20
    print(f"{k:6s} n {len(e)} slope median {np.median(s):+.1f} falling<-20%/h {fall.mean()*100:.0f}% | corr(slope,jump) {np.corrcoef(s,d)[0,1]:+.2f} | jump if falling: median {np.median(d[fall]):+.1f}, >+5 {np.mean(d[fall]>5)*100:.0f}% | jump if not: {np.median(d[~fall]):+.1f}, >+5 {np.mean(d[~fall]>5)*100:.0f}%")
    if k!="random":
        e2=[x for x in e if "d_pp_20s" in x]; s2=np.array([x["pp_slope_pct_h"] for x in e2]); d2=np.array([x["d_pp_20s"] for x in e2]); f2=s2<-20
        print(f"        20 s jump: corr {np.corrcoef(s2,d2)[0,1]:+.2f}; if falling median {np.median(d2[f2]):+.1f} >+5 {np.mean(d2[f2]>5)*100:.0f}% (n {f2.sum()}); if not {np.median(d2[~f2]):+.1f} >+5 {np.mean(d2[~f2]>5)*100:.0f}%")
print("== MAP drift before the event (mmHg/h) and the MAP jump")
for k in ("zero","flush","random"):
    e=g(k,lambda x:"d_map_1min" in x); s=np.array([x["map_slope_h"] for x in e]); d=np.array([x["d_map_1min"] for x in e])
    print(f"{k:6s} corr(MAP slope, MAP jump) {np.corrcoef(s,d)[0,1]:+.2f}; |MAP slope|>10 mmHg/h {np.mean(np.abs(s)>10)*100:.0f}%")
print("== cuff agreement just before vs just after (single events with a reading on each side)")
for k in ("zero","flush","random"):
    e=g(k,lambda x:"cuff_dmap_last" in x and "cuff_dmap_after" in x)
    b=np.array([abs(x["cuff_dmap_last"]) for x in e]); a=np.array([abs(x["cuff_dmap_after"]) for x in e])
    pb=np.array([x["cuff_ppr_last"] for x in e]); pa=np.array([x["cuff_ppr_after"] for x in e])
    print(f"{k:6s} n {len(e)} |dMAP| before {np.median(b):.1f} after {np.median(a):.1f}; improved by >5 mmHg {np.mean(b-a>5)*100:.0f}% worsened by >5 {np.mean(a-b>5)*100:.0f}% | PP ratio before {np.median(pb):.2f} after {np.median(pa):.2f}")
    if k=="zero":
        for x in e: print("   zero case %d: |dMAP| %5.1f -> %5.1f  PPr %.2f -> %.2f  20s dDBP %s  pre PP slope %+.0f%%/h" % (x["case"],abs(x["cuff_dmap_last"]),abs(x["cuff_dmap_after"]),x["cuff_ppr_last"],x["cuff_ppr_after"],x.get("d_dbp_20s"),x["pp_slope_pct_h"]))
print("== events with >=3 cuff readings before: growing disagreement?")
for k in ("zero","flush","random"):
    e=g(k,lambda x:"cuff_absdmap_slope_h" in x)
    for x in e: print(f"  {k} case {x['case']}: |dMAP| slope {x['cuff_absdmap_slope_h']:+.1f}/h over {x['pre_min']} min ({x['cuff_n']} readings), PPr slope {x['cuff_ppr_slope_h']:+.2f}/h, last |dMAP| {abs(x['cuff_dmap_last']):.1f}, after {x.get('cuff_dmap_after')}, 20s dDBP {x.get('d_dbp_20s')}")
