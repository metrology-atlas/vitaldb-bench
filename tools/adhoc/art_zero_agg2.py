# SPDX-License-Identifier: MIT
"""Aggregate art_zero_jumps output given as argv[1] (20 s window run): the same groups as art_zero_agg plus 'other access'.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,sys,numpy as np
rs=[json.loads(l) for l in open(sys.argv[1])]
G={"zero (open to air)":[],"flush, no air":[],"other access":[],"occlusion (cuff)":[],"random control":[]}
for r in rs:
    for q in r.get("accesses",[]):
        if not q["delta"]: continue
        if q.get("air"): G["zero (open to air)"].append(q["delta"])
        elif "high" in q["kinds"]: G["flush, no air"].append(q["delta"])
        else: G["other access"].append(q["delta"])
    G["occlusion (cuff)"]+=[q["delta"] for q in r.get("occlusions",[]) if q["delta"]]
    G["random control"]+=r.get("controls",[])
def show(m,thr):
    for k,v in G.items():
        d=np.array([x[m] for x in v if x.get(m) is not None],dtype=float)
        if d.size<10: continue
        print(f"  {k:20s} n {d.size:5d} median {np.median(d):+8.3f} |d| p50 {np.median(np.abs(d)):7.3f} p90 {np.percentile(np.abs(d),90):7.3f} >{thr}: {np.mean(np.abs(d)>thr)*100:4.0f}%")
for m,thr in (("map",10),("dbp",10),("sbp",10),("pw50_ms",20),("pw50_rr",0.03),("rr_ms",50)):
    print(m); show(m,thr)
# flush: signed pulse width change split by sign, and relation to SBP change
F=[x for x in G["flush, no air"]+G["zero (open to air)"] if x.get("pw50_ms") is not None]
d=np.array([x["pw50_ms"] for x in F]); s=np.array([x["sbp"] for x in F]); pp=np.array([x["sbp"]-x["dbp"] for x in F])
print("flush accesses: pw50 narrower after (<-10 ms) %.0f%%, wider (>+10 ms) %.0f%%; corr(dpw50, dSBP) %.2f, corr(dpw50, dPP) %.2f"%(np.mean(d<-10)*100,np.mean(d>10)*100,np.corrcoef(d,s)[0,1],np.corrcoef(d,pp)[0,1]))
R=[x for x in G["random control"] if x.get("pw50_ms") is not None]; dr=np.array([x["pw50_ms"] for x in R])
print("random: narrower %.0f%%, wider %.0f%%"%(np.mean(dr<-10)*100,np.mean(dr>10)*100))
pre=[q["pre"]["pw50_ms"] for r in rs for q in r.get("accesses",[]) if q.get("pre") and q["pre"].get("pw50_ms")]
print("pre-access pw50 ms p10/50/90",np.percentile(pre,[10,50,90]).round())
