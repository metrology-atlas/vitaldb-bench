# SPDX-License-Identifier: MIT
"""Aggregate art_zero_jumps output (zero206.jsonl): pressure change across zeroings, flushes, cuff occlusions and random controls; open-to-air readings.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,numpy as np
rs=[json.loads(l) for l in open('zero206.jsonl')]
print("cases",len(rs),"errors",sum('error' in r for r in rs))
G={"zero (open-to-air plateau)":[],"flush/draw, no air":[],"occlusion (cuff)":[],"random control":[]}
air=[];pairs=[];nib=[]
for r in rs:
    for q in r.get("accesses",[]):
        if q.get("air"): key="zero (open-to-air plateau)"
        elif "high" in q["kinds"]: key="flush/draw, no air"
        else: continue
        if q["delta"]: G[key].append(q["delta"])
        for e in q.get("air",[]):
            air.append(e)
            if q["delta"] and e["n"]>=3: pairs.append((e["first"],e["step"],q["delta"]["map"],q["delta"]["dbp"]))
        if q.get("air") and q.get("nibp_pre") is not None and q.get("nibp_post") is not None:
            nib.append((q["nibp_pre"],q["nibp_post"], q["air"][0]["first"]))
    for q in r.get("occlusions",[]):
        if q["delta"]: G["occlusion (cuff)"].append(q["delta"])
    G["random control"]+=r.get("controls",[])
for k,v in G.items():
    if not v: continue
    for m in ("map","dbp","sbp"):
        d=np.array([x[m] for x in v])
        print(f"{k:28s} {m}: n {d.size:5d} median {np.median(d):+.1f} |d| p50 {np.median(np.abs(d)):.1f} p90 {np.percentile(np.abs(d),90):.1f}  >5: {np.mean(np.abs(d)>5)*100:.0f}%  >10: {np.mean(np.abs(d)>10)*100:.0f}%")
A=np.array([(e["first"],e["last"],e["step"],e["n"]) for e in air])
print("air plateaus",len(A),"first reading p10/50/90",np.percentile(A[:,0],[10,50,90]).round(1),"last p10/50/90",np.percentile(A[:,1],[10,50,90]).round(1))
big=np.abs(A[:,0])>=3
print("air first |>=3| mmHg:",big.sum(),f"({big.mean()*100:.0f}%)","of which ends nearer 0 (|last|<1 and |last|<|first|):",int(np.sum(big&(np.abs(A[:,1])<1)&(np.abs(A[:,1])<np.abs(A[:,0])))))
P=np.array(pairs)
if P.size:
    print("air-first vs delta MAP corr %.2f, vs delta DBP corr %.2f, n %d"%(np.corrcoef(P[:,0],P[:,2])[0,1],np.corrcoef(P[:,0],P[:,3])[0,1],len(P)))
    m=np.abs(P[:,1])>=3
    if m.sum()>5: print("  where a step >=3 occurred on the plateau: slope of dMAP on step %.2f, corr %.2f, n %d"%(np.polyfit(P[m,1],P[m,2],1)[0],np.corrcoef(P[m,1],P[m,2])[0,1],m.sum()))
N=np.array(nib)
if N.size:
    print("NIBP agreement around zero accesses: n",len(N),"|ART-NIBP| before p50 %.1f after p50 %.1f; before>after in %.0f%%"%(np.median(np.abs(N[:,0])),np.median(np.abs(N[:,1])),np.mean(np.abs(N[:,0])>np.abs(N[:,1]))*100))
    m=np.abs(N[:,2])>=3
    if m.sum()>3: print("  with air offset >=3: n",m.sum(),"before %.1f after %.1f"%(np.median(np.abs(N[m,0])),np.median(np.abs(N[m,1]))))
