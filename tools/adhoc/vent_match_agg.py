# SPDX-License-Identifier: MIT
"""Aggregate vent_sync_pilot breath matching (vent20.jsonl): confident blocks, residuals and per-case slopes.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,numpy as np
rs=[json.loads(l) for l in open('vent20.jsonl')]
conf=lambda b: b[3]>=0.3 and b[3]-b[4]>=0.15
allres=[];confres=[];nw=0;nconf=0;slopes=[];per_case={}
for r in rs:
    g=np.array([x for x in r["rows"]["cvp"] if x[2]>=0.8 and x[3] is not None and x[3]<100],dtype=float)
    if not g.size: continue
    sst=np.array(r["stretch_starts_s"]); seg=np.searchsorted(sst,g[:,0])
    lag=g[:,1].copy(); verified=np.zeros(len(g),bool)
    # one breath offset per Primus stretch: the confident blocks' vote, weighted by their margin
    votes={}
    for b in r.get("match_cvp",[]):
        if conf(b): votes.setdefault(b[1],{}).setdefault(b[2],0.0); votes[b[1]][b[2]]+=b[3]-b[4]
    kst={s_:max(v,key=v.get) for s_,v in votes.items()}
    for b in r.get("match_cvp",[]):
        if not conf(b) or b[2]!=kst.get(b[1]): continue
        m=(seg==b[1])&(np.abs(g[:,0]-b[0])<=900)
        lag[m]+=b[6]; verified[m]=True
    nw+=len(g); nconf+=verified.sum()
    for s in np.unique(seg):
        for sel,store in ((seg==s,allres),((seg==s)&verified,confres)):
            x=g[sel,0]; y=lag[sel]
            if len(x)>=6 and np.ptp(x)>=3600:
                k=np.polyfit(x/3600,y,1); store+=list(y-np.polyval(k,x/3600))
                if store is confres: slopes.append(k[0]/3.6); per_case.setdefault(r["case"],[]).append(round(k[0]/3.6))
print("good CVP windows",nw,"with breath identity confirmed",nconf,f"({nconf/nw*100:.0f}%)")
print("|resid| about stretch lines, all good windows p50,p90:",np.percentile(np.abs(allres),[50,90]).round(), " confirmed only:",np.percentile(np.abs(confres),[50,90]).round(),len(confres))
print("stretch slopes (confirmed) ppm p10,50,90",np.percentile(slopes,[10,50,90]).round(),len(slopes)); print(per_case)
