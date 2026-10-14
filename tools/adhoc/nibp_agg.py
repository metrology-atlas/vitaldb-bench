# SPDX-License-Identifier: MIT
"""Aggregate nibp_scan output (nibp_art206.jsonl) with art_line_labels.csv: cuff-line differences around line accesses.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,csv,os,numpy as np
rs=[json.loads(l) for l in open('nibp_art206.jsonl')]
acc={}
for r in csv.DictReader(open(os.path.join(os.environ['VITALDB_WORK'], 'results/pat/art_line_labels.csv'))):
    if r["label"]=="art_line:access": acc.setdefault(int(r["case"]),[]).append((float(r["start_s"]),float(r["end_s"]),"low" in r["kinds"],"high" in r["kinds"]))
P=[];case_stats=[];shift={"zero":[],"flush":[],"random":[]};ppshift={"zero":[],"flush":[],"random":[]}
rng=np.random.default_rng(0)
for r in rs:
    if not r.get("art"): continue
    t0=r["first"]; T=np.array(r["t"])+t0; V=np.array(r["v"],float)
    A=r["art"]; ok=np.array([a is not None for a in A])
    if ok.sum()<3: continue
    Aa=np.array([a[:3] for a in A if a is not None],float); Tn=T[ok]; Vn=V[ok]
    # merge readings < 10 s apart: keep the last
    keep=np.r_[np.diff(Tn)>10,True]; Aa,Tn,Vn=Aa[keep],Tn[keep],Vn[keep]
    dmap=Aa[:,0]-Vn[:,2]; dsbp=Aa[:,1]-Vn[:,0]; ddbp=Aa[:,2]-Vn[:,1]
    ppr=(Aa[:,1]-Aa[:,2])/np.maximum(Vn[:,0]-Vn[:,1],5)
    for i in range(len(Tn)): P.append((dmap[i],dsbp[i],ddbp[i],ppr[i],Vn[i,2]))
    if len(Tn)>=4: case_stats.append((np.median(dmap),np.std(dmap),np.median(ppr),len(Tn)))
    def around(t, store, store2):
        b=(Tn>=t-1800)&(Tn<t); a_=(Tn>t)&(Tn<=t+1800)
        if b.sum()>=2 and a_.sum()>=2:
            store.append(np.median(dmap[a_])-np.median(dmap[b])); store2.append(np.median(ppr[a_])-np.median(ppr[b]))
    for s,e,low,high in acc.get(r["case"],[]):
        if low: around(e, shift["zero"], ppshift["zero"])
        elif high: around(e, shift["flush"], ppshift["flush"])
    if len(Tn)>=4:
        for tt in rng.uniform(Tn[0],Tn[-1],10): around(tt, shift["random"], ppshift["random"])
P=np.array(P)
print("pairs",len(P),"cases",len(case_stats))
for i,n in enumerate(("ART-NIBP MAP","ART-NIBP SBP","ART-NIBP DBP")):
    print(f"{n}: median {np.median(P[:,i]):+.1f} p10/p90 {np.percentile(P[:,i],10):+.1f}/{np.percentile(P[:,i],90):+.1f}  |d|>10: {np.mean(np.abs(P[:,i])>10)*100:.0f}%  |d|>20: {np.mean(np.abs(P[:,i])>20)*100:.0f}%")
print("PP ratio ART/NIBP p10/50/90",np.percentile(P[:,3],[10,50,90]).round(2),"<0.7:",f"{np.mean(P[:,3]<0.7)*100:.0f}%",">1.5:",f"{np.mean(P[:,3]>1.5)*100:.0f}%")
C=np.array(case_stats)
print("per case: median dMAP p10/50/90",np.percentile(C[:,0],[10,50,90]).round(1),"within-case SD of dMAP p25/50/75",np.percentile(C[:,1],[25,50,75]).round(1))
print("share of variance of dMAP between cases: %.2f"%(np.var(C[:,0])/(np.var(C[:,0])+np.median(C[:,1])**2)))
for k in shift:
    d=np.array(shift[k]); q=np.array(ppshift[k])
    if d.size: print(f"shift in median dMAP across {k}: n {d.size} |shift| p50 {np.median(np.abs(d)):.1f} p90 {np.percentile(np.abs(d),90):.1f} >5: {np.mean(np.abs(d)>5)*100:.0f}% | PP-ratio |shift| p50 {np.median(np.abs(q)):.2f} >0.2: {np.mean(np.abs(q)>0.2)*100:.0f}%")
# bias by pressure level (known NIBP behaviour)
for lo,hi in ((0,65),(65,80),(80,95),(95,300)):
    m=(P[:,4]>=lo)&(P[:,4]<hi); print(f"NIBP MAP {lo}-{hi}: n {m.sum()} median dMAP {np.median(P[m,0]):+.1f}")
