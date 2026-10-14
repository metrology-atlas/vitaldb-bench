# SPDX-License-Identifier: MIT
"""Aggregate vent_sync_pilot output (vent20.jsonl): per-stretch drift slopes, residuals and agreement between methods.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,numpy as np
rs=[json.loads(l) for l in open('vent20.jsonl')]
def good_cvp(r):
    a=np.array(r["rows"]["cvp"],dtype=float)
    if not a.size: return a
    return a[(a[:,2]>=0.8)&(a[:,3]<100)]
per_slopes=[];resid=[];case_sl={};agree={"ramp":[],"sbp":[]};dsd={"ramp":[],"sbp":[]}
levch=[];levch_at=[]
for r in rs:
    g=good_cvp(r)
    if len(g)<4: continue
    sst=np.array(r["stretch_starts_s"]); seg=np.searchsorted(sst,g[:,0])
    sl=[]
    for s in np.unique(seg):
        x=g[seg==s]
        if len(x)>=6 and np.ptp(x[:,0])>=3600:
            k=np.polyfit(x[:,0]/3600,x[:,1],1); sl.append(k[0]/3.6)  # ms/h -> ppm
            resid+=list(x[:,1]-np.polyval(k,x[:,0]/3600))
            # modulation slopes in the same stretch
            for m in ("ramp","sbp"):
                y=np.array(r["rows"].get(m,[]),dtype=float)
                if not y.size: continue
                y=y[(y[:,2]>=0.05)]
                ys=y[np.searchsorted(sst,y[:,0])==s] if y.size else y
                if len(ys)>=6 and np.ptp(ys[:,0])>=3600:
                    T=np.median(ys[:,4])*1000; lag=np.unwrap(ys[:,1]/T*2*np.pi)/(2*np.pi)*T
                    km=np.polyfit(ys[:,0]/3600,lag,1); agree[m].append((k[0]/3.6,km[0]/3.6))
                    # per-window difference to CVP (wrapped)
                    cv=dict(zip(np.round(x[:,0],1),x[:,1]))
                    d=[((v-cv[t]+T/2)%T)-T/2 for t,v in zip(np.round(ys[:,0],1),ys[:,1]) if t in cv]
                    if len(d)>=5: dsd[m].append(np.std(np.array(d)-np.median(d)))
    if sl: case_sl[r["case"]]=sl; per_slopes+=sl
    # level changes between consecutive good windows (<=15 min apart)
    for i in range(1,len(g)):
        if g[i,0]-g[i-1,0]<=900:
            dl=g[i,1]-g[i-1,1]
            if abs(dl)>300:
                levch.append(dl); levch_at.append(np.any((sst>g[i-1,0]-300)&(sst<=g[i,0]+300)))
ps=np.array(per_slopes)
print("cases with CVP slope:",len(case_sl),"stretch slopes:",len(ps),"ppm p10,25,50,75,90",np.percentile(ps,[10,25,50,75,90]).round(0))
print("within-case spread of stretch slopes (cases with >=2): ",[ (c,np.round(v,0).tolist()) for c,v in case_sl.items() if len(v)>=2])
print("residual about stretch lines |res| p50,p90 ms",np.percentile(np.abs(resid),[50,90]).round(0),len(resid))
for m in agree:
    a=np.array(agree[m]); 
    if a.size: print(m,"stretch slope vs CVP: n",len(a),"|diff| p50,p90 ppm",np.percentile(np.abs(a[:,1]-a[:,0]),[50,90]).round(0),"corr %.2f"%np.corrcoef(a.T)[0,1], "| per-window SD of (mod - CVP) ms p50",np.round(np.median(dsd[m])))
print("CVP level changes >300 ms between consecutive windows:",len(levch),"at a Primus stamp jump:",int(np.sum(levch_at)), "sizes p10,50,90",np.percentile(levch,[10,50,90]).round())
# modulation strength
for m in ("rr","ramp","sbp","pp"):
    r2=[x[2] for r in rs for x in r["rows"].get(m,[])]
    print(m,"R2 p25,50,75",np.percentile(r2,[25,50,75]).round(3),"windows R2>=0.05: %.0f%%"%(np.mean(np.array(r2)>=0.05)*100))
print("CVP windows good/all", sum(len(good_cvp(r)) for r in rs), sum(len(r["rows"]["cvp"]) for r in rs))
