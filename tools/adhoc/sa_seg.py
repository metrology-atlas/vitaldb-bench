# SPDX-License-Identifier: MIT
"""Shared-artefact ECG-EEG1 offsets split into segments between jumps (argv[1] output, argv[2] units.csv): spans, residuals and jump sizes.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,sys,csv,numpy as np
rs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
U={int(r["case"]):r for r in csv.DictReader(open(sys.argv[2]))}
def good(rows):
    a=np.array(rows); return a[(a[:,2]>=0.3)&(a[:,2]-a[:,3]>=0.15)] if a.size else np.empty((0,5))
P=[];jumps=[];spans=[];res=[];per=[]
for r in rs:
    g=good(r["burst"].get("ecg~eeg1",[]))
    if len(g)<4: continue
    g=g[np.argsort(g[:,0])]
    br=[0]
    for i in range(1,len(g)):
        dl=g[i,1]-g[i-1,1]; dtt=g[i,0]-g[i-1,0]
        if dl > 60 + 0.1*dtt/3.6*0 or dtt>3600 : br.append(i)
    br.append(len(g))
    u=U.get(r["case"],{})
    try: pred=-(float(u["bis_drift_ppm"])-float(u["snuadc_drift_ppm"]))
    except: pred=None
    sl=[]
    for a,b in zip(br[:-1],br[1:]):
        s=g[a:b]
        if len(s)>=4 and np.ptp(s[:,0])>=1200:
            k=np.polyfit(s[:,0],s[:,1],1); sl.append(k[0]*1000)  # ms/s*1000 = ppm
            res+=list(s[:,1]-np.polyval(k,s[:,0]))
    for i in br[1:-1]:
        if g[i,0]-g[i-1,0]<=900: jumps.append((g[i,1]-g[i-1,1], g[i,0]))
    if sl and pred is not None:
        P.append((r["case"],round(float(np.median(sl)),1),round(pred,1),len(sl)))
for p in P: print(p)
a=np.array([(p[1],p[2]) for p in P])
print("slope vs predicted: median diff %.1f ppm, MAD %.1f, corr %.2f"%(np.median(a[:,0]-a[:,1]),np.median(np.abs(a[:,0]-a[:,1])),np.corrcoef(a.T)[0,1]))
j=np.array([x[0] for x in jumps]); print("jumps (consecutive windows <=15 min apart):",len(j),"size p10,50,90",np.percentile(j,[10,50,90]).round(0))
print("residual about segment line: |res| p50,p90 ms",np.percentile(np.abs(res),[50,90]).round(1),len(res))
