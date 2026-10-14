# SPDX-License-Identifier: MIT
"""Aggregate flush_ringing output (full_fr.jsonl, full_n.jsonl): usable flush releases, damping and natural frequency.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,numpy as np
rs=[json.loads(l) for l in open('full_fr.jsonl')]
N={json.loads(l)["case"]:json.loads(l) for l in open('full_n.jsonl')}
R=[];F=[]
for r in rs:
    for f in r.get("flushes",[]):
        F.append(f)
        for x in f["releases"]: R.append({**x,"case":r["case"],"pre":f["pre"],"post":f["post"]})
print("flush accesses",len(F),"with a usable release",sum(bool(f["releases"]) for f in F),"releases",len(R),"cases",len({x['case'] for x in R}))
ring=[x for x in R if x.get("ringing")]; noring=[x for x in R if x.get("ringing") is False]
print("ringing (undershoot >5 mmHg): %d (%.0f%%); no undershoot: %d"%(len(ring),len(ring)/len(R)*100,len(noring)))
fall=np.array([x["fall_ms"] for x in R if x["fall_ms"] is not None]); fr=np.array([x["fall_ms"] for x in ring if x["fall_ms"] is not None]); fn_=np.array([x["fall_ms"] for x in noring if x["fall_ms"] is not None])
print("fall 80->20% ms p10/50/90 all",np.percentile(fall,[10,50,90]),"ringing",np.percentile(fr,[10,50,90]),"no ringing",np.percentile(fn_,[10,50,90]))
z=np.array([x["zeta"] for x in ring if x.get("fn_hz")]); fnz=np.array([x["fn_hz"] for x in ring if x.get("fn_hz")])
print("zeta p10/25/50/75/90",np.percentile(z,[10,25,50,75,90]).round(2),"fn Hz p10/25/50/75/90",np.percentile(fnz,[10,25,50,75,90]).round(1))
# simple Gardner-style zones (approximate): underdamped <0.3 ; optimal 0.3-0.7 ; high 0.7-1
for lo,hi in ((0,0.2),(0.2,0.3),(0.3,0.45),(0.45,0.7),(0.7,1.01)):
    m=(z>=lo)&(z<hi); print(f"   zeta {lo}-{hi}: {m.mean()*100:.0f}%  fn median {np.median(fnz[m]) if m.any() else float('nan'):.1f}")
print("fn < 10 Hz: %.0f%%, 10-15: %.0f%%, 15-25: %.0f%%, >25: %.0f%%"%tuple(np.mean((fnz>=a)&(fnz<b))*100 for a,b in ((0,10),(10,15),(15,25),(25,999))))
# within-case changes
per={}
for x in ring:
    if x.get("fn_hz"): per.setdefault(x["case"],[]).append((x["t"],x["fn_hz"],x["zeta"]))
sl=[];sz=[];spread=[]
for c,v in per.items():
    v=np.array(sorted(v))
    if len(v)>=3 and np.ptp(v[:,0])>3600:
        h=(v[:,0]-v[0,0])/3600; sl.append(np.polyfit(h,v[:,1],1)[0]); sz.append(np.polyfit(h,v[:,2],1)[0]); spread.append(np.std(v[:,1]))
sl=np.array(sl); sz=np.array(sz)
print(f"cases with >=3 ringing releases over >1 h: {sl.size}; fn slope Hz/h median {np.median(sl):+.2f} (falling {np.mean(sl<0)*100:.0f}%), zeta slope median {np.median(sz):+.3f}/h; within-case SD of fn median {np.median(spread):.1f} Hz")
# relation to pressures after the flush
def pp(w): return None if w is None else w[1]-w[2]
for nm,sel in (("ringing, zeta<0.25",lambda x:x.get("ringing") and x.get("zeta",1)<0.25),("ringing, 0.25-0.5",lambda x:x.get("ringing") and 0.25<=x.get("zeta",1)<0.5),("ringing, >=0.5",lambda x:x.get("ringing") and x.get("zeta",1)>=0.5),("no undershoot, fall<=15 ms",lambda x:x.get("ringing") is False and (x["fall_ms"] or 99)<=15),("no undershoot, fall>15 ms",lambda x:x.get("ringing") is False and (x["fall_ms"] or 0)>15)):
    e=[x for x in R if sel(x)]
    post=[pp(x["post"]) for x in e if x["post"]]; dpp=[pp(x["post"])-pp(x["pre"]) for x in e if x["post"] and x["pre"]]
    print(f"{nm:28s} n {len(e):5d}  post PP median {np.median(post) if post else float('nan'):.1f}  dPP median {np.median(dpp) if dpp else float('nan'):+.1f} (n {len(dpp)})")
# cuff PP ratio within 60 min after the release, by damping class
def cuff_after(c,t):
    r=N.get(c)
    if not r or not r.get("art"): return []
    T=[r["first"]+x for x in r["t"]]; out=[]
    for i,(tt,a) in enumerate(zip(T,r["art"])):
        if a and t<tt<=t+3600: v=r["v"][i]; out.append((a[1]-a[2])/max(v[0]-v[1],5))
    return out
for nm,sel in (("zeta<0.25",lambda x:x.get("ringing") and x.get("zeta",1)<0.25),("zeta 0.25-0.5",lambda x:x.get("ringing") and 0.25<=x.get("zeta",1)<0.5),("zeta>=0.5",lambda x:x.get("ringing") and x.get("zeta",1)>=0.5),("no undershoot",lambda x:x.get("ringing") is False)):
    v=[q for x in R if sel(x) for q in cuff_after(x["case"],x["t"])]
    print(f"cuff PP ratio (line/cuff) within 1 h after, {nm:14s}: n {len(v)} median {np.median(v) if v else float('nan'):.2f}  >1.5: {np.mean(np.array(v)>1.5)*100 if v else 0:.0f}%  <0.7: {np.mean(np.array(v)<0.7)*100 if v else 0:.0f}%")

# continuous: per release, median cuff PP ratio within 1 h after vs zeta and fn
pts=[]
for x in R:
    if x.get("ringing") and x.get("fn_hz"):
        v=cuff_after(x["case"],x["t"])
        if v: pts.append((x["zeta"],x["fn_hz"],np.median(v)))
P=np.array(pts)
from scipy.stats import spearmanr
print("releases with cuff after:",len(P),"spearman PPr~zeta %.2f (p %.3g), PPr~fn %.2f (p %.3g)"%(*spearmanr(P[:,0],P[:,2]),*spearmanr(P[:,1],P[:,2])))
for lo,hi in ((0,0.2),(0.2,0.35),(0.35,0.5),(0.5,1.01)):
    m=(P[:,0]>=lo)&(P[:,0]<hi); print(f"   zeta {lo}-{hi}: n {m.sum()} PP ratio median {np.median(P[m,2]):.2f}")
cs={}
for x in R:
    if x.get("ringing") is False:
        v=cuff_after(x["case"],x["t"])
        if v: cs.setdefault(x["case"],[]).extend(v)
print("no-undershoot cuff readings by case:",{c:(len(v),round(float(np.median(v)),2)) for c,v in cs.items()})
