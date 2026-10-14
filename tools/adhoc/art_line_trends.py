# SPDX-License-Identifier: MIT
"""ART pulse pressure after line accesses against random anchors, and cuff-line agreement trends (reads artmin206.jsonl, nibp_art206.jsonl from art_minute_series and nibp_scan). Drift document: 'art_minute_series + art_line_trends (+ null)'.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,numpy as np
M={json.loads(l)["case"]:json.loads(l) for l in open('artmin206.jsonl')}
N={json.loads(l)["case"]:json.loads(l) for l in open('nibp_art206.jsonl')}
rng=np.random.default_rng(1)
# ---------- B: continuous ART pulse pressure after accesses vs random anchors ----------
bins=np.arange(0,100,10)
def traj(series, anchors, nxt):
    out=[]
    t=series[:,0]; pp=series[:,2]-series[:,3]; mp=series[:,1]
    for a0,stop in zip(anchors,nxt):
        m=(t>=a0)&(t<min(stop,a0+6000))
        if m.sum()<30: continue
        tt=(t[m]-a0)/60; base=np.median(pp[m][tt<10]) if (tt<10).sum()>=3 else None
        if base is None: continue
        out.append((tt, pp[m]/base, mp[m]-np.median(mp[m][tt<10])))
    return out
A=[];R=[];slopes_a=[];slopes_r=[]
for c,r in M.items():
    if "error" in r or len(r["minutes"])<60: continue
    s=np.array(r["minutes"],float)
    ends=sorted(x[2] for x in r["accesses"]); starts=sorted(x[0] for x in r["accesses"])
    nx=[next((st for st in starts if st>e),1e18) for e in ends]
    A+=traj(s,ends,nx)
    rnd=rng.uniform(s[0,0],s[-1,0],len(ends) or 3)
    nxr=[next((st for st in starts if st>e),1e18) for e in rnd]
    R+=traj(s,rnd,nxr)
def summarize(T,label):
    rows=[]
    for lo in bins:
        v=[x for tt,p,_ in T for x in p[(tt>=lo)&(tt<lo+10)]]
        w=[x for tt,_,mm in T for x in mm[(tt>=lo)&(tt<lo+10)]]
        rows.append((lo,len(v),np.median(v) if v else np.nan,np.median(w) if w else np.nan))
    sl=[np.polyfit(tt,p,1)[0]*60 for tt,p,_ in T if np.ptp(tt)>=40]
    print(f"{label}: segments {len(T)}; PP slope per hour (fraction of start PP) median {np.median(sl):+.3f}, IQR {np.percentile(sl,25):+.3f}..{np.percentile(sl,75):+.3f}; falling {np.mean(np.array(sl)<0)*100:.0f}%")
    print("   min since anchor: "+" ".join(f"{lo}:{p:.3f}/{m:+.1f}" for lo,n,p,m in rows))
    return sl
sa=summarize(A,"after line access"); sr=summarize(R,"after random time")
# ---------- A: cuff vs line agreement trends ----------
pairs=[]
for c,r in N.items():
    if not r.get("art"): continue
    t0=r["first"]; T=np.array(r["t"])+t0; V=np.array(r["v"],float)
    ok=[i for i,a in enumerate(r["art"]) if a is not None]
    if not ok: continue
    acc=sorted(x[2] for x in M.get(c,{}).get("accesses",[]))
    mins=M.get(c,{}).get("minutes",[])
    tline=mins[0][0] if mins else None
    keep=[]
    for i in ok:
        if keep and T[i]-T[keep[-1]]<=10: keep[-1]=i
        else: keep.append(i)
    for i in keep:
        a=r["art"][i]; prev=[e for e in acc if e<=T[i]]
        pairs.append((c,T[i],a[0]-V[i,2],(a[1]-a[2])/max(V[i,0]-V[i,1],5),V[i,2],(T[i]-prev[-1])/60 if prev else np.nan,(T[i]-tline)/3600 if tline else np.nan))
P=np.array([p[1:] for p in pairs],float); cases=np.array([p[0] for p in pairs])
b=np.polyfit(P[:,3],P[:,1],1); adj=P[:,1]-np.polyval(b,P[:,3])   # remove NIBP pressure-level bias
print(f"\npairs {len(P)}; NIBP level bias: dMAP = {b[1]:.1f} {b[0]:+.2f}*NIBP_MAP")
sl_d=[];sl_abs=[];sl_ppr=[]
for c in np.unique(cases):
    m=cases==c
    if m.sum()>=4 and np.ptp(P[m,0])>=3600:
        h=(P[m,0]-P[m,0].min())/3600
        dev=adj[m]   # level-adjusted cuff-line MAP difference (anchoring on the first readings biases the slope upward)
        sl_d.append(np.polyfit(h,adj[m],1)[0]); sl_abs.append(np.polyfit(h,np.abs(dev),1)[0]); sl_ppr.append(np.polyfit(h,P[m,2],1)[0])
for nm,s in (("adjusted dMAP slope (mmHg/h)",sl_d),("|level-adjusted dMAP| slope (mmHg/h)",sl_abs),("PP ratio slope (/h)",sl_ppr)):
    s=np.array(s); print(f"{nm}: cases {s.size} median {np.median(s):+.2f} IQR {np.percentile(s,25):+.2f}..{np.percentile(s,75):+.2f} positive {np.mean(s>0)*100:.0f}%")
for nm,col,edges in (("hours since line start",5,[0,1,2,3,4,6,12]),("min since last access",4,[0,15,30,60,120,600])):
    x=P[:,col]
    print(nm+":"," ".join(f"[{lo},{hi}): n {int(np.sum((x>=lo)&(x<hi)))} |adj dMAP| {np.median(np.abs(adj[(x>=lo)&(x<hi)])):.1f} PPr {np.median(P[(x>=lo)&(x<hi),2]):.2f}" for lo,hi in zip(edges[:-1],edges[1:]) if np.sum((x>=lo)&(x<hi))>=10))

# ---------- figure ----------
import sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from math import comb
k=int(np.sum(np.array(sl_abs)>0)); n=len(sl_abs); p=sum(comb(n,i) for i in range(k,n+1))/2**n
print(f"sign test |dMAP| growing: {k}/{n}, one-sided p = {p:.3f}")
fig,ax=plt.subplots(1,3,figsize=(18,5))
for T,lab,col in ((A,"after a line access","C3"),(R,"after a random time","0.4")):
    med=[];q1=[];q3=[]
    for lo in bins:
        v=[x for tt,pp,_ in T for x in pp[(tt>=lo)&(tt<lo+10)]]
        med.append(np.median(v)); q1.append(np.percentile(v,25)); q3.append(np.percentile(v,75))
    ax[0].plot(bins+5,med,"o-",color=col,label=lab); ax[0].fill_between(bins+5,q1,q3,color=col,alpha=.15)
ax[0].axhline(1,color="k",lw=.6); ax[0].set_xlabel("minutes since the anchor"); ax[0].set_ylabel("ART pulse pressure / first 10 min")
ax[0].set_title(f"ART pulse pressure between accesses ({len(A)} segments, 206 cases)",fontsize=10); ax[0].legend(fontsize=8); ax[0].set_ylim(0.7,1.3)
ax[1].hist(sl_abs,bins=np.arange(-6,6.5,0.5),color="C0"); ax[1].axvline(0,color="k",lw=.6)
ax[1].set_xlabel("slope of |cuff–line MAP difference| (level-adjusted, mmHg/h)"); ax[1].set_ylabel("cases")
ax[1].set_title(f"Growing disagreement: {k}/{n} cases positive (p = {p:.3f}), median {np.median(sl_abs):+.2f} mmHg/h",fontsize=10)
ax[2].hist(sl_ppr,bins=np.arange(-0.3,0.31,0.02),color="C2"); ax[2].axvline(0,color="k",lw=.6)
ax[2].set_xlabel("slope of pulse-pressure ratio, line / cuff (per hour)"); ax[2].set_ylabel("cases")
ax[2].set_title(f"Pulse-pressure ratio trend: {np.mean(np.array(sl_ppr)<0)*100:.0f}% falling, median {np.median(sl_ppr):+.3f}/h",fontsize=10)
fig.suptitle("Trends that would indicate a worsening arterial line",fontsize=12); fig.tight_layout(); fig.savefig(sys.argv[1],dpi=80)
