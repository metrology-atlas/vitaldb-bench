# SPDX-License-Identifier: MIT
"""Null (random-anchor) version of art_line_trends for cuff-line agreement; reuses the top of art_line_trends.py (reads nibp_art206.jsonl).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import os
import json,numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'art_line_trends.py')).read().split("# ---------- figure")[0].split("# ---------- A:")[0].split("rng=np.random")[0])
N={json.loads(l)["case"]:json.loads(l) for l in open('nibp_art206.jsonl')}
rows=[]
for c,r in N.items():
    if not r.get("art"): continue
    T=np.array(r["t"])+r["first"]; V=np.array(r["v"],float)
    ok=[i for i,a in enumerate(r["art"]) if a is not None]; keep=[]
    for i in ok:
        if keep and T[i]-T[keep[-1]]<=10: keep[-1]=i
        else: keep.append(i)
    for i in keep: rows.append((c,T[i],r["art"][i][0]-V[i,2],V[i,2]))
R=np.array([x[1:] for x in rows],float); cs=np.array([x[0] for x in rows])
b=np.polyfit(R[:,2],R[:,1],1); adj=R[:,1]-np.polyval(b,R[:,2])
def stat(order_fn):
    pos=n=0; sl=[]
    for c in np.unique(cs):
        m=np.flatnonzero(cs==c)
        if m.size>=4 and np.ptp(R[m,0])>=3600:
            h=(R[m,0]-R[m,0].min())/3600; y=adj[m][order_fn(m.size)]
            dev=np.abs(y-np.median(y[:2])); s=np.polyfit(h,dev,1)[0]; sl.append(s); pos+=s>0; n+=1
    return pos,n,np.median(sl)
rng=np.random.default_rng(0)
obs=stat(lambda k: np.arange(k)); print("observed positive",obs)
null=[stat(lambda k: rng.permutation(k)) for _ in range(300)]
pn=np.array([x[0] for x in null]); mn=np.array([x[2] for x in null])
print("null (shuffled order): positive median %.0f (p95 %.0f), slope median %.2f; p(positive>=obs) = %.3f; p(slope>=obs) = %.3f"%(np.median(pn),np.percentile(pn,95),np.median(mn),np.mean(pn>=obs[0]),np.mean(mn>=obs[2])))
# alternative unbiased: signed slope of adjusted dMAP magnitude relative to case median (no start anchoring)
pos=n=0; sl=[]
for c in np.unique(cs):
    m=np.flatnonzero(cs==c)
    if m.size>=4 and np.ptp(R[m,0])>=3600:
        h=(R[m,0]-R[m,0].min())/3600; s=np.polyfit(h,np.abs(adj[m]),1)[0]; sl.append(s); pos+=s>0; n+=1
print("slope of |level-adjusted dMAP| (no anchoring): positive %d/%d, median %+.2f mmHg/h"%(pos,n,np.median(sl)))
