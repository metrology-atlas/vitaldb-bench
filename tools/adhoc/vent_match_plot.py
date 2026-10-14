# SPDX-License-Identifier: MIT
"""Figure: breath-matched CVP lags per case (argv[1] = vent_sync_pilot output, argv[2] = cases).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,sys,numpy as np,matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rs={json.loads(l)["case"]:json.loads(l) for l in open(sys.argv[1])}
cases=[int(x) for x in sys.argv[2].split(",")]
conf=lambda b: b[3]>=0.3 and b[3]-b[4]>=0.15
fig,ax=plt.subplots(len(cases)//2,2,figsize=(18,2.6*len(cases)//2))
for a,c in zip(ax.flat,cases):
    r=rs[c]; g=np.array([x for x in r["rows"]["cvp"] if x[2]>=0.8 and x[3] is not None and x[3]<100],dtype=float)
    if not g.size: a.set_title(f"case {c}: no good CVP windows",fontsize=9); continue
    sst=np.array(r["stretch_starts_s"]); seg=np.searchsorted(sst,g[:,0]); lag=g[:,1].copy(); ver=np.zeros(len(g),bool); fixed=np.zeros(len(g),bool)
    # one breath offset per Primus stretch: the confident blocks' vote, weighted by their margin
    votes={}
    for b in r.get("match_cvp",[]):
        if conf(b): votes.setdefault(b[1],{}).setdefault(b[2],0.0); votes[b[1]][b[2]]+=b[3]-b[4]
    kst={s_:max(v,key=v.get) for s_,v in votes.items()}
    for b in r.get("match_cvp",[]):
        if not conf(b) or b[2]!=kst.get(b[1]): continue
        m=(seg==b[1])&(np.abs(g[:,0]-b[0])<=900); lag[m]+=b[6]; ver[m]=True; fixed[m]|=b[2]!=0
    a.plot(g[~ver,0]/3600,lag[~ver],"o",mfc="none",color="C0",ms=4,label="breath identity unconfirmed")
    a.plot(g[ver,0]/3600,lag[ver],"o",color="C0",ms=4,label="confirmed by breath matching")
    if fixed.any(): a.plot(g[fixed,0]/3600,g[fixed,1],"x",color="C3",ms=5,label="before fix")
    for s in sst: a.axvline(s/3600,color="r",lw=.6,ls="--")
    a.set_title(f"case {c}",fontsize=9); a.set_ylabel("Primus→SNUADC lag (ms)"); a.set_xlabel("hours")
handles={}
for a in ax.flat:
    for h,l in zip(*a.get_legend_handles_labels()): handles.setdefault(l,h)
ax.flat[0].legend(handles.values(),handles.keys(),fontsize=7)
fig.suptitle("Ventilator (Primus) lag from CVP after breath matching; red dashed = Primus stamp jumps",fontsize=11)
fig.tight_layout(); fig.savefig(sys.argv[3],dpi=65)
