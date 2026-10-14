# SPDX-License-Identifier: MIT
"""Figure: CVP-based Primus lag over time per case (argv[1] = vent_sync_pilot output, argv[2] = cases).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,sys,numpy as np,matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rs={json.loads(l)["case"]:json.loads(l) for l in open(sys.argv[1])}
cases=[int(x) for x in sys.argv[2].split(",")]
fig,ax=plt.subplots(len(cases)//2,2,figsize=(18,2.8*len(cases)//2))
for a,c in zip(ax.flat,cases):
    r=rs[c]; cv=np.array(r["rows"]["cvp"],dtype=float)
    if cv.size:
        g=(cv[:,2]>=0.8)&(cv[:,3]<100)
        a.plot(cv[g,0]/3600,cv[g,1],"o",ms=4,color="C0",label="CVP (good)")
        a.plot(cv[~g,0]/3600,cv[~g,1],".",ms=3,color="0.75",label="CVP (other)")
        ref=np.median(cv[g,1]) if g.any() else 0
    else: ref=0
    for k,col in (("ramp","C1"),("sbp","C2")):
        x=np.array(r["rows"].get(k,[]),dtype=float)
        if not x.size: continue
        x=x[x[:,2]>=0.05]
        if not x.size: continue
        T=np.median(x[:,4])*1000; lag=np.unwrap(x[:,1]/T*2*np.pi)/(2*np.pi)*T
        a.plot(x[:,0]/3600,lag-np.median(lag)+ref,"^" if k=="ramp" else "s",ms=3,color=col,alpha=.7,label=f"{k} (shifted)")
    for s in r.get("stretch_starts_s",[]): a.axvline(s/3600,color="r",lw=.6,ls="--")
    a.set_title(f"case {c}: CVP fit {r['fits'].get('cvp',{}).get('ppm',float('nan')):.0f} ppm; red = Primus stamp jumps >1 s",fontsize=9)
    a.set_ylabel("Primus→SNUADC lag (ms)"); a.set_xlabel("hours")
ax.flat[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(sys.argv[3],dpi=65)
