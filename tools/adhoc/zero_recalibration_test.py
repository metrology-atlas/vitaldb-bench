# SPDX-License-Identifier: MIT
"""Does zeroing recalibrate slow drift? Bootstrap of the line-minus-cuff slope before and after zeroing (zdrift.jsonl from zero_calibration_drift).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,sys,numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
E=[json.loads(l) for l in open('zdrift.jsonl')]
rng=np.random.default_rng(0)
def boot(p,cases,n=2000):
    uc=np.unique(cases); idx={c:np.flatnonzero(cases==c) for c in uc}; out=[]
    for _ in range(n):
        s=np.concatenate([idx[c] for c in rng.choice(uc,uc.size)])
        q=p[s]; out.append(np.polyfit(q[:,0],q[:,1],1)[0] if np.std(q[:,0])>0 else np.nan)
    return np.nanpercentile(out,[2.5,97.5])
res={}
for k,col in (("zero","art_dmap_20s"),("zero","art_dmap_1min"),("flush","art_dmap_20s"),("random","art_dmap_1min")):
    e=[x for x in E if x["kind"]==k and col in x]; p=np.array([(x[col],x["step"]) for x in e]); cs=np.array([x["case"] for x in e])
    sl=np.polyfit(p[:,0],p[:,1],1)[0]; ci=boot(p,cs); res[(k,col)]=(p,sl,ci)
    print(f"{k:6s} {col:13s} n {len(p)} cases {np.unique(cs).size} slope {sl:+.2f} 95% CI (case bootstrap) {ci[0]:+.2f}..{ci[1]:+.2f}")
# suspect zeros
e=[x for x in E if x["kind"]=="zero" and "art_ddbp_20s" in x and abs(x["art_ddbp_20s"])>7]
p=np.array([(x["art_dmap_20s"],x["step"]) for x in e]); cs=np.array([x["case"] for x in e])
print(f"suspect zeros (|dDBP|>7): n {len(p)} slope {np.polyfit(p[:,0],p[:,1],1)[0]:+.2f} CI {boot(p,cs)}")
fig,ax=plt.subplots(1,2,figsize=(13,5))
for a_,(k,col,lab) in zip(ax,(("zero","art_dmap_20s","zeroings (line jump in 20 s windows)"),("random","art_dmap_1min","random times (line change, 1-min windows)"))):
    p,sl,ci=res[(k,col)]
    a_.scatter(p[:,0],p[:,1],s=6,alpha=.4)
    xs=np.array([-30,30]); a_.plot(xs,xs,"k--",lw=.8,label="pure recalibration (slope 1)"); a_.plot(xs,sl*xs,"C3",label=f"fit slope {sl:+.2f} (95% CI {ci[0]:+.2f}..{ci[1]:+.2f})")
    a_.axhline(0,color="0.6",lw=.5); a_.set_xlim(-30,30); a_.set_ylim(-30,30)
    a_.set_xlabel("ART MAP change at the event (mmHg)"); a_.set_ylabel("change in line − cuff MAP across the event (mmHg)")
    a_.set_title(f"{lab}: n {len(p)}",fontsize=10); a_.legend(fontsize=8)
fig.suptitle("Does the line move away from the cuff at zeroing? (1,440 cases)",fontsize=12); fig.tight_layout(); fig.savefig(sys.argv[1],dpi=80)
