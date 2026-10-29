# SPDX-License-Identifier: MIT
"""Usage: python tools/primus_awp_sim_score.py   (writes sim2.json)

Experiment 2: scoring of candidate schemes (repeat share, observed lines, unobserved lines).
"""
import numpy as np, itertools, json
from primus_awp_sim_chain import chain, L, SNR, F2, G1, G2
def spec(rep):
    c=rep-rep.mean(); n=c.size; F=np.abs(np.fft.rfft(c,n=4*n)); f=np.fft.rfftfreq(4*n,1/F2); return f,F
def score(rep):
    f,S=spec(rep); keep=f>2; f,S=f[keep],S[keep]; top=S.max()
    near=np.zeros(f.size,bool); got=[]
    for l in L:
        m=np.abs(f-l)<0.0015; got.append(S[m].max()/top if m.any() else 0); near|=np.abs(f-l)<0.006
    out=S[~near].max()/top
    return np.array(got), out
cands=set()
for m in range(-6,7):
    for n in range(-6,7):
        for p in range(-3,4):
            x=m*G1+n*G2+p*F2
            if 20<x<130 and abs(m)+abs(n)<=6: cands.add(round(x,4))
cands=sorted(cands); res=[]
for combo in [[a] for a in cands]+[list(p) for p in itertools.permutations(cands,2) if p[1]>=0.5*p[0]]:
    rep=chain(combo); g,o=score(rep)
    res.append(dict(rates=combo, matched=int((g>0.1).sum()), purity=round(float(o),3), rep_share=round(float(rep.mean()),4),
                    corr=round(float(np.corrcoef(np.log(g+1e-3),np.log(SNR))[0,1]),2), rel=np.round(g,2).tolist()))
res.sort(key=lambda r:(r["matched"]-10*r["purity"]),reverse=True)
json.dump(res,open("sim2.json","w"))
for r in res[:20]: print(r)
