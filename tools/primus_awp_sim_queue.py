# SPDX-License-Identifier: MIT
"""Usage: python tools/primus_awp_sim_queue.py   (writes fifo.json)

Experiment 2: a source near 58 Hz sent in packets on its own timer and played out one sample per 16 ms from a queue.
"""
import numpy as np, json, itertools
L=np.array([4.2275,8.5215,12.357,12.749,16.585,20.8445,24.581,25.1055,28.941,29.333])
SNR=np.array([8.2,5.9,7.3,7.3,18.5,8.6,5.2,17.1,6.1,12.6]); F2=62.5; G1,G2=20.8445,25.1055; T=600.0
def fifo(fa, fp, maxq=None, ph=(0.37,0.71)):
    ts=np.arange(0,T,1/fa)+ph[0]/fa                     # sample instants (source clock)
    tp=np.arange(0,T+1,1/fp)+ph[1]/fp                   # packet send instants (packet timer)
    avail=tp[np.searchsorted(tp,ts,side="left")]        # each sample available at the next packet
    to=np.arange(0,T,1/F2)+0.5/F2
    out=np.empty(to.size,int); q=0; last=-1
    for j,t in enumerate(to):                           # consume one sample per output tick if available
        if q<ts.size and avail[q]<=t:
            last=q; q+=1
            if maxq is not None:
                n_av=np.searchsorted(avail,t,side="right")
                while n_av-q>maxq: q+=1; last=q-1
        out[j]=last
    o=out[out>=0]; return (np.diff(o)==0).astype(float)
def score(rep):
    c=rep-rep.mean(); n=c.size; F=np.abs(np.fft.rfft(c,n=4*n)); f=np.fft.rfftfreq(4*n,1/F2); k=f>2; f,F=f[k],F[k]; top=F.max()
    near=np.zeros(f.size,bool); got=[]
    for l in L:
        m=np.abs(f-l)<0.0015; got.append(F[m].max()/top); near|=np.abs(f-l)<0.006
    return np.array(got), float(F[~near].max()/top)
cands=set()
for m in range(-6,7):
    for n in range(-6,7):
        for p in range(-3,4):
            x=m*G1+n*G2+p*F2
            if 1<x<130 and abs(m)+abs(n)<=6: cands.add(round(x,4))
cands=sorted(cands); fas=[c for c in cands if 54<c<62.5]
print(len(cands), fas)
res=[]
for fa in fas:
    for fp in cands:
        for mq in (None,):
            rep=fifo(fa,fp,mq); g,o=score(rep)
            res.append(dict(fa=fa,fp=fp,maxq=mq,matched=int((g>0.1).sum()),purity=round(o,3),rep_share=round(float(rep.mean()),4),
                            corr=round(float(np.corrcoef(np.log(g+1e-3),np.log(SNR))[0,1]),2),rel=np.round(g,2).tolist()))
res.sort(key=lambda r:(r["matched"]-10*r["purity"]+3*r["corr"]),reverse=True)
json.dump(res,open("fifo.json","w"))
for r in res[:12]: print(r)
