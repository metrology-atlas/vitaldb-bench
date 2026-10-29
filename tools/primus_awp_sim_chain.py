# SPDX-License-Identifier: MIT
"""Usage: python tools/primus_awp_sim_chain.py   (writes sim_top.json in the current folder)

Experiment 2 of the AWP conversion study: rates held in series onto 62.5 Hz, scored against the 10 observed lines.
"""
import numpy as np, itertools, json
L=np.array([4.2275,8.5215,12.357,12.749,16.585,20.8445,24.581,25.1055,28.941,29.333])
SNR=np.array([8.2,5.9,7.3,7.3,18.5,8.6,5.2,17.1,6.1,12.6])
F2=62.5; T=600.0
G1,G2=20.8445,25.1055
def fold(f): f=np.mod(f,F2); return np.minimum(f,F2-f)
def chain(rates, T=T, ph=(0.37,0.71,0.13)):
    """source at rates[0]; each later rate holds the latest value of the previous stage; final output at 62.5 Hz.
    Returns repeat indicator of the output (same source sample as previous output)."""
    t=np.arange(0,T,1/rates[0])+ph[0]/rates[0]; idx=np.arange(t.size)
    for j,r in enumerate(list(rates[1:])+[F2]):
        to=np.arange(0,T,1/r)+(ph[j+1] if j+1<len(ph) else 0.5)/r
        k=np.searchsorted(t,to,side="right")-1; ok=k>=0
        t,idx=to[ok],idx[k[ok]]
    return (np.diff(idx)==0).astype(float)
def lines(rep):
    c=rep-rep.mean(); n=c.size; F=np.abs(np.fft.rfft(c,n=4*n)); f=np.fft.rfftfreq(4*n,1/F2)
    base=np.median(F[f>1]) + 1e-12
    return f,F/base
def score(rep):
    f,S=lines(rep); got=[]
    for l in L:
        m=np.abs(f-l)<0.004; got.append(S[m].max())
    got=np.array(got)
    # strongest simulated lines not in measured list
    o=np.argsort(S)[::-1]; extra=[]
    for i in o[:400]:
        if f[i]<2: continue
        if np.all(np.abs(L-f[i])>0.01) and all(abs(f[i]-e)>0.05 for e,_ in extra): extra.append((round(float(f[i]),3),round(float(S[i]),1)))
        if len(extra)>=3: break
    return got, extra, rep.mean()
if __name__=="__main__":
    cands=set()
    for m in range(-6,7):
        for n in range(-6,7):
            for p in range(-3,4):
                x=m*G1+n*G2+p*F2
                if 20<x<130 and abs(m)+abs(n)<=6: cands.add(round(x,4))
    cands=sorted(cands); print(len(cands),"candidate rates")
    out=[]
    # single stage
    for a in cands:
        g,e,s=score(chain([a])); out.append((int((g>4).sum()),[a],s,g.round(1).tolist(),e))
    # two stage
    for a,b in itertools.permutations(cands,2):
        if b<a*0.5: continue
        g,e,s=score(chain([a,b])); out.append((int((g>4).sum()),[a,b],s,g.round(1).tolist(),e))
    out.sort(key=lambda r:(r[0],-abs(r[2]-0.075)),reverse=True)
    for r in out[:25]: print(r)
    json.dump(out[:200],open("sim_top.json","w"))
