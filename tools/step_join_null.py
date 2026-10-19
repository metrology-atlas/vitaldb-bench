# SPDX-License-Identifier: MIT
"""Null controls for join_z (tools/step_classify.py): at ordinary record junctions with no step, and with one message
or half a message removed. python tools/step_join_null.py ROOT 1,2,3 OUT.json (40 random cases are enough)."""
import sys
import numpy as np
import json
from pathlib import Path
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
R=Path(sys.argv[1]); cases=[int(c) for c in sys.argv[2].split(',')]
rng=np.random.default_rng(0)
out={}
def z(a,b):
    pred=2*a[-1]-a[-2]; e=np.abs(a[2:]-(2*a[1:-1]-a[:-2]))
    m=np.median(e); flat=e.size==0 or e.max()==0
    s=m if m>0 else (e.mean() if e.mean()>0 else 1.0)
    return abs(b[0]-pred)/s, flat
for c in cases:
    v=read_vital(R/'vital_files'/f'{c:04d}.vital')
    for n,tr in v.tracks.items():
        if tr.track_type!=1 or '#' in n or n.split('/')[0] not in ('SNUADC','Primus','BIS'): continue
        x=np.asarray(tr.values,float); sz=np.asarray(tr.record_sizes,int); st=np.r_[0,np.cumsum(sz)[:-1]]
        m=np.bincount(sz).argmax(); K=np.flatnonzero((sz[:-2]==m)&(sz[1:-1]==m)&(sz[2:]==m))
        if K.size<10: continue
        for k in rng.choice(K,min(200,K.size),replace=False):
            a=x[st[k]:st[k]+m]; b=x[st[k+1]:st[k+1]+m]; b2=x[st[k+2]:st[k+2]+m]
            z0,fl=z(a,b); z1,_=z(a,b2)
            d=out.setdefault(n,{'z0':[],'z1':[],'zh':[],'flat':0,'n':0})
            d['z0'].append(z0); d['z1'].append(z1); d['n']+=1; d['flat']+=fl
            h=m//2
            if h>=2: d['zh'].append(z(a,np.r_[b[h:],b2])[0])
res={}
for n,d in out.items():
    q=lambda v:[round(float(np.quantile(v,p)),2) for p in (.1,.5,.9)] if v else None
    res[n]=dict(n=d['n'],flat=int(d['flat']),ord_q=q(d['z0']),ord_lt3=round(float(np.mean(np.array(d['z0'])<3)),3),
      lost1_q=q(d['z1']),lost1_lt3=round(float(np.mean(np.array(d['z1'])<3)),3),lostHalf_lt3=round(float(np.mean(np.array(d['zh'])<3)),3) if d['zh'] else None)
json.dump(res,open(sys.argv[3],'w'),indent=1); print(json.dumps(res,indent=1))
