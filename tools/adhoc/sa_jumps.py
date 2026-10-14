# SPDX-License-Identifier: MIT
"""Shared-artefact offset jumps against record-stamp jumps in the same cases (argv[1] = shared_artifacts output, argv[2] = case list). Drift document: 'shared_artifacts and sa_jumps, 191 cases'.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,os,sys,numpy as np
from pathlib import Path
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
R=Path(os.environ["VITALDB_ROOT"]) / "vital_files"
rs={json.loads(l)["case"]:json.loads(l) for l in open(sys.argv[1])}
cases=[int(c) for c in sys.argv[2].split(",")]
def good(rows):
    a=np.array(rows); return a[(a[:,2]>=0.3)&(a[:,2]-a[:,3]>=0.15)] if a.size else np.empty((0,5))
def stamp_jumps(tr):
    rt=tr.record_times; off=tr.record_offsets[:rt.size]/float(tr.srate)
    res=rt-rt[0]-off           # stamp minus sample-count time
    d=np.diff(res); k=np.flatnonzero(np.abs(d)>0.03)
    return rt[0], [(float(rt[i+1]), float(d[i])) for i in k], res
import os
done=set()
if os.path.exists(sys.argv[3]): done={json.loads(l)["case"] for l in open(sys.argv[3])}
fh=open(sys.argv[3],"a")
for c in cases:
    if c in done: continue
    out=[]
    v=read_vital(R/f"{c:04d}.vital",track_names={"SNUADC/ECG_II","BIS/EEG1_WAV"})
    e,b=v.tracks["SNUADC/ECG_II"],v.tracks["BIS/EEG1_WAV"]
    t0=np.floor(min(e.record_times[0],b.record_times[0]))
    _,je,re_=stamp_jumps(e); _,jb,rb=stamp_jumps(b)
    if "ecg~eeg1" not in rs[c].get("burst",{}): continue
    g=good(rs[c]["burst"]["ecg~eeg1"]); g=g[np.argsort(g[:,0])]
    for i in range(1,len(g)):
        dl=g[i,1]-g[i-1,1]
        if dl>60 and g[i,0]-g[i-1,0]<=900:
            a,bb=t0+g[i-1,0]-300,t0+g[i,0]+300
            se=[x for x in je if a<=x[0]<=bb]; sb=[x for x in jb if a<=x[0]<=bb]
            out.append(dict(case=c,lag_jump=round(dl),ecg_steps=[round(x[1]*1000) for x in se],eeg_steps=[round(x[1]*1000) for x in sb],t=float(g[i,0])))
    fh.write(json.dumps(dict(case=c,jumps=out,ecg_steps=len(je),eeg_steps=len(jb),ecg_step_med=float(np.median([x[1] for x in je])) if je else None,eeg_step_med=float(np.median([x[1] for x in jb])) if jb else None))+"\n"); fh.flush()
    if 0: print(c,"ECG stamp jumps",len(je),"sizes ms p50",np.median([x[1] for x in je])*1000 if je else None,"| EEG",len(jb),np.median([x[1] for x in jb])*1000 if jb else None,
          "| resid range ECG %.0f ms EEG %.0f ms"%(np.ptp(re_)*1000,np.ptp(rb)*1000))
