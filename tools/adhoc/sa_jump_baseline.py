# SPDX-License-Identifier: MIT
"""Baseline for sa_jumps: how often shared-artefact offset jumps coincide with stamp jumps, against random times (jumps191.jsonl, sa191.jsonl).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import os
import json,sys,numpy as np
from pathlib import Path
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
R=Path(os.environ["VITALDB_ROOT"]) / "vital_files"
J=[json.loads(l) for l in open('jumps191.jsonl')]
rs={json.loads(l)["case"]:json.loads(l) for l in open('sa191.jsonl')}
hit=[0,0];hitR=[0,0];tight=[0,0];tightR=[0,0]
for r in J[: int(sys.argv[1])]:
    if not r["jumps"]: continue
    c=r["case"]; v=read_vital(R/f"{c:04d}.vital",track_names={"SNUADC/ECG_II","BIS/EEG1_WAV"})
    st=[]
    for n in ("SNUADC/ECG_II","BIS/EEG1_WAV"):
        tr=v.tracks[n]; rt=tr.record_times; res=rt-rt[0]-tr.record_offsets[:rt.size]/tr.srate
        d=np.diff(res); k=np.flatnonzero((np.abs(d)>0.03)&(np.abs(d)<2)); st+=list(rt[k+1])
    st=np.array(st); t0=np.floor(min(v.tracks[n].record_times[0] for n in ("SNUADC/ECG_II","BIS/EEG1_WAV")))
    g=np.array(rs[c]["burst"]["ecg~eeg1"]); 
    for j in r["jumps"]:
        for sh,H,T in ((0,hit,tight),(2400,hitR,tightR)):
            a=t0+j["t"]-600+sh; b=t0+j["t"]+300+sh
            H[0]+=np.any((st>=a)&(st<=b)); H[1]+=1
            T[0]+=np.any((st>=t0+j["t"]-300+sh)&(st<=t0+j["t"]+sh)); T[1]+=1
print("wide interval: lag jumps with a step",hit,"same width 40 min later",hitR)
print("tight (5 min before window centre):",tight,"40 min later",tightR)
