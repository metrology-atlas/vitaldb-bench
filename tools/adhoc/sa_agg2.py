# SPDX-License-Identifier: MIT
"""Shared-artifact offsets between channel pairs (argv[1]): EEG1-EEG2 and ECG-EEG differences per window and per case.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,sys,numpy as np
rs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
def good(rows):
    a=np.array(rows); 
    if not a.size: return {}
    g=a[(a[:,2]>=0.3)&(a[:,2]-a[:,3]>=0.15)]
    return {float(x[0]):x for x in g}
D12=[];DC=[];DC_ee=[];allw=[];sd_case=[]
for r in rs:
    b=r.get("burst",{})
    g1,g2=good(b.get("ecg~eeg1",[])),good(b.get("ecg~eeg2",[]))
    ee=good(b.get("eeg1~eeg2",[]))
    both=sorted(set(g1)&set(g2))
    D12+= [g1[t][1]-g2[t][1] for t in both]
    ts=sorted(g1)
    for i in range(1,len(ts)):
        if ts[i]-ts[i-1]==300: DC.append(g1[ts[i]][1]-g1[ts[i-1]][1])
    te=sorted(ee)
    for i in range(1,len(te)):
        if te[i]-te[i-1]==300: DC_ee.append(ee[te[i]][1]-ee[te[i-1]][1])
    if len(g1)>=5: sd_case.append(np.std([g1[t][1] for t in ts]))
    allw+= [g1[t][1] for t in ts]
q=lambda x:[round(float(v),1) for v in np.percentile(np.abs(x),[50,90])] if len(x) else None
print("ecg~eeg1 vs ecg~eeg2 same window |diff| p50,p90:",q(D12),len(D12))
print("ecg~eeg1 consecutive windows (overlapping 50%) |diff|:",q(DC),len(DC))
print("eeg1~eeg2 consecutive |diff|:",q(DC_ee),len(DC_ee))
print("within-case SD of ecg~eeg1 lag (cases>=5 good) median, range:",round(float(np.median(sd_case)),1),round(min(sd_case),1),round(max(sd_case),1))
print("all good-window lags percentiles 5,25,50,75,95:",np.percentile(allw,[5,25,50,75,95]).round(0),len(allw))
