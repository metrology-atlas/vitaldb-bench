# SPDX-License-Identifier: MIT
"""Summarise shared_artifacts output (argv[1]): good burst windows per channel pair (ECG, EEG, ART, PLETH) and the ventilator pair.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,sys,numpy as np
rs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print("cases",len(rs),"errors",[ (r['case'],r['error']) for r in rs if 'error' in r])
def summ(rows,rmin=0.3,marg=0.15,col_r=2,col_s=3):
    a=np.array(rows) if rows else np.empty((0,5))
    if not a.size: return None
    g=a[(a[:,col_r]>=rmin)&(a[:,col_r]-a[:,col_s]>=marg)]
    out={"win":len(a),"good":len(g)}
    if len(g):
        out.update(med=round(float(np.median(g[:,1])),1),iqr=round(float(np.subtract(*np.percentile(g[:,1],[75,25]))),1),
                   rng=round(float(np.ptp(g[:,1])),1),hrs=round(float(np.ptp(g[:,0]))/3600,2))
        if len(g)>=4: out["slope_ms_h"]=round(float(np.polyfit(g[:,0]/3600,g[:,1],1)[0]),1)
    return out
for pair in ["ecg~eeg1","ecg~eeg2","eeg1~eeg2","ecg~art","ecg~pleth","art~eeg1"]:
    print("==",pair)
    for r in rs:
        s=summ(r.get("burst",{}).get(pair))
        if s and s["good"]: print(r["case"],s)
    print("cases with >=1 good:",sum(1 for r in rs if (summ(r.get("burst",{}).get(pair)) or {}).get("good")))
print("== vent")
for r in rs:
    v=r.get("vent",{})
    if "rows" in v:
        s=summ(v["rows"],0.5,0.0); print(r["case"],v["ref"],s)
    else: print(r["case"],v)
print("== hum snr medians / pairs")
for r in rs:
    h=r.get("hum",{}); print(r["case"],h.get("snr_median"),{k:(v.get("blocks"),v.get("resultant"),v.get("jitter_ms_mad")) for k,v in h.items() if k!="snr_median"})
