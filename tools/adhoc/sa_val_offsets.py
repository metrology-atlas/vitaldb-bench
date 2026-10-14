# SPDX-License-Identifier: MIT
"""Per-case offsets left after the stamp model, by synchronisation mode (sync_full/units.csv, corr191all.jsonl, sa191.jsonl).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,csv,os,numpy as np
U={int(r["case"]):r for r in csv.DictReader(open(os.path.join(os.environ['VITALDB_WORK'], 'results/sync_full/units.csv')))}
cor={json.loads(l)["case"]:json.loads(l) for l in open('corr191all.jsonl')}
raw={json.loads(l)["case"]:json.loads(l) for l in open('sa191.jsonl')}
def good(rows):
    a=np.array(rows) if rows else np.empty((0,5))
    if not a.size: return a
    g=a[(a[:,2]>=0.3)&(a[:,2]-a[:,3]>=0.15)]; return g[(g[:,1]>-2500)&(g[:,1]<500)]
def mode(c):
    k=cor[c]["model"]["clocks"]["ecg"]; return "step" if k["rate_ppm"]<-20 else "small"
W={};R={}
for c in cor:
    g=good(cor[c]["model"]["burst"].get("ecg~eeg1",[])); 
    if len(g): W[c]=g[:,1]
    r=good(raw[c]["burst"].get("ecg~eeg1",[]))
    if len(r): R[c]=r[:,1]
med={c:np.median(v) for c,v in W.items()}
def err_with(const_fn, data):
    e=[]
    for c,v in data.items(): e+=list(np.abs(v-const_fn(c)))
    return np.percentile(e,[50,90,99]).round()
glob=np.median(list(med.values()))
print("windows",sum(len(v) for v in W.values()),"cases",len(W))
print("raw stamps, no correction: |lag| (error if ignored) p50,90,99", np.percentile(np.abs(np.concatenate(list(R.values()))),[50,90,99]).round())
print("raw stamps minus global constant %.0f:"%np.median(np.concatenate(list(R.values()))), err_with(lambda c: np.median(np.concatenate(list(R.values()))),R))
print("model, minus global constant %.0f:"%glob, err_with(lambda c: glob, W))
# leave-one-out per mode
def loo_mode(c):
    m=mode(c); v=[med[k] for k in med if k!=c and mode(k)==m]; return np.median(v)
print("model, minus leave-one-out mode constant:", err_with(loo_mode,W), {m:round(float(np.median([med[k] for k in med if mode(k)==m]))) for m in ("step","small")}, {m:sum(mode(k)==m for k in med) for m in ("step","small")})
def loo_mode_unit(c):
    m=mode(c); u=U[c]["bis_unit"]; v=[med[k] for k in med if k!=c and mode(k)==m and U[k]["bis_unit"]==u and u]
    return np.median(v) if len(v)>=3 else loo_mode(c)
print("model, minus leave-one-out mode+BIS-unit constant:", err_with(loo_mode_unit,W))
print("model, minus own case constant (bursts available):", err_with(lambda c: med[c],W))
