# SPDX-License-Identifier: MIT
"""Validate the stamp model with shared artefacts: raw against corrected offsets (sa191.jsonl, corr191all.jsonl, sync_full/units.csv).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,csv,os,numpy as np
raw={json.loads(l)["case"]:json.loads(l) for l in open('sa191.jsonl')}
cor={json.loads(l)["case"]:json.loads(l) for l in open('corr191all.jsonl')}
U={int(r["case"]):r for r in csv.DictReader(open(os.path.join(os.environ['VITALDB_WORK'], 'results/sync_full/units.csv')))}
def good(rows):
    a=np.array(rows) if rows else np.empty((0,5)); 
    if not a.size: return a
    g=a[(a[:,2]>=0.3)&(a[:,2]-a[:,3]>=0.15)]; return g[(g[:,1]>-2500)&(g[:,1]<500)]
def stats(g):
    g=g[np.argsort(g[:,0])]; m=np.median(g[:,1]); dev=np.abs(g[:,1]-m)
    sl=np.polyfit(g[:,0]/3600,g[:,1],1)[0]
    jumps=np.sum((np.diff(g[:,1])>60)&(np.diff(g[:,0])<=900))
    return dict(med=m,sd=np.std(g[:,1]),dev=dev,slope=sl,jumps=jumps,n=len(g))
out={k:[] for k in ("raw","teeth","syncfull","model")}; devs={k:[] for k in out}; per={}
errs=[c for c,r in cor.items() if 'error' in r]
for c in raw:
    if c not in cor or 'error' in cor[c]: continue
    gs={"raw":good(raw[c]["burst"].get("ecg~eeg1",[])),"teeth":good(cor[c]["teeth"]["burst"].get("ecg~eeg1",[])),"syncfull":good(cor[c]["syncfull"]["burst"].get("ecg~eeg1",[])),"model":good(cor[c]["model"]["burst"].get("ecg~eeg1",[]))}
    if min(len(g) for g in gs.values())<5: continue
    for k,g in gs.items():
        s=stats(g); out[k].append((s["sd"],abs(s["slope"]),s["jumps"],s["med"])); devs[k]+=list(s["dev"])
    per[c]=stats(gs["model"])
print("errors",len(errs),errs[:5]); print("cases compared",len(per))
for k,v in out.items():
    a=np.array(v); d=np.array(devs[k])
    print(k,"within-case SD med %.0f (IQR %.0f-%.0f) | |slope| med %.0f ms/h | jumps/case mean %.2f | |dev from case median| p50 %.0f p90 %.0f p99 %.0f ms"%(np.median(a[:,0]),*np.percentile(a[:,0],[25,75]),np.median(a[:,1]),a[:,2].mean(),*np.percentile(d,[50,90,99])))
# between cases: constant offset
m=np.array([p["med"] for p in per.values()]); print("corrected case offsets ms p5,25,50,75,95",np.percentile(m,[5,25,50,75,95]).round())
def setup(c):
    u=U.get(c,{}); return u.get('snuadc_port','')+(" step" if u.get('snuadc_drift_ppm') and float(u['snuadc_drift_ppm'])<-20 else " small")
by={}
for c,p in per.items(): by.setdefault(setup(c),[]).append(p["med"]); by.setdefault("BIS "+U.get(c,{}).get('bis_unit',''),[]).append(p["med"])
for k,v in sorted(by.items(),key=lambda x:-len(x[1])):
    if len(v)>=5: print(k,len(v),"median %.0f IQR %.0f"%(np.median(v),np.subtract(*np.percentile(v,[75,25]))))
# worst cases
w=sorted(per.items(),key=lambda x:-x[1]["sd"])[:8]
for c,p in w: print("worst",c,setup(c),"sd %.0f slope %.0f jumps %d n %d"%(p["sd"],p["slope"],p["jumps"],p["n"]),cor[c]["model"]["clocks"]["ecg"],cor[c]["model"]["clocks"]["eeg1"])
src={}
for c,r in cor.items():
    if 'error' in r: continue
    for k in ("ecg","eeg1"):
        s=r["model"]["clocks"][k]["source"]; src[(k,s)]=src.get((k,s),0)+1
print(src)
