# SPDX-License-Identifier: MIT
"""Figure: shared-artefact offsets over time for the eight cases with most good ECG-EEG1 windows (argv[1] output, argv[3] units.csv).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,sys,csv,numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
U={int(r["case"]):r for r in csv.DictReader(open(sys.argv[3]))}
def good(rows):
    a=np.array(rows)
    return a[(a[:,2]>=0.3)&(a[:,2]-a[:,3]>=0.15)] if a.size else np.empty((0,5))
cs=sorted(rs,key=lambda r:-len(good(r["burst"].get("ecg~eeg1",[]))))[:8]
fig,ax=plt.subplots(3,4,figsize=(20,11))
for i,r in enumerate(cs):
    a=ax.flat[i]
    for k,c,m in [("ecg~eeg1","C0","o"),("ecg~eeg2","C1","x")]:
        g=good(r["burst"].get(k,[])); a.plot(g[:,0]/3600,g[:,1],m,color=c,ms=4,label=k)
    g=good(r["burst"].get("eeg1~eeg2",[])); a.plot(g[:,0]/3600,g[:,1]-750,".",color="0.6",ms=3,label="eeg1~eeg2 (-750 ms)")
    u=U.get(r["case"],{})
    a.set_title(f"case {r['case']}  SNUADC {u.get('snuadc_drift_ppm','?')} ppm, BIS {u.get('bis_drift_ppm','?')} ppm",fontsize=9)
    a.set_ylim(-1300,-200); a.set_xlabel("hours"); a.set_ylabel("EEG lag vs ECG (ms)")
    if i==0:a.legend(fontsize=7)
allw=np.concatenate([good(r["burst"].get("ecg~eeg1",[]))[:,1] for r in rs])
nout=int(np.sum((allw<-2000)|(allw>0))); ax.flat[8].hist(allw[(allw>=-2000)&(allw<=0)],bins=np.arange(-2000,1,40)); ax.flat[8].set_title(f"all good windows, ECG~EEG1 ({len(allw)} windows, {nout} outside -2..0 s, {sum(1 for r in rs if len(good(r['burst'].get('ecg~eeg1',[]))))} cases)",fontsize=9); ax.flat[8].set_xlabel("lag (ms)")
ee=np.concatenate([good(r["burst"].get("eeg1~eeg2",[]))[:,1] for r in rs])
ax.flat[9].hist(ee,bins=np.arange(-60,61,2)); ax.flat[9].set_title("control: EEG1~EEG2 (same device)",fontsize=9); ax.flat[9].set_xlabel("lag (ms)")
# per-window precision
D=[]
for r in rs:
    g1={x[0]:x[1] for x in good(r["burst"].get("ecg~eeg1",[]))}; g2={x[0]:x[1] for x in good(r["burst"].get("ecg~eeg2",[]))}
    D+=[g1[t]-g2[t] for t in set(g1)&set(g2)]
ax.flat[10].hist(D,bins=np.arange(-100,101,4)); ax.flat[10].set_title("same window: (ECG~EEG1) - (ECG~EEG2)",fontsize=9); ax.flat[10].set_xlabel("ms")
ax.flat[11].axis("off")
ax.flat[11].text(0,0.5,"Negative lag: the same burst is stamped\nearlier in BIS EEG than in SNUADC ECG.\nWindows 10 min, step 5 min; good = r>=0.3\nand margin>=0.15 over lags >1 s away.\n"+f"{len(rs)} cases with BIS EEG.",fontsize=10)
fig.suptitle("Shared broadband bursts (electrocautery/EMG) in SNUADC ECG and BIS EEG: stamp offset between devices",fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[2],dpi=72)
