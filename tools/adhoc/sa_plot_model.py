# SPDX-License-Identifier: MIT
"""Figure: shared-artefact offsets before and after the stamp model (corr191all.jsonl, sa191.jsonl) for six example cases.

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import json,csv,os,sys,numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cor={json.loads(l)["case"]:json.loads(l) for l in open('corr191all.jsonl')}
raw={json.loads(l)["case"]:json.loads(l) for l in open('sa191.jsonl')}
def good(rows):
    a=np.array(rows) if rows else np.empty((0,5))
    if not a.size: return a
    g=a[(a[:,2]>=0.3)&(a[:,2]-a[:,3]>=0.15)]; return g[(g[:,1]>-2500)&(g[:,1]<500)]
cases=[4644,359,2580,5780,750,1703]
fig=plt.figure(figsize=(20,11)); gs=fig.add_gridspec(3,4)
for i,c in enumerate(cases):
    a=fig.add_subplot(gs[i//3, i%3])
    r=good(raw[c]["burst"]["ecg~eeg1"]); m=good(cor[c]["model"]["burst"]["ecg~eeg1"])
    a.plot(r[:,0]/3600,r[:,1],"o",color="0.7",ms=4,label="recorded stamps")
    a.plot(m[:,0]/3600,m[:,1],"o",color="C0",ms=4,label="stamp model")
    k=cor[c]["model"]["clocks"]
    a.set_title(f"case {c}: SNUADC {k['ecg']['rate_ppm']:.0f} ppm ({k['ecg']['source']}), BIS {k['eeg1']['rate_ppm']:.0f} ppm ({k['eeg1']['source']})",fontsize=9)
    a.set_ylim(-1300,-300); a.set_xlabel("hours"); a.set_ylabel("EEG minus ECG (ms)")
    if i==0: a.legend(fontsize=8)
# error ladder
a=fig.add_subplot(gs[0:2,3])
labels=["recorded\nstamps","stamps −\nconstant","model −\nconstant","model −\nmode const.","model −\ncase const."]
p50=[792,126,38,33,16]; p90=[1005,259,154,116,67]
x=np.arange(5); a.bar(x-0.2,p50,0.4,label="median"); a.bar(x+0.2,p90,0.4,label="90th percentile")
for xi,(u,v) in enumerate(zip(p50,p90)): a.text(xi-0.2,u+15,str(u),ha="center",fontsize=8); a.text(xi+0.2,v+15,str(v),ha="center",fontsize=8)
a.set_xticks(x); a.set_xticklabels(labels,fontsize=8); a.set_ylabel("|EEG−ECG timing error| per 10-min window (ms)"); a.set_ylim(0,1100); a.legend(fontsize=8)
a.set_title("Timing error by correction level (3,683 windows, 165 cases)",fontsize=9)
# offsets by mode
a=fig.add_subplot(gs[2,0:2])
med={c:np.median(good(cor[c]["model"]["burst"].get("ecg~eeg1",[]))[:,1]) for c in cor if len(good(cor[c]["model"]["burst"].get("ecg~eeg1",[])))>=5}
mode=lambda c:"step" if cor[c]["model"]["clocks"]["ecg"]["rate_ppm"]<-20 else "small"
for m,col in (("step","C0"),("small","C1")):
    v=[med[c] for c in med if mode(c)==m]; a.hist(v,bins=np.arange(-1000,-450,10),alpha=.7,color=col,label=f"SNUADC {m} mode ({len(v)} cases, median {np.median(v):.0f} ms)")
a.set_xlabel("constant EEG−ECG offset left after the model, per case (ms)"); a.legend(fontsize=8); a.set_ylabel("cases")
# within-case SD
a=fig.add_subplot(gs[2,2:4])
sdr=[];sdm=[]
for c in med:
    r=good(raw[c]["burst"]["ecg~eeg1"]); m=good(cor[c]["model"]["burst"]["ecg~eeg1"])
    if len(r)>=5: sdr.append(np.std(r[:,1])); sdm.append(np.std(m[:,1]))
a.hist(sdr,bins=np.arange(0,300,10),alpha=.6,color="0.6",label=f"recorded stamps (median {np.median(sdr):.0f} ms)")
a.hist(sdm,bins=np.arange(0,300,10),alpha=.7,color="C0",label=f"stamp model (median {np.median(sdm):.0f} ms)")
a.set_xlabel("within-case SD of the EEG−ECG offset (ms)"); a.set_ylabel("cases"); a.legend(fontsize=8)
fig.suptitle("Stamp sawtooth model (physiosync.stamp_clock) validated against shared ECG/EEG bursts",fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[1],dpi=72)
