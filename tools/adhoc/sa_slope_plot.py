# SPDX-License-Identifier: MIT
"""Figure: ramp slope from shared bursts against the slope predicted from the BIS and SNUADC drift (argv[1] tuples).

Ad hoc helper from the VitalDB timing audit, kept for provenance (see tools/adhoc/README.md). Reads
per-case files that stay on the analysis machine; not maintained as a tool.
"""
import sys,ast,subprocess,numpy as np,matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows=[ast.literal_eval(l) for l in open(sys.argv[1]) if l.startswith('(')]
a=np.array([(r[1],r[2]) for r in rows])
fig,ax=plt.subplots(1,2,figsize=(12,5))
ax[0].scatter(a[:,1],a[:,0],s=12); ax[0].plot([-160,40],[-160,40],"k--",lw=.8)
ax[0].set_xlim(-160,40); ax[0].set_ylim(-160,40)
ax[0].set_xlabel("predicted −(BIS − SNUADC drift), ppm (sync_full)"); ax[0].set_ylabel("ramp slope from shared bursts, ppm")
k=np.abs(a[:,0]-a[:,1])<=25
ax[0].set_title(f"{len(a)} cases; {k.sum()} within 25 ppm (r={np.corrcoef(a[k].T)[0,1]:.2f}); {np.sum(a[:,1]>40)} off-scale predictions",fontsize=9)
ax[1].bar(["lag jump:\nstep within window","same interval\n40 min later"],[170/216*100,26/216*100],color=["C0","0.6"])
ax[1].set_ylabel("% of 216 lag jumps with a stamp step (ECG or EEG)"); ax[1].set_ylim(0,100)
ax[1].set_title("Lag jumps coincide with record-stamp steps",fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[2],dpi=80)
