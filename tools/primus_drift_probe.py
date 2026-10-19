# SPDX-License-Identifier: MIT
"""Primus clock drift from breath matching, and what happens at its AWP pauses. Per recording, this machine only.

    python tools/primus_drift_probe.py CASES OUT.jsonl        (CASES comma-separated; stops after 150 s, rerun to resume)

For recordings with Primus AWP and SNUADC CVP: the AWP-to-CVP lag in 10 min windows (examples/ppv_by_breath.py
cvp_lag; SNUADC on the stamp model, so on the recorder PC's timeline), the Primus AWP pauses (stamp gaps > 5 s), and
two fits of the lag against time: one line, and one common slope with one offset per stretch between pauses. Writes
per recording the slopes (ppm, Primus against the SNUADC stamp model), the residual SD of each fit, and per pause the
observed step in the fitted lag against the step a reset of the stretch's accumulated drift would give.
"""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import ppv_by_breath as P
from pat import sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
from pathlib import Path
root=Path.home()/'mnt/vitalDB/vitaldb-a-high-fidelity-multi-parameter-vital-signs-database-in-surgical-patients-1.0.0'
W=Path.home()/'mnt/vitalDB/work'
modes={int(k):v for k,v in json.load(open(W/'results/sync_full/snuadc_mode_label.json')).items()}
ports={int(k):(v or {}).get('port') for k,v in json.load(open(W/'results/sync_full/snuadc_mode.json')).items()}
rates=json.load(open(W/'results/pat/snuadc_step_rates.json'))
out=open(sys.argv[2],'a'); t0=time.time()
for c in map(int, sys.argv[1].split(',')):
    if time.time()-t0>150: break
    try:
        v=read_vital(root/'vital_files'/f'{c:04d}.vital', track_names={'Primus/AWP','SNUADC/CVP'})
        aw,cv=v.tracks.get('Primus/AWP'),v.tracks.get('SNUADC/CVP')
        if aw is None or cv is None or not cv.record_times.size: continue
        ta=sample_times(aw); xa=np.nan_to_num(aw.physical().astype(float))
        on=P.breath_onsets(ta,xa,float(aw.srate))
        if on.size<200: continue
        tc=P.model_times(cv,ports.get(c),modes.get(c),rates)
        tl,ll=P.cvp_lag(ta,xa,tc,cv.physical().astype(float),on)
        if tl.size<10: continue
        z=tl[0]; tl=tl-z; g=np.diff(ta); pauses=ta[1:][g>5]-z
        st=np.searchsorted(pauses,tl)          # stretch index per window
        # single line
        b1=np.polyfit(tl,ll,1)[0]
        # common slope, offset per stretch
        u=np.unique(st); X=np.column_stack([tl]+[(st==s).astype(float) for s in u])
        beta,*_=np.linalg.lstsq(X,ll,rcond=None); b=beta[0]; off=dict(zip(u,beta[1:]))
        jumps=[]
        for k in range(len(u)-1):
            s0,s1=u[k],u[k+1]
            if s1==s0+1:
                p=pauses[s0]
                # lag just before vs just after the pause, from the fits
                before=off[s0]+b*p; after=off[s1]+b*p
                L=p-(pauses[s0-1] if s0>0 else tl[st==s0].min())
                jumps.append([float(after-before), float(-b*L)])
        res_single=float(np.std(ll-np.polyval(np.polyfit(tl,ll,1),tl))); res_str=float(np.std(ll-X@beta))
        out.write(json.dumps({'case':c,'n':int(tl.size),'pauses':int(pauses.size),'slope_single_ppm':b1*1e6,'slope_common_ppm':b*1e6,
                   'res_single_ms':res_single*1000,'res_stretch_ms':res_str*1000,'jumps':jumps,'tl':np.round(tl,1).tolist(),'ll':np.round(ll,4).tolist(),'p':np.round(pauses,1).tolist()})+'\n'); out.flush()
    except Exception:
        pass
