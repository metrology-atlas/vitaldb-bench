# SPDX-License-Identifier: MIT
"""Two small tests on one recording.
(a) Arterial sawtooth: does the ECG->ART delay follow the slow phase of the 240 Hz cycle seen in ART?
    psi(t) = phase of ART's step-size pattern demodulated at exactly 240 Hz on the SNUADC sample count, in 10 s blocks.
    If ART is taken on its own ~240 Hz clock and placed on the 500 Hz grid without timing correction, its content
    lags or leads by up to one 240 Hz period (4.17 ms) as psi turns, so per-beat delay (detrended) vs psi is a ramp.
    Control: the same binning with psi shifted by a quarter of the record (wrong alignment).
(b) CO2 grid: change pattern of Primus/CO2 (value differs from previous) demodulated at 12.5 Hz (every 5 samples at
    62.5 Hz) in 60 s blocks. Locked -> constant phase; drifting -> phase walks at a steady rate.

Usage: python tools/art_cycle_and_co2_grid.py ROOT OUTDIR CASE [MINUTES]   (appends OUTDIR/saw2.jsonl, saves OUTDIR/c<case>.npz)
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy import signal
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital
root, outd, case = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]); MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 90
FS = 500.0
v = read_vital(root / "vital_files" / f"{case:04d}.vital", track_names={"SNUADC/ECG_II", "SNUADC/ART", "Primus/CO2"})
res = {"case": case}
def arr(k):
    t = v.tracks.get(k); return None if t is None or not t.record_times.size else np.asarray(t.values, float)
ecg, art, co2 = arr("SNUADC/ECG_II"), arr("SNUADC/ART"), arr("Primus/CO2")
save = {}
if ecg is not None and art is not None:
    n = min(ecg.size, art.size); n0 = int(FS * 600); N = min(n - n0, int(FS * 60 * MIN))
    e, a = ecg[n0:n0 + N], art[n0:n0 + N]
    ok = np.isfinite(e) & np.isfinite(a); e = np.where(ok, e, 0); a = np.where(ok, a, 0)
    # 240 Hz slow phase in 10 s blocks
    d = np.abs(np.diff(a)); r = d / (np.convolve(d, np.ones(25) / 25, "same") + 1e-9); r = r - np.median(r)
    B = int(FS * 10); nb = r.size // B; k = np.arange(r.size)
    z = (r * np.exp(-2j * np.pi * 240 * k / FS))[:nb * B].reshape(nb, B).mean(1)
    good = ok[:nb * B].reshape(nb, B).all(1) & (np.abs(z) > 3 * np.median(np.abs(z)) / 3)
    psi = np.unwrap(np.angle(z)); tb = (np.arange(nb) + 0.5) * 10
    # R peaks and ART upstrokes
    sos = signal.butter(2, (5, 25), "bandpass", fs=FS, output="sos"); ef = signal.sosfiltfilt(sos, e)
    if np.abs(ef.min()) > ef.max(): ef = -ef
    pk, _ = signal.find_peaks(ef, distance=int(0.33 * FS), height=np.percentile(ef, 99) * 0.4)
    sa = signal.butter(2, (0.5, 15), "bandpass", fs=FS, output="sos"); da = np.gradient(signal.sosfiltfilt(sa, a))
    tr, dl = [], []
    for p in pk:
        lo, hi = p + int(0.08 * FS), p + int(0.45 * FS)
        if hi >= da.size - 1 or not ok[p:hi].all(): continue
        m = lo + int(np.argmax(da[lo:hi]))
        if m in (lo, hi - 1) or da[m] <= 0: continue
        den = da[m - 1] - 2 * da[m] + da[m + 1]; frac = np.clip(0.5 * (da[m - 1] - da[m + 1]) / den, -.5, .5) if den else 0
        tr.append(p / FS); dl.append((m + frac - p) / FS * 1000)
    tr, dl = np.array(tr), np.array(dl)
    # detrend: 3 min running median (sawtooth expected period ~1-3 min would be partly removed by longer windows
    # only if shorter; we use psi binning so the residual ramp survives as long as period < window)
    base = np.array([np.median(dl[np.abs(tr - t) < 90]) for t in tr]); res_ = dl - base
    keep = np.abs(res_) < 15; tr, dl, res_ = tr[keep], dl[keep], res_[keep]
    psib = np.interp(tr, tb, psi); ph = (psib / (2 * np.pi)) % 1
    edges = np.linspace(0, 1, 13)
    prof = np.array([np.mean(res_[(ph >= edges[i]) & (ph < edges[i + 1])]) for i in range(12)])
    psic = np.interp((tr + tr.max() / 4) % tr.max(), tb, psi); phc = (psic / (2 * np.pi)) % 1
    profc = np.array([np.mean(res_[(phc >= edges[i]) & (phc < edges[i + 1])]) for i in range(12)])
    # least-squares sawtooth amplitude: res ~ s * (ph - 0.5) for best phase offset
    best = max(((np.polyfit(((ph - o) % 1) - 0.5, res_, 1)[0], o) for o in np.linspace(0, 1, 24, endpoint=False)), key=lambda x: abs(x[0]))
    rate_ppm = np.polyfit(tb[good], psi[good], 1)[0] / (2 * np.pi) / 240 * 1e6
    res.update(minutes=round(N / FS / 60, 1), beats=int(tr.size), cycle_rate_ppm=round(float(rate_ppm), 1),
               cycle_period_s=round(float(1 / abs(rate_ppm * 240e-6)), 1) if rate_ppm else None,
               delay_median_ms=round(float(np.median(dl)), 1), resid_sd_ms=round(float(np.std(res_)), 2),
               profile_ptp_ms=round(float(np.ptp(prof)), 2), control_ptp_ms=round(float(np.ptp(profc)), 2),
               ramp_ms_per_cycle=round(float(best[0]), 2), sem_ms=round(float(np.std(res_) / np.sqrt(tr.size / 12)), 2))
    save.update(tr=tr, dl=dl, res=res_, tb=tb, psi=psi, prof=prof, profc=profc)
if co2 is not None and co2.size > 62.5 * 1500:
    F2 = 62.5; m0 = int(F2 * 600); x = co2[m0:m0 + int(F2 * 60 * MIN)]
    ch = (np.diff(x) != 0).astype(float); act = np.convolve(ch, np.ones(25), "same") > 0
    ch = ch - ch[act].mean(); ch[~act] = 0; k = np.arange(ch.size)
    # update rate: scan the fundamental near 25 Hz on 10-min pieces
    fu = []
    for s0 in range(0, ch.size - int(F2 * 600), int(F2 * 1200)):
        seg = ch[s0:s0 + int(F2 * 600)]; kk = np.arange(seg.size); fs_ = np.arange(24.90, 25.10, 0.0005)
        amp = np.array([abs(np.mean(seg * np.exp(-2j * np.pi * f * kk / F2))) for f in fs_]); fu.append(float(fs_[amp.argmax()]))
    # phase of the update grid against an exact 25 Hz grid on the file's sample count, 5 s blocks
    B = int(F2 * 5); nb = ch.size // B
    zc = (ch * np.exp(-2j * np.pi * 25 * k / F2))[:nb * B].reshape(nb, B).mean(1)
    pc = np.unwrap(np.angle(zc)); tbc = (np.arange(nb) + 0.5) * 5
    sl = np.polyfit(tbc, pc, 1)[0]
    res.update(co2_update_hz=[round(f, 4) for f in fu], co2_ppm_vs_25=round(float((np.median(fu) / 25 - 1) * 1e6)),
               co2_phase_slope_ppm=round(float(sl / (2 * np.pi) / 25 * 1e6)), co2_slip_period_s=round(float(abs(2 * np.pi / sl)), 1))
    save.update(tbc=tbc, pc=pc, co2seg=x[:int(F2 * 120)])
np.savez(outd / f"c{case}.npz", **save)
with (outd / "saw2.jsonl").open("a") as fh: fh.write(json.dumps(res) + "\n")
print(json.dumps(res))
