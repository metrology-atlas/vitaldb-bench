# SPDX-License-Identifier: MIT
"""Physiosync sawtooth fit on given cases, and slips seen in ART->PLETH. This machine only.

    python tools/ppg_slips_art_check.py --root R --cases 383 561 --out out.jsonl [--minutes 400]

Per case (whole ECG/PLETH/ART overlap from 10 min in): physiosync fit (slips, tiers, creep per tooth); at each
ECG->PPG slip the step (median of 15 beats after minus before) in ECG->ART and ART->PPG; and the test of an earlier
R-peak analysis: steps between consecutive 30-beat medians of ART->PPG, counted near +-33 ms and +-16.7 ms against a
sign-flipped control.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
EX = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EX)); sys.path.insert(0, str(EX.parent / "tools"))
import pulse_arrival_time as P

def step_at(t, y, ts, k=15):
    i = np.searchsorted(t, ts)
    if i < k or i + k > t.size: return None
    return float(np.median(y[i:i+k]) - np.median(y[i-k:i]))

def one(root, case, minutes):
    v = P.read_vital(root / "vital_files" / f"{case:04d}.vital", track_names={P.ECG, P.PPG, P.ART})
    tr = {k: v.tracks.get(k) for k in (P.ECG, P.PPG, P.ART)}
    ts = {k: P.sample_times(tr[k]) for k in (P.ECG, P.PPG, P.ART)}
    t0 = max(ts[k][0] for k in (P.ECG, P.PPG)) + 600
    t1 = min(min(ts[k][-1] for k in (P.ECG, P.PPG, P.ART)), t0 + 60 * minutes)
    seg = {}
    for k in (P.ECG, P.PPG, P.ART):
        a, b = np.searchsorted(ts[k], t0), np.searchsorted(ts[k], t1)
        seg[k] = (ts[k][a:b], np.nan_to_num(tr[k].physical()[a:b].astype(float)))
    r_t = seg[P.ECG][0][P.r_peaks(seg[P.ECG][1], P.FS)]
    p_t = P.upstrokes(*seg[P.PPG], r_t, 0.10, 0.80)[0]
    a_t = P.upstrokes(*seg[P.ART], r_t, 0.05, 0.45)[0]
    ok = np.isfinite(p_t) & np.isfinite(a_t)
    r, p, a = r_t[ok], p_t[ok], a_t[ok]
    m = P.fit_ecg_ppg_delay(r, p, adopted_pat_s=P.ADOPTED, frame_s=1/30, matched=True, ref_t=a)
    ea, ap = a - r, p - a
    steps = [(step_at(r, ea, s), step_at(r, ap, s)) for s in m.slips_t]
    steps = [s for s in steps if None not in s]
    # the earlier R-peak analysis's test on ART->PLETH: 30-beat medians, steps between consecutive medians
    n = ap.size // 30
    med = np.array([np.median(ap[i*30:(i+1)*30]) for i in range(n)])
    d = np.diff(med) * 1000
    mad = float(np.median(np.abs(d - np.median(d)))) if d.size else None
    near = lambda x, c, w=5: int(np.sum(np.abs(np.abs(x) - c) < w))
    rates = [x * 1e6 for x, tt in zip(m.tooth_rate, m.tooth_tier) if tt in "ABC" and np.isfinite(x)]
    return {"case": case, "minutes": round((t1 - t0) / 60, 1), "beats": int(ok.sum()),
            "slips": len(m.slips_t), "inserted": len(m.inserted_t), "tiers": "".join(m.tooth_tier),
            "creep_ppm_median": round(float(np.median(rates)), 1) if rates else None,
            "creep_ppm_range": [round(min(rates), 1), round(max(rates), 1)] if rates else None,
            "ecg_art_step_ms_median": round(float(np.median([s[0] for s in steps])) * 1000, 1) if steps else None,
            "art_ppg_step_ms_median": round(float(np.median([s[1] for s in steps])) * 1000, 1) if steps else None,
            "art_ppg_steps_ms": [round(s[1] * 1000, 1) for s in steps],
            "test_30beat": {"n_steps": int(d.size), "mad_ms": round(mad, 1) if mad is not None else None,
                            "near_33": near(d, 33.3), "near_16": near(d, 16.7),
                            "control_near_33": near(d + 33.3 * np.sign(np.random.default_rng(0).standard_normal(d.size)), 66.6)}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", type=int); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--minutes", type=float, default=400)
    a = ap.parse_args()
    done = {json.loads(l)["case"] for l in a.out.read_text().splitlines()} if a.out.exists() else set()
    for c in a.cases:
        if c in done: continue
        t = time.time()
        try: res = one(a.root, c, a.minutes)
        except Exception as e: res = {"case": c, "error": repr(e)}
        res["seconds"] = round(time.time() - t)
        with a.out.open("a") as f: f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)
