# SPDX-License-Identifier: MIT
"""Shared-burst lag ramps against the drift predicted from the clock corrections, and lag jumps against record-stamp
steps, on the whole release (metrology-atlas study shared-bursts, figure shared-bursts-slopes).

    python tools/shared_bursts_slopes.py --bursts $W/burst_validation_stamps --units $W/results/sync_full/units.csv \
        --root $VITALDB_ROOT --out $W/shared_bursts_slopes [--time-budget 160]

Inputs: the ECG II ~ EEG1 burst lags on the recorded stamps for every recording (tools/sa_corrected.py, variant
``stamps``), and each recording's SNUADC and BIS drift from the clock-correction scan (units.csv). As the ad hoc
pilot (tools/adhoc/sa_seg.py, sa_jumps.py, sa_jump_baseline.py, sa_slope_plot.py), on every recording:

* good windows: peak r >= 0.3 and 0.15 above the best peak more than 1 s away; recordings with 4 or more;
* segments between lag jumps (successive good windows > 60 ms apart, or more than an hour apart); a straight line
  through each segment of 4 or more windows spanning 20 min or more; the recording's ramp is the median segment
  slope (ppm), against the prediction -(BIS drift - SNUADC drift);
* lag jumps (> 60 ms between good windows up to 15 min apart): whether either track's record stamps step (by
  30 ms to 2 s, against the sample count) from 10 min before to 5 min after the jump, against the same interval
  40 min later (chance). Needs each file with a jump read once (resumable: OUT/jumps.jsonl; stops before
  --time-budget, rerun to finish).

Writes summary.json and shared-bursts-slopes.png.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import numpy as np

from vitaldb_reader.vital_file import read_vital  # noqa: E402

ECG, EEG = "SNUADC/ECG_II", "BIS/EEG1_WAV"


def good(rows):
    a = np.array(rows, float).reshape(-1, 5)
    return a[(a[:, 2] >= 0.3) & (a[:, 2] - a[:, 3] >= 0.15)]


def load(folder):
    out = {}
    for f in sorted(glob.glob(str(Path(folder) / "part_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            m = r.get("stamps")
            if m and "burst" in m and m["burst"].get("ecg~eeg1"):
                g = good(m["burst"]["ecg~eeg1"])
                if len(g) >= 4:
                    out[r["case"]] = (m.get("t0"), g[np.argsort(g[:, 0])])
    return out


def segments(g):
    br = [0]
    for i in range(1, len(g)):
        if g[i, 1] - g[i - 1, 1] > 60 or g[i, 0] - g[i - 1, 0] > 3600:
            br.append(i)
    br.append(len(g))
    return br


def stamp_steps(tr):
    rt = tr.record_times
    res = rt - rt[0] - tr.record_offsets[:rt.size] / float(tr.srate)
    d = np.diff(res)
    k = np.flatnonzero((np.abs(d) > 0.03) & (np.abs(d) < 2))
    return rt[k + 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bursts", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=160)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = load(a.bursts)
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    slopes, jumps = [], {}
    for c, (t0, g) in B.items():
        br = segments(g)
        sl = []
        for s_, e_ in zip(br[:-1], br[1:]):
            s = g[s_:e_]
            if len(s) >= 4 and np.ptp(s[:, 0]) >= 1200:
                sl.append(np.polyfit(s[:, 0], s[:, 1], 1)[0] * 1000)        # ms per s x 1000 = ppm
        u = U.get(c, {})
        try:
            pred = -(float(u["bis_drift_ppm"]) - float(u["snuadc_drift_ppm"]))
        except (KeyError, ValueError):
            pred = None
        if sl and pred is not None:
            slopes.append((c, float(np.median(sl)), pred, len(sl)))
        jl = [(float(g[i, 0]), float(g[i, 1] - g[i - 1, 1])) for i in br[1:-1] if g[i, 0] - g[i - 1, 0] <= 900]
        if jl:
            jumps[c] = (t0, jl)
    # stamp steps around each lag jump (reads the file; resumable)
    jf = out / "jumps.jsonl"
    done = {json.loads(x)["case"]: json.loads(x) for x in jf.read_text().splitlines() if x.strip()} if jf.exists() else {}
    t_start = time.time()
    with jf.open("a") as fh:
        for c, (t0, jl) in sorted(jumps.items()):
            if c in done:
                continue
            if time.time() - t_start > a.time_budget:
                break
            v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG, EEG})
            st = np.concatenate([stamp_steps(v.tracks[n]) for n in (ECG, EEG) if n in v.tracks and v.tracks[n].record_times.size > 1])
            base = t0 if t0 is not None else np.floor(min(v.tracks[n].record_times[0] for n in (ECG, EEG) if n in v.tracks))
            rows = []
            for t, dl in jl:
                hit = bool(np.any((st >= base + t - 600) & (st <= base + t + 300)))
                ctrl = bool(np.any((st >= base + t - 600 + 2400) & (st <= base + t + 300 + 2400)))
                rows.append([t, round(dl, 1), hit, ctrl])
            rec = {"case": c, "jumps": rows}
            fh.write(json.dumps(rec) + "\n")
            done[c] = rec
    finished = all(c in done for c in jumps)
    A = np.array([(s[1], s[2]) for s in slopes])
    k = np.abs(A[:, 0] - A[:, 1]) <= 25
    J = [r for c in done for r in done[c]["jumps"]]
    S = {"recordings_with_4_good": len(B), "slope_recordings": len(slopes), "slopes_within_25ppm": int(k.sum()),
         "r_within_25ppm": round(float(np.corrcoef(A[k].T)[0, 1]), 3), "r_all": round(float(np.corrcoef(A.T)[0, 1]), 3),
         "median_slope_minus_pred_ppm": round(float(np.median(A[:, 0] - A[:, 1])), 2),
         "jump_recordings": len(jumps), "jump_recordings_read": len(done), "complete": finished,
         "lag_jumps": len(J), "jumps_with_stamp_step_pct": round(100 * float(np.mean([r[2] for r in J])), 1) if J else None,
         "chance_pct": round(100 * float(np.mean([r[3] for r in J])), 1) if J else None,
         "jump_ms_p10_50_90": [round(float(x), 0) for x in np.percentile([r[1] for r in J], [10, 50, 90])] if J else None}
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))
    if not finished:
        print("not finished: run again to read the remaining files")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(A[:, 1], A[:, 0], s=6, alpha=0.6)
    ax[0].plot([-160, 40], [-160, 40], "k--", lw=0.8)
    ax[0].set_xlim(-160, 40)
    ax[0].set_ylim(-160, 40)
    ax[0].set_xlabel("predicted −(BIS − SNUADC drift), ppm (clock-correction scan)")
    ax[0].set_ylabel("ramp slope from shared bursts, ppm")
    ax[0].set_title(f"{len(A)} recordings; {int(k.sum())} within 25 ppm (r = {S['r_within_25ppm']:.2f})", fontsize=9)
    ax[1].bar(["lag jump:\nstep within window", "same interval\n40 min later"], [S["jumps_with_stamp_step_pct"], S["chance_pct"]], color=["C0", "0.6"])
    ax[1].set_ylabel(f"% of {len(J)} lag jumps with a stamp step (ECG or EEG)")
    ax[1].set_ylim(0, 100)
    ax[1].set_title("Lag jumps coincide with record-stamp steps", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "shared-bursts-slopes.png", dpi=110)


if __name__ == "__main__":
    main()
