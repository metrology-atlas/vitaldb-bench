# SPDX-License-Identifier: MIT
"""Where and how the ECG-PPG sawtooth stops. Per-case numbers only.

    python tools/pause_char.py --beats DIR --art DIR2 --priors P --out out.jsonl [--part 0 --of 3]

For each case the physiosync fit (setup prior and arterial reference where available) gives teeth and
tiers. With L = frame / the recording's tooth rate:

* pauses: end teeth labelled N because the sawtooth is absent; interior teeth longer than 1.5 L with
  the sawtooth absent (``sawtooth_evidence``); whole stretches with no slip at all (no sawtooth seen).
* onset of a pause inside its tooth: the series (arterial->PPG where available, else PAT, 15-beat
  medians) is fitted by "creep at -r from the tooth's slip until tau, flat after" (or the mirror image
  for a pause that ends at a slip) plus a constant, tau scanned; tau as a share of L, and how much
  better than "flat throughout" the fit is.
* the delay level during a pause relative to the sawtooth cycle: (series at the pause side) minus
  (series at the active side corrected by the sawtooth), at the boundary, in frames.
* 15-beat-median series of PAT and arterial->PPG of up to 6 cases with pauses, for plots (this machine).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physiosync import fit_ecg_ppg_delay, sawtooth_evidence

F = 1 / 30


def med15(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    n = y.size // 15
    return t[: 15 * n].reshape(n, 15).mean(1), np.median(y[: 15 * n].reshape(n, 15), 1)


def onset(tt: np.ndarray, yy: np.ndarray, r: float, from_slip_at_start: bool) -> dict:
    """Where inside the stretch the creep stops (from_slip_at_start: creep from tt[0], then flat) or
    starts (creep from tau up to the slip at tt[-1])."""
    if tt.size < 30:
        return {}
    flat = yy - np.median(yy)
    ss_flat = float(flat @ flat)
    best = (np.inf, None)
    for tau in np.linspace(tt[0], tt[-1], 60):
        if from_slip_at_start:
            model = -r * (np.minimum(tt, tau) - tt[0])
        else:
            model = -r * (np.maximum(tt, tau) - tau)
        res = yy - model
        res = res - np.median(res)
        ss = float(res @ res)
        if ss < best[0]:
            best = (ss, tau)
    tau = float(best[1])
    creep = (tau - tt[0]) if from_slip_at_start else (tt[-1] - tau)
    return {"tau_s": round(tau - float(tt[0]), 1), "creep_s": round(float(creep), 1),
            "gain": round(1 - best[0] / ss_flat, 3) if ss_flat > 0 else None}


def two_break(tt: np.ndarray, yy: np.ndarray, r: float) -> dict:
    """Interior tooth: creep from the slip at tt[0] until tau1, flat, creep again from tau2 to the slip at
    tt[-1]. Joint scan; creep_total / L = 1 if the content still gained exactly one frame over the tooth."""
    if tt.size < 30:
        return {}
    flat = yy - np.median(yy)
    ss_flat = float(flat @ flat)
    grid = np.linspace(tt[0], tt[-1], 50)
    best = (np.inf, None, None)
    for i, t1 in enumerate(grid):
        for t2 in grid[i:]:
            model = -r * (np.minimum(tt, t1) - tt[0]) - r * (np.maximum(tt, t2) - t2)
            res = yy - model
            res = res - np.median(res)
            ss = float(res @ res)
            if ss < best[0]:
                best = (ss, t1, t2)
    # uniform creep (the balancing model) for comparison
    rb = (1 / 30) / (tt[-1] - tt[0])
    res = yy + rb * (tt - tt[0])
    res = res - np.median(res)
    ss_bal = float(res @ res)
    return {"creep_after_s": round(float(best[1] - tt[0]), 1), "creep_before_s": round(float(tt[-1] - best[2]), 1),
            "gain": round(1 - best[0] / ss_flat, 3) if ss_flat > 0 else None,
            "gain_vs_balanced": round(1 - best[0] / ss_bal, 3) if ss_bal > 0 else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    pri = {int(k): v for k, v in json.loads(Path(a.priors).read_text()).items()}
    files = [f for i, f in enumerate(sorted(Path(a.beats).glob("*.npz"))) if i % a.of == a.part]
    with open(a.out, "a") as fh:
        for f in files:
            c = int(f.stem)
            res: dict = {"case": c}
            try:
                z = np.load(f)
                r_t, p_t = z["r_t"], z["p_t"]
                af = Path(a.art) / f.name
                a_t = np.load(af)["a_t"] if af.exists() else None
                if a_t is not None and np.isfinite(a_t).mean() < 0.3:
                    a_t = None
                m = fit_ecg_ppg_delay(r_t, p_t, adopted_pat_s=0.27, frame_s=F, matched=True, rate_prior_ppm=pri.get(c),
                                      apply_min_tier="C", ref_t=a_t)
                # the recording's own rate (plausible full teeth), else the setup's
                r = (m.qc.get("recording_rate_ppm") or pri.get(c, 0)) * 1e-6
                res["rate_ppm"] = round(r * 1e6, 2)
                res["minutes"] = round((m.t_end - m.t_start) / 60, 1)
                res["rate_conflict"] = bool(m.qc.get("rate_conflict"))
                res["tiers"] = m.tooth_tier
                res["reasons"] = m.tooth_reason
                edges = m.tooth_edges
                res["edges_min"] = [round((e - m.t_start) / 60, 2) for e in edges]
                use_abp = a_t is not None and np.isfinite(a_t).mean() > 0.7
                y_all = (p_t - a_t) if use_abp else (p_t - r_t)
                res["series_used"] = "abp_ppg" if use_abp else "ecg_ppg"
                pauses = []
                n = len(edges) - 1
                L = F / r if r > 0 else None
                saw = m._saw(r_t)
                for k in range(n):
                    a_, b_ = edges[k], edges[k + 1]
                    kind = None
                    if m.tooth_tier[k] == "N" and n == 1:
                        kind = "whole (no slip)"
                    elif m.tooth_tier[k] == "N" and "absent" in m.tooth_reason[k]:
                        kind = "start" if k == 0 else "end"
                    elif 0 < k < n - 1 and L and b_ - a_ > 1.5 * L and not res["rate_conflict"]:
                        sel = (r_t >= a_) & (r_t < b_)
                        ev = sawtooth_evidence(r_t[sel], y_all[sel], r, F)
                        if ev["teeth"] >= 1.5 and ev["score"] < -0.1:
                            kind = "middle"
                    if kind is None:
                        continue
                    sel = (r_t >= a_) & (r_t < b_)
                    tt, yy = med15(r_t[sel], y_all[sel])
                    okk = np.isfinite(yy)
                    slope = float(np.polyfit(tt[okk], yy[okk], 1)[0]) if okk.sum() > 10 else float("nan")
                    row = {"slope_ppm": round(slope * 1e6, 2) if np.isfinite(slope) else None, "kind": kind, "start_min": round((a_ - m.t_start) / 60, 1), "minutes": round((b_ - a_) / 60, 1),
                           "teeth_expected": round((b_ - a_) / L, 1) if L else None}
                    if kind in ("end", "middle"):
                        row["stop"] = onset(tt, yy, r, True)       # creep after the last slip, then stops
                    if kind in ("start", "middle"):
                        row["resume"] = onset(tt, yy, r, False)    # flat, then creep up to the next slip
                    if kind == "middle":
                        row["two_break"] = two_break(tt, yy, r)
                    # level of the pause relative to the sawtooth: compare 2 min either side of the boundary
                    lev = []
                    for bnd, pause_side in ((a_, 1), (b_, -1)):
                        if (pause_side == 1 and k == 0) or (pause_side == -1 and k == n - 1):
                            continue
                        w_p = (r_t >= bnd) & (r_t < bnd + 120) if pause_side == 1 else (r_t >= bnd - 120) & (r_t < bnd)
                        w_a = (r_t >= bnd - 120) & (r_t < bnd) if pause_side == 1 else (r_t >= bnd) & (r_t < bnd + 120)
                        if w_p.sum() > 30 and w_a.sum() > 30:
                            yp = np.nanmedian(y_all[w_p])
                            ya = np.nanmedian(y_all[w_a] - (saw[w_a] - np.nanmedian(saw)))
                            lev.append(round(float((yp - ya) / F), 3))
                    row["level_frames"] = lev
                    pauses.append(row)
                res["pauses"] = pauses
                if pauses:
                    tt, yy = med15(r_t, p_t - r_t)
                    res["series"] = [np.round((tt - m.t_start) / 60, 2).tolist(), np.round(yy * 1000, 1).tolist()]
                    if a_t is not None:
                        ta, ya = med15(r_t, p_t - a_t)
                        res["series_abp"] = [np.round((ta - m.t_start) / 60, 2).tolist(), np.round(ya * 1000, 1).tolist()]
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(res) + "\n")
            fh.flush()
    print(json.dumps({"part": a.part, "done": len(Path(a.out).read_text().splitlines())}))


if __name__ == "__main__":
    main()
