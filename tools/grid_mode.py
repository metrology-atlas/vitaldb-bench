# SPDX-License-Identifier: MIT
"""Experiment: a constant-rate, phase-anchored sawtooth per recording, checked against arterial pressure.

    python tools/grid_mode.py --beats DIR --art DIR2 --priors P --out out.jsonl [--tol 0.01] [--part 0 --of 3]

The current fit (physiosync) treats every tooth on its own. Here the recording's rate r (its own
corroborated rate, else the setup's when they do not conflict) fixes the tooth length L = frame / r, and
slips are chained: from each strong step (0.6-1.5 frame over 15 beats, >= 0.4 frame over 3), the next slip
is expected one L on, and is accepted only if a step of at least ``--min-step`` frames lies within
``--tol`` x L (at least 10 s) of it; the chain re-anchors on every accepted slip (slow drift) and stops
where none is found (a pause or a phase jump). Teeth between chained slips are corrected (balanced: rate =
frame / length). End teeth are corrected only up to the next expected slip.

The fit is ECG and PPG only; the arterial line is the check, as in abp_validate: for every tooth, whether
the current fit had it as A/B (old) or not (new), the ABP->PPG step at its slips and slope inside, before
and after this correction. Per-case numbers, this machine only.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from physiosync import fit_ecg_ppg_delay
from physiosync.ecg_ppg import _local_steps

F = 1 / 30


def med15(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    n = y.size // 15
    return t[: 15 * n].reshape(n, 15).mean(1), np.median(y[: 15 * n].reshape(n, 15), 1)


def slope_ppm(t: np.ndarray, y: np.ndarray) -> float:
    tt, yy = med15(t, y)
    return float(stats.theilslopes(yy, tt)[0] * 1e6) if tt.size >= 10 else float("nan")


def step_at(t: np.ndarray, y: np.ndarray, x: float, k: int = 15) -> float:
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    i = int(np.searchsorted(t, x))
    if i < k or i + k > t.size or t[i + k - 1] - t[i - k] > 120:
        return float("nan")
    return float(np.median(y[i : i + k]) - np.median(y[i - k : i]))


def chain_slips(tr: np.ndarray, pat: np.ndarray, L: float, tol: float, min_step: float) -> list[list[float]]:
    bound = np.r_[tr[0], 0.5 * (tr[1:] + tr[:-1])]
    s15 = _local_steps(tr, pat, 15)
    s3 = _local_steps(tr, pat, 3)
    s15f = np.nan_to_num(s15, nan=-np.inf)
    # step peaks, at least 60 s apart
    order = np.argsort(-s15f)
    peaks: list[int] = []
    for i in order:
        if s15f[i] < min_step * F:
            break
        if all(abs(bound[i] - bound[j]) >= 60 for j in peaks):
            peaks.append(int(i))
    peaks.sort()
    pt = bound[peaks]
    strong = [p for p in peaks if 0.6 * F <= s15[p] <= 1.5 * F
              and np.nanmax(s3[max(p - 3, 0) : p + 4]) >= 0.4 * F]
    used: set[int] = set()
    chains = []
    w = max(10.0, tol * L)
    for p in sorted(strong, key=lambda i: -s15[i]):
        if p in used:
            continue
        ch = [p]
        for d in (1, -1):
            x = bound[p]
            while tr[0] + 30 < x + d * L < tr[-1] - 30:
                cand = [q for q in peaks if abs(bound[q] - (x + d * L)) <= w and q not in used and q not in ch]
                if not cand:
                    break
                q = min(cand, key=lambda q: abs(bound[q] - (x + d * L)))
                ch.append(q)
                x = bound[q]
        if len(ch) >= 2:
            used.update(ch)
            chains.append(sorted(float(bound[q]) for q in ch))
    del pt
    return sorted(chains)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--min-step", type=float, default=0.5)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    pri = {int(k): v for k, v in json.loads(Path(a.priors).read_text()).items()}
    files = [f for i, f in enumerate(sorted(Path(a.art).glob("*.npz"))) if i % a.of == a.part]
    with open(a.out, "a") as fh:
        for f in files:
            c = int(f.stem)
            res: dict = {"case": c}
            try:
                z = np.load(Path(a.beats) / f.name)
                r_t, p_t = z["r_t"], z["p_t"]
                a_t = np.load(f)["a_t"]
                if np.isfinite(a_t).mean() < 0.5:
                    res["skip"] = "arterial line on < 50% of beats"
                    fh.write(json.dumps(res) + "\n")
                    continue
                m = fit_ecg_ppg_delay(r_t, p_t, adopted_pat_s=0.27, frame_s=F, matched=True,
                                      rate_prior_ppm=pri.get(c), apply_min_tier="C")
                own = m.qc.get("recording_rate_ppm")
                prior = pri.get(c)
                conflict = bool(m.qc.get("rate_conflict"))
                rate = own if own else (prior if prior and not conflict else None)
                res.update({"rate_ppm": rate, "rate_from": "recording" if own else ("setup" if rate else None),
                            "conflict": conflict})
                if not rate:
                    res["skip"] = "no rate"
                    fh.write(json.dumps(res) + "\n")
                    continue
                L = F / (rate * 1e-6)
                pat = p_t - r_t
                chains = chain_slips(r_t, pat, L, a.tol, a.min_step)
                # old tiers per beat (from the current fit)
                edges = np.asarray(m.tooth_edges)
                k_old = np.clip(np.searchsorted(edges, r_t, side="right") - 1, 0, len(m.tooth_tier) - 1)
                old_ab = np.isin(np.asarray(m.tooth_tier)[k_old], ["A", "B"])
                y0 = p_t - a_t
                teeth = []
                slips = []
                for ch in chains:
                    slips.extend(ch)
                    for s0, s1 in itertools.pairwise(ch):
                        teeth.append((s0, s1, "full"))
                    # end extensions: up to one tooth before the first / after the last slip of a chain
                    teeth.append((max(ch[0] - L, float(r_t[0])), ch[0], "before"))
                    teeth.append((ch[-1], min(ch[-1] + L, float(r_t[-1])), "after"))
                rows = []
                covered = np.zeros(r_t.size, bool)
                for s0, s1, kind in teeth:
                    sel = (r_t >= s0) & (r_t < s1)
                    if sel.sum() < 30 or covered[sel].mean() > 0.5:
                        continue
                    covered[sel] = True
                    rr = F / (s1 - s0) if kind == "full" else rate * 1e-6
                    # correction inside: the delay creeps down at rr; remove it (zero mean in the tooth)
                    corr = -rr * (r_t[sel] - s0)
                    y1 = y0[sel] - (corr - np.nanmean(corr))
                    rows.append({"kind": kind, "minutes": round((s1 - s0) / 60, 1), "old_ab_share": round(float(old_ab[sel].mean()), 2),
                                 "slope_before_ppm": round(slope_ppm(r_t[sel], y0[sel]), 2),
                                 "slope_after_ppm": round(slope_ppm(r_t[sel], y1), 2),
                                 "rate_ppm": round(rr * 1e6, 2)})
                srows = []
                for x in slips:
                    srows.append({"abp_step_ms": round(step_at(r_t, y0, x) * 1e3, 2),
                                  "ecg_art_step_ms": round(step_at(r_t, a_t - r_t, x) * 1e3, 2),
                                  "old_ab": bool(old_ab[min(int(np.searchsorted(r_t, x)), r_t.size - 1)])})
                res.update({"minutes": round((r_t[-1] - r_t[0]) / 60, 1), "chains": len(chains), "slips": srows, "teeth": rows,
                            "covered_min": round(float(covered.mean()) * (r_t[-1] - r_t[0]) / 60, 1),
                            "old_ab_min": round(float(old_ab.mean()) * (r_t[-1] - r_t[0]) / 60, 1),
                            "covered_new_min": round(float((covered & ~old_ab).mean()) * (r_t[-1] - r_t[0]) / 60, 1),
                            "old_lost_min": round(float((~covered & old_ab).mean()) * (r_t[-1] - r_t[0]) / 60, 1)})
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(res) + "\n")
            fh.flush()
    print(json.dumps({"part": a.part, "done": len(Path(a.out).read_text().splitlines())}))


if __name__ == "__main__":
    main()
