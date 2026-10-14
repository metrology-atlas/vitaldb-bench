# SPDX-License-Identifier: MIT
"""Does the ECG-PPG correction hold up against the arterial pressure? Per-case numbers, this machine only.

    python tools/abp_validate.py --beats DIR --art DIR2 --priors P --out out.jsonl [--part 0 --of 3]

The fit uses ECG and PPG only (no arterial reference), so the arterial line is an independent check: the
PPG's frame slips and creep do not exist in the arterial upstroke, so in arterial->PPG a correct slip shows
as a one-frame step that the correction removes, and a tooth's creep as a slope the correction flattens.
A wrong correction adds a step or a slope instead. Each fitted slip and tooth, with its tier (all of A, B,
C corrected for the check), gives:

* slip: arterial->PPG step (median of 15 beats after minus before) before and after the correction, and
  the ECG->arterial step at the same place (physiology moving both).
* tooth: arterial->PPG slope (Theil-Sen on 15-beat medians) before and after, in ppm, and the fitted rate.
* N teeth: arterial->PPG slope and the largest 15-beat step inside (is it really free of slips?).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "physiosync"))
from physiosync import fit_ecg_ppg_delay

F = 1 / 30
ORDER = "ABCNU"


def med15(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    n = y.size // 15
    return t[: 15 * n].reshape(n, 15).mean(1), np.median(y[: 15 * n].reshape(n, 15), 1)


def step_at(t: np.ndarray, y: np.ndarray, x: float, k: int = 15) -> float:
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    i = int(np.searchsorted(t, x))
    if i < k or i + k > t.size or t[i + k - 1] - t[i - k] > 120:
        return float("nan")
    return float(np.median(y[i : i + k]) - np.median(y[i - k : i]))


def slope_ppm(t: np.ndarray, y: np.ndarray) -> float:
    tt, yy = med15(t, y)
    if tt.size < 10:
        return float("nan")
    return float(stats.theilslopes(yy, tt)[0] * 1e6)


def max_step(t: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if y.size < 60:
        return float("nan")
    w = np.lib.stride_tricks.sliding_window_view(y, 15)
    med = np.median(w, axis=1)
    d = med[15:] - med[:-15]
    return float(np.max(np.abs(d))) if d.size else float("nan")


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
    files = [f for i, f in enumerate(sorted(Path(a.art).glob("*.npz"))) if i % a.of == a.part]
    with open(a.out, "a") as fh:
        for f in files:
            c = int(f.stem)
            res: dict = {"case": c}
            try:
                z = np.load(Path(a.beats) / f.name)
                r_t, p_t = z["r_t"], z["p_t"]
                a_t = np.load(f)["a_t"]
                res["art_coverage"] = round(float(np.isfinite(a_t).mean()), 3)
                if res["art_coverage"] < 0.5:
                    res["skip"] = "arterial line on < 50% of beats"
                    fh.write(json.dumps(res) + "\n")
                    continue
                m = fit_ecg_ppg_delay(r_t, p_t, adopted_pat_s=0.27, frame_s=F, matched=True,
                                      rate_prior_ppm=pri.get(c), apply_min_tier="C")
                res["rate_conflict"] = bool(m.qc.get("rate_conflict"))
                pc = m.correct(p_t)
                y0, y1 = p_t - a_t, pc - a_t          # arterial->PPG before / after
                ea = a_t - r_t                         # ECG->arterial (physiology)
                edges, tiers = m.tooth_edges, m.tooth_tier
                n = len(edges) - 1
                slips = []
                for j in range(1, n):
                    x = edges[j]
                    worst = max(tiers[j - 1], tiers[j], key=ORDER.index)
                    slips.append({"tier": worst, "inserted": x in m.inserted_t,
                                  "before_ms": round(step_at(r_t, y0, x) * 1e3, 2),
                                  "after_ms": round(step_at(r_t, y1, x) * 1e3, 2),
                                  "ecg_art_ms": round(step_at(r_t, ea, x) * 1e3, 2)})
                teeth = []
                for k in range(n):
                    sel = (r_t >= edges[k]) & (r_t < edges[k + 1])
                    row = {"tier": tiers[k], "minutes": round((edges[k + 1] - edges[k]) / 60, 1),
                           "rate_ppm": round(m.tooth_rate[k] * 1e6, 2) if k < len(m.tooth_rate) else None,
                           "applied": m.applied(k),
                           "slope_before_ppm": round(slope_ppm(r_t[sel], y0[sel]), 2),
                           "slope_after_ppm": round(slope_ppm(r_t[sel], y1[sel]), 2),
                           "slope_ecg_art_ppm": round(slope_ppm(r_t[sel], ea[sel]), 2)}
                    if tiers[k] == "N":
                        row["max_step_ms"] = round(max_step(r_t[sel], y0[sel]) * 1e3, 2)
                    teeth.append(row)
                res["slips"], res["teeth"] = slips, teeth
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(res, allow_nan=True) + "\n")
            fh.flush()
    print(json.dumps({"part": a.part, "done": len(Path(a.out).read_text().splitlines())}))


if __name__ == "__main__":
    main()
