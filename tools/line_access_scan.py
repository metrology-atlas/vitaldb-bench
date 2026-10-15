# SPDX-License-Identifier: MIT
"""Arterial line accesses (lineaccess) in VitalDB cases, matched to lab results. This machine only.

    python tools/line_access_scan.py --root R --cases cases.json --out out.jsonl --labels L.csv
        [--beats DIR --art DIR2] [--part 0 --of 3]

Per case: SNUADC/ART in mmHg, accesses and unavailable stretches (lineaccess defaults), lab draws (results
within 60 s grouped; lab_data dt is from the case start = the file's first record), each draw matched to
the latest access 0-15 min before it; the same with the draw times moved at random (20 circular shifts) as
a control. Appends one JSON line per case and the labels to L.csv (absolute and relative times).

With --beats/--art (the benchmark stretches): arterial-side timing steps (ECG->ART moves by > 10 ms while
ECG->PPG does not) and their distance to the nearest access, against random times.
Resumable: cases already in --out are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lineaccess import line_accesses, match_times
from pat import sample_times
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

ART = "SNUADC/ART"


def med_step(t: np.ndarray, y: np.ndarray, k: int = 15) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if y.size < 3 * k:
        return np.empty(0), np.empty(0)
    w = np.lib.stride_tricks.sliding_window_view(y, k)
    med = np.median(w, axis=1)
    s = np.full(y.size, np.nan)
    s[k : y.size - k + 1] = med[k:] - med[:-k]
    return t, s


def arterial_steps(r_t: np.ndarray, p_t: np.ndarray, a_t: np.ndarray, thr: float = 0.010, sep: float = 60.0) -> list[float]:
    ea, ep = a_t - r_t, p_t - r_t
    ok = np.isfinite(ea)
    t, sa = med_step(r_t[ok], ea[ok])
    _, sp = med_step(r_t[ok], ep[ok])
    if not t.size:
        return []
    cand = np.flatnonzero((np.abs(np.nan_to_num(sa)) > thr) & (np.abs(np.nan_to_num(sp)) <= thr))
    picked: list[int] = []
    for i in cand[np.argsort(-np.abs(sa[cand]))]:
        if all(abs(t[i] - t[j]) >= sep for j in picked):
            picked.append(int(i))
    return sorted(float(t[i]) for i in picked)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--beats", default=None)
    ap.add_argument("--art", default=None)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    root = Path(a.root)
    lab = pd.read_csv(root / "lab_data.csv")
    cases = [c for i, c in enumerate(json.loads(Path(a.cases).read_text())["cases"]) if i % a.of == a.part]
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    new_lab = not Path(a.labels).exists()
    rng = np.random.default_rng(a.part)
    with out.open("a") as fh, open(a.labels, "a", newline="") as lf:
        w = csv.writer(lf)
        if new_lab:
            w.writerow(["case", "label", "start_s", "end_s", "start_rel_s", "end_rel_s", "n_events", "kinds", "max_mmhg",
                        "recovered_ok", "lab_matched"])
        for c in cases:
            if c in done:
                continue
            res: dict = {"case": c}
            try:
                v = read_vital(root / "vital_files" / f"{c:04d}.vital", track_names={ART})
                tr = v.tracks.get(ART)
                if tr is None or not tr.record_times.size:
                    res["skip"] = "no ART"
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    continue
                t0 = v.origin
                t = sample_times(tr)
                x = tr.physical().astype(np.float64)
                acc = line_accesses(x, float(tr.srate), t=t)
                del x
                dts = np.sort(lab.loc[lab.caseid == c, "dt"].to_numpy(dtype=float))
                span = (float(t[0] - t0), float(t[-1] - t0))
                dts = dts[(dts >= span[0]) & (dts <= span[1])]
                draws: list[float] = []
                for d in dts:
                    if not draws or d - draws[-1] > 60:
                        draws.append(float(d))
                draw_abs = np.array(draws) + t0
                m = match_times(acc, draw_abs)
                matched = {x["access"] for x in m if x["access"] is not None}
                ctrl = []
                L = span[1] - span[0]
                for _ in range(20):
                    sh = (np.array(draws) - span[0] + rng.uniform(0, L)) % L + span[0] + t0
                    ctrl.append(float(np.mean([x["access"] is not None for x in match_times(acc, sh)])) if draws else 0.0)
                ac = [x for x in acc if x.kind == "access"]
                oc = [x for x in acc if x.kind == "occlusion"]
                res.update({
                    "hours": round(L / 3600, 2), "accesses": len(ac), "occlusions": len(oc),
                    "unavailable_min": round(sum(x.end - x.start for x in acc if x.kind == "unavailable") / 60, 1),
                    "access_kinds": {k: sum(k in x.kinds for x in ac) for k in ("high", "low", "flat")},
                    "access_s_median": round(float(np.median([x.end - x.start for x in ac])), 1) if ac else None,
                    "recovery_s_median": round(float(np.median([x.recovered - x.end for x in ac])), 1) if ac else None,
                    "not_recovered": sum(not x.recovered_ok for x in ac),
                    "draws": len(draws), "draws_matched": sum(x["access"] is not None for x in m),
                    "lags_s": [round(x["lag_s"], 1) for x in m if x["lag_s"] is not None],
                    "control_match_share": round(float(np.mean(ctrl)), 3),
                    "accesses_with_lab": len(matched),
                })
                if a.beats and a.art and (Path(a.beats) / f"{c:04d}.npz").exists() and (Path(a.art) / f"{c:04d}.npz").exists():
                    z = np.load(Path(a.beats) / f"{c:04d}.npz")
                    a_t = np.load(Path(a.art) / f"{c:04d}.npz")["a_t"]
                    st = arterial_steps(z["r_t"], z["p_t"], a_t)
                    starts = np.array([x.start for x in ac])
                    lo, hi = float(z["r_t"][0]), float(z["r_t"][-1])

                    def near(times: np.ndarray, starts: np.ndarray = starts) -> list[float]:
                        return [float(np.min(np.abs(starts - s))) if starts.size else float("inf") for s in times]

                    res["stretch"] = {"minutes": round((hi - lo) / 60, 1), "arterial_steps": len(st),
                                      "step_to_access_s": [round(d, 1) for d in near(np.array(st))],
                                      "random_to_access_s": [round(d, 1) for d in near(rng.uniform(lo, hi, 50))],
                                      "step_to_occlusion_s": [round(d, 1) for d in near(np.array(st), np.array([x.start for x in oc]))],
                                      "random_to_occlusion_s": [round(d, 1) for d in near(rng.uniform(lo, hi, 50), np.array([x.start for x in oc]))],
                                      "accesses_in_stretch": int(np.sum((starts >= lo) & (starts <= hi)))}
                for i, x in enumerate(acc):
                    end = x.end if x.kind == "unavailable" else x.recovered
                    w.writerow([c, f"art_line:{x.kind}", round(x.start, 3), round(end, 3), round(x.start - t0, 1),
                                round(end - t0, 1), x.n_events, "+".join(x.kinds),
                                round(x.max_mmhg, 1) if np.isfinite(x.max_mmhg) else "", x.recovered_ok, i in matched])
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            lf.flush()
    print(json.dumps({"part": a.part, "done": len(out.read_text().splitlines())}))


if __name__ == "__main__":
    main()
