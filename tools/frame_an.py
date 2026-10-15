# SPDX-License-Identifier: MIT
"""The 30 Hz PPG frame clock, seen through raw-sample PPG fiducials. Per-case numbers only.

    python tools/frame_an.py --root R --beats DIR --out out.jsonl [--part 0 --of 3]

For each case exported by ``beats_export`` (longest continuous stretch, matched beats):

1. Raw-sample fiducials around every PPG upstroke: systolic peak (raw max), foot (raw min), steepest
   rise over 1, 2 and 4 samples, the sharpest upward kink (max 2nd difference) before the upstroke and
   the sharpest downward kink near the peak; and the filtered upstroke as a control.
2. For each: the resultant of the inter-beat intervals at the frame period (500/30 = 16.667 samples) and
   at control periods, i.e. how strongly the fiducial locks to the frame comb.
3. Frame clock: the resultant of the *absolute* sample index of the fiducials modulo P, scanned over
   P = 16.667 * (1 + e), e in +/-500 ppm (1 ppm steps; whole stretch and each half). A continuous frame
   clock gives a sharp peak; its e is the frame rate error relative to the SNUADC sample clock.
4. At each modelled sawtooth slip (physiosync fit): does the PPG waveform repeat or hold a frame?
   In the beat interval of the slip, the best match of 17 samples with the 17 before them (normalised
   mean absolute difference) and the flattest 17-sample run, compared with the same measures in
   control beat intervals of the same case.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pat import PPG
from physiosync import fit_ecg_ppg_delay
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import (
    read_vital,
)

warnings.filterwarnings("ignore")
FS = 500.0
P0 = FS / 30.0
CTRL = (13.7, 15.1, 17.3, 19.1)


def fiducials(x: np.ndarray, up: np.ndarray) -> dict[str, np.ndarray]:
    n = x.size
    d1 = np.diff(x)                 # d1[k] = x[k+1] - x[k]
    d2 = x[2:] - x[:-2]
    d4 = x[4:] - x[:-4]
    dd = np.diff(x, 2)              # dd[k] = x[k+2] - 2x[k+1] + x[k], centred at k+1
    w = int(0.4 * FS)
    out = {k: [] for k in ("peak", "foot", "grad1", "grad2", "grad4", "kink_up", "kink_peak")}
    keep = []
    for u in up:
        if u - w < 2 or u + w + 5 >= n:
            continue
        keep.append(u)
        pk = u + int(np.argmax(x[u : u + w]))
        out["peak"].append(pk)
        out["foot"].append(u - w + int(np.argmin(x[u - w : u + 1])))
        a, b = u - 30, u + 30
        out["grad1"].append(a + int(np.argmax(d1[a:b])) + 1)
        out["grad2"].append(a + int(np.argmax(d2[a:b])) + 1)
        out["grad4"].append(a + int(np.argmax(d4[a:b])) + 2)
        out["kink_up"].append(u - 60 + int(np.argmax(dd[u - 60 : u])) + 1)
        out["kink_peak"].append(pk - 30 + int(np.argmin(dd[pk - 30 : pk + 30])) + 1)
    res = {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}
    res["upstroke_filtered"] = np.asarray(keep, dtype=np.int64)
    return res


def resultant(v: np.ndarray, P: float) -> float:
    return float(np.abs(np.exp(2j * np.pi * v / P).mean())) if v.size else float("nan")


def ibi_lock(idx: np.ndarray) -> dict:
    ibi = np.diff(idx)
    ibi = ibi[(ibi > 150) & (ibi < 1000)]
    return {"n": int(ibi.size), "r_frame": round(resultant(ibi, P0), 4),
            "r_ctrl": round(float(np.mean([resultant(ibi, p) for p in CTRL])), 4),
            "r_60hz": round(resultant(ibi, FS / 60), 4)}


def phase_scan(idx: np.ndarray, span_ppm: float = 500.0, step_ppm: float = 1.0) -> dict:
    """Best frame period for the absolute indices (a continuous frame clock locks them all)."""
    if idx.size < 200:
        return {}
    e = np.arange(-span_ppm, span_ppm + step_ppm / 2, step_ppm) * 1e-6
    P = P0 * (1 + e)
    x = idx.astype(np.float64) - idx[0]
    r = np.empty(e.size)
    for k in range(0, e.size, 200):
        ph = 2 * np.pi * x[None, :] / P[k : k + 200, None]
        r[k : k + 200] = np.abs(np.exp(1j * ph).mean(axis=1))
    i = int(np.argmax(r))
    # sharpness: resultant at the peak vs the median of the scan
    return {"best_ppm": round(float(e[i] * 1e6), 1), "r_best": round(float(r[i]), 4),
            "r_median": round(float(np.median(r)), 4), "r_nominal": round(float(r[e.size // 2]), 4)}


def slip_frames(x: np.ndarray, bnd_idx: list[tuple[int, int]], ctrl_idx: list[tuple[int, int]]) -> dict:
    """Repeated/held frame tests: best normalised match of 17 samples with the 17 before, and flattest
    17-sample run, inside each beat interval (sample index ranges)."""
    L = 17

    def measures(a: int, b: int) -> tuple[float, float] | None:
        seg = x[a - L : b + L].astype(np.float64)
        if seg.size < 4 * L:
            return None
        scale = np.mean(np.abs(np.diff(seg))) + 1e-9
        win = np.lib.stride_tricks.sliding_window_view(seg, L)
        rep = np.mean(np.abs(win[L:] - win[:-L]), axis=1) / scale
        flat = (win.max(axis=1) - win.min(axis=1)) / (scale * L)
        return float(rep.min()), float(flat.min())

    def collect(rngs):
        v = [m for a, b in rngs if (m := measures(a, b)) is not None]
        return np.array(v) if v else np.empty((0, 2))

    s, c = collect(bnd_idx), collect(ctrl_idx)
    out = {"n_slip": len(s), "n_ctrl": len(c)}
    if len(s) and len(c):
        out.update({"rep_slip": np.round(s[:, 0], 3).tolist(), "flat_slip": np.round(s[:, 1], 3).tolist(),
                    "rep_ctrl_q": np.round(np.quantile(c[:, 0], [0.05, 0.25, 0.5]), 3).tolist(),
                    "flat_ctrl_q": np.round(np.quantile(c[:, 1], [0.05, 0.25, 0.5]), 3).tolist(),
                    # share of slip intervals more extreme than the 5% of controls
                    "rep_below_ctrl5": float(np.mean(s[:, 0] < np.quantile(c[:, 0], 0.05))),
                    "flat_below_ctrl5": float(np.mean(s[:, 1] < np.quantile(c[:, 1], 0.05)))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    files = sorted(Path(a.beats).glob("*.npz"))
    files = [f for i, f in enumerate(files) if i % a.of == a.part]
    t_start, slowest = time.time(), 0.0
    rng = np.random.default_rng(0)
    with out.open("a") as fh:
        for f in files:
            c = int(f.stem)
            if c in done:
                continue
            if time.time() - t_start + max(1.5 * slowest, 5) > a.time_budget:
                break
            t1 = time.time()
            res: dict = {"case": c}
            try:
                z = np.load(f)
                case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={PPG})
                p = case.tracks[PPG]
                i0, n = int(z["ppg_first_index"]), int(z["ppg_count"])
                x = np.asarray(p.values[i0 : i0 + n]).astype(np.int64)
                up = z["up_index"].astype(np.int64) - i0
                fid = fiducials(x, up)
                res["ibi_lock"] = {k: ibi_lock(v) for k, v in fid.items()}
                res["phase"] = {}
                for k in ("peak", "foot", "grad1", "kink_peak"):
                    v = fid[k]
                    h = v.size // 2
                    res["phase"][k] = {"all": phase_scan(v), "half1": phase_scan(v[:h]), "half2": phase_scan(v[h:])}
                # slips from the physiosync fit; beat intervals (in PPG sample indices) that contain them
                r_t, p_t = z["r_t"], z["p_t"]
                m = fit_ecg_ppg_delay(r_t, p_t, adopted_pat_s=0.27, frame_s=1 / 30, matched=True)
                res["rate_source"] = m.qc.get("rate_source")
                res["tooth_rate_ppm"] = [round(r_ * 1e6, 2) for r_ in m.tooth_rate]
                t0 = float(z["ppg_t0"])
                up_t = t0 + up / FS  # upstroke times  # noqa: F841 - kept for interactive inspection
                slips = [s_ for s_ in m.tooth_edges[1:-1] if s_ not in m.inserted_t]
                # a slip between beats k-1 and k of the PAT series appears in the PPG between their
                # upstrokes (the PPG lags the ECG by the PAT)
                bnd = []
                for s_ in slips:
                    k = int(np.searchsorted(r_t, s_))
                    if 1 <= k < p_t.size:
                        a_, b_ = int(round((p_t[k - 1] - t0) * FS)), int(round((p_t[k] - t0) * FS))
                        if 0 < a_ < b_ < x.size - 40:
                            bnd.append((a_, b_))
                ks = rng.choice(np.arange(1, p_t.size), size=min(300, p_t.size - 1), replace=False)
                ctrl = [(int(round((p_t[k - 1] - t0) * FS)), int(round((p_t[k] - t0) * FS))) for k in ks]
                ctrl = [(a_, b_) for a_, b_ in ctrl if 20 < a_ < b_ < x.size - 40]
                res["slip_frames"] = slip_frames(x, bnd, ctrl)
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"
            res["seconds"] = round(time.time() - t1, 2)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            slowest = max(slowest, res["seconds"])
    print(json.dumps({"part": a.part, "done": sum(1 for _ in out.open()), "seconds": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
