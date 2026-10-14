# SPDX-License-Identifier: MIT
"""Arterial pressure as a second timing reference for the ECG-PPG delay. Per-case numbers only.

    python tools/art_check.py --root R --beats DIR --out out.jsonl [--part 0 --of 3]

On the stretch exported by ``beats_export`` (continuous ECG_II + PLETH, matched R-peaks and PPG upstrokes):

1. Is there a real arterial line? SNUADC/ART is recorded with every case; per 10 s window the stretch
   counts as arterial when the median is 30-200 mmHg, pulse pressure 15-150 mmHg and beats are present.
2. ART fiducials at every R-peak: filtered steepest upstroke (as for the PPG), raw foot and raw systolic
   peak. Frame locking of raw ART fiducials (interval resultant at 33.3 ms, absolute frame-clock scan),
   as for the PPG, to see whether ART passes through the framed path.
3. Three timing series on the beats that have all three fiducials: ECG->ABP, ECG->PPG, ABP->PPG.
   Beat-to-beat noise (robust SD of successive differences / sqrt 2) and slow wander of each; the
   physiosync slip detector on each, with the sign-flipped series as control; steps at the ECG->PPG
   slips seen in the other two series.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "physiosync"))
from frame_an import ibi_lock, phase_scan
from pat import ECG, sample_times
from physiosync.ecg_ppg import _local_steps, _p2p
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

from physiosync import detect_slips

warnings.filterwarnings("ignore")
ART = "SNUADC/ART"
FS = 500.0


def noise_ms(x: np.ndarray) -> float:
    d = np.diff(x)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) * 1000) if d.size > 20 else float("nan")


def slips_summary(t: np.ndarray, x: np.ndarray) -> dict:
    s = detect_slips(t, x, frame_s=1 / 30)
    n = detect_slips(t, 2 * np.median(x) - x, frame_s=1 / 30)
    return {"slips": len(s["t"]), "null": len(n["t"]), "steps_ms": [round(v * 1000, 1) for v in s["step_s"]],
            "t": [float(v) for v in s["t"]]}


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
    files = [f for i, f in enumerate(sorted(Path(a.beats).glob("*.npz"))) if i % a.of == a.part]
    t_start, slowest = time.time(), 0.0
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
                r_t, p_t = z["r_t"], z["p_t"]
                case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ART, ECG})
                tr = case.tracks.get(ART)
                if tr is None or tr.srate != 500 or not tr.record_times.size:
                    res["skip"] = "no ART"
                else:
                    ta = sample_times(tr)
                    i0 = int(np.searchsorted(ta, r_t[0] - 2))
                    i1 = int(np.searchsorted(ta, r_t[-1] + 2))
                    seg_t = ta[i0:i1]
                    if seg_t.size < 600 * FS or np.max(np.abs(np.diff(seg_t) - 1 / FS)) > 0.01 / FS:
                        res["skip"] = "ART not continuous over the stretch"
                    else:
                        raw = np.asarray(tr.values[i0:i1])
                        x = tr.physical()[i0:i1].astype(np.float64)
                        # 1. arterial windows
                        w = int(10 * FS)
                        nw = x.size // w
                        X = x[: nw * w].reshape(nw, w)
                        med = np.median(X, axis=1)
                        pp = np.quantile(X, 0.95, axis=1) - np.quantile(X, 0.05, axis=1)
                        valid_w = (med > 30) & (med < 200) & (pp > 15) & (pp < 150)
                        res["art_valid_frac"] = round(float(valid_w.mean()), 3)
                        res["art_median_mmHg"] = round(float(np.median(med[valid_w])), 1) if valid_w.any() else None
                        if valid_w.mean() < 0.5:
                            res["skip"] = "no arterial line (ART not pulsatile)"
                        else:
                            # 2. ART fiducials: filtered steepest upstroke within 0.05-0.45 s after each R
                            sos = signal.butter(2, (0.5, 10), btype="bandpass", fs=FS, output="sos")
                            y = signal.sosfiltfilt(sos, x)
                            d = np.gradient(y)
                            ok_s = np.repeat(valid_w, w)[: x.size]
                            ok_s = np.r_[ok_s, np.zeros(x.size - ok_s.size, bool)]
                            up, ft, pk, keep = [], [], [], []
                            for k, rt in enumerate(r_t):
                                a_ = int(round((rt + 0.05 - seg_t[0]) * FS))
                                b_ = int(round((rt + 0.45 - seg_t[0]) * FS))
                                if a_ < 200 or b_ + 200 >= x.size or not ok_s[a_:b_].all():
                                    continue
                                m = a_ + int(np.argmax(d[a_:b_]))
                                if d[m] <= 0:
                                    continue
                                den = d[m - 1] - 2 * d[m] + d[m + 1]
                                off = 0.5 * (d[m - 1] - d[m + 1]) / den if den != 0 else 0.0
                                up.append(m + float(np.clip(off, -0.5, 0.5)))
                                ft.append(m - 150 + int(np.argmin(raw[m - 150 : m + 1])))
                                pk.append(m + int(np.argmax(raw[m : m + 200])))
                                keep.append(k)
                            keep = np.asarray(keep)
                            up, ft, pk = np.asarray(up), np.asarray(ft, dtype=np.int64), np.asarray(pk, dtype=np.int64)
                            res["beats_with_art"] = int(keep.size)
                            res["frame"] = {"ibi_lock": {"peak": ibi_lock(pk), "foot": ibi_lock(ft)},
                                            "phase_peak": phase_scan(pk), "phase_foot": phase_scan(ft)}
                            # 3. the three series on common beats
                            ta_up = seg_t[0] + up / FS
                            tt = r_t[keep]
                            e_a = ta_up - tt
                            e_p = p_t[keep] - tt
                            a_p = p_t[keep] - ta_up
                            good = (e_a > 0.05) & (e_a < 0.45)
                            tt, e_a, e_p, a_p = tt[good], e_a[good], e_p[good], a_p[good]
                            res["n_common"] = int(tt.size)
                            if tt.size > 600:
                                res["series"] = {}
                                for name, s in (("ecg_abp", e_a), ("ecg_ppg", e_p), ("abp_ppg", a_p)):
                                    res["series"][name] = {"median_ms": round(float(np.median(s)) * 1000, 1),
                                                           "noise_ms": round(noise_ms(s), 2),
                                                           "wander_ms": round(_p2p(tt, s), 1),
                                                           **slips_summary(tt, s)}
                                # steps in the other series at each ECG->PPG slip (15 beats either side)
                                sp = res["series"]["ecg_ppg"]["t"]
                                bnd = np.r_[tt[0], 0.5 * (tt[1:] + tt[:-1])]
                                st = {n: _local_steps(tt, s, 15) for n, s in (("ecg_abp", e_a), ("ecg_ppg", e_p), ("abp_ppg", a_p))}
                                at = []
                                for x_ in sp:
                                    k = int(np.argmin(np.abs(bnd - x_)))
                                    lo_, hi_ = max(k - 20, 0), min(k + 21, tt.size)
                                    kk = lo_ + int(np.nanargmax(np.where(np.isfinite(st["ecg_ppg"][lo_:hi_]), st["ecg_ppg"][lo_:hi_], -np.inf)))
                                    at.append({n: round(float(v[kk]) * 1000, 1) if np.isfinite(v[kk]) else None for n, v in st.items()})
                                res["steps_at_ecg_ppg_slips"] = at
                                for v in res["series"].values():
                                    v.pop("t")
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"
            res["seconds"] = round(time.time() - t1, 2)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            slowest = max(slowest, res["seconds"])
    print(json.dumps({"part": a.part, "done": sum(1 for _ in out.open()), "seconds": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
