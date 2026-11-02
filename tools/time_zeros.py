"""Where the VitalDB Arrhythmia Database's annotation times count from (metrology-atlas study time-zeros).

For each annotated case, the annotated R peaks (``time_second``, "from the beginning of the recording") are aligned
with R peaks detected in SNUADC ECG II, whose times are taken from the ECG track's own first sample:

* impulse trains of both in 10 ms bins over the annotated window, FFT cross-correlation over +/-120 s for the
  offset (detected minus annotated);
* each annotation matched to the nearest detected beat within 150 ms after that shift, and a straight line fitted
  to the matched differences (the constant offset at the window's centre, and a drift in ppm);
* the offset compared with the ECG track's start delay: its first record minus the file's zero, the earliest
  record of any track. If the annotations count from the full file's zero, offset = -delay.

    python tools/time_zeros.py run --root $VITALDB_ROOT --ann $ARR --out $W/time_zeros/cases.jsonl [--part K --of N]
    python tools/time_zeros.py agg --cases $W/time_zeros/cases.jsonl --out $W/time_zeros

ARR is the Arrhythmia Database folder (metadata.csv, Annotation_Files/). ``run`` is resumable and stops before
``--time-budget`` seconds; ``agg`` writes summary.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pat import ECG, r_peaks, sample_times  # noqa: E402
from pat_res import ecg_subsample  # noqa: E402
from vitaldb_reader.vital_file import read_vital  # noqa: E402

FS = 500.0
BIN = 0.01
SEARCH = 120.0
MATCH = 0.150


def align(ann: np.ndarray, det: np.ndarray) -> dict:
    lo, hi = ann.min() - SEARCH - 5, ann.max() + SEARCH + 5
    n = int(np.ceil((hi - lo) / BIN)) + 1
    A = np.zeros(n)
    D = np.zeros(n)
    np.add.at(A, np.clip(((ann - lo) / BIN).astype(int), 0, n - 1), 1)
    dd = det[(det >= lo) & (det <= hi)]
    np.add.at(D, np.clip(((dd - lo) / BIN).astype(int), 0, n - 1), 1)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    C = np.fft.irfft(np.conj(np.fft.rfft(A, m)) * np.fft.rfft(D, m), m)      # C[k] = sum A[t] D[t + k]
    lags = np.r_[np.arange(0, n), np.arange(-n + 1, 0)]
    Cl = np.r_[C[:n], C[m - n + 1:]]
    ok = np.abs(lags * BIN) <= SEARCH
    k = lags[ok][np.argmax(Cl[ok])]
    o0 = k * BIN
    j = np.searchsorted(det, ann + o0)
    cand = np.stack([det[np.clip(j - 1, 0, det.size - 1)], det[np.clip(j, 0, det.size - 1)]])
    near = cand[np.argmin(np.abs(cand - (ann + o0)), axis=0), np.arange(ann.size)]
    d = near - ann
    hit = np.abs(d - o0) <= MATCH
    out = {"coarse_offset_s": round(float(o0), 3), "annotations": int(ann.size), "matched": int(hit.sum()),
           "matched_share": round(float(hit.mean()), 4)}
    if hit.sum() >= 20:
        x = ann[hit] - ann[hit].mean()
        b, a = np.polyfit(x, d[hit], 1)
        res = d[hit] - (a + b * x)
        out |= {"offset_s": round(float(a), 5), "drift_ppm": round(float(b * 1e6), 3),
                "resid_mad_ms": round(float(1.4826 * np.median(np.abs(res - np.median(res))) * 1e3), 2)}
    return out


def run(a) -> None:
    meta = pd.read_csv(Path(a.ann) / "metadata.csv")
    cases = [int(c) for i, c in enumerate(meta["case_id"]) if i % a.of == a.part]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t0, n = time.time(), 0
    with out.open("a") as fh:
        for c in cases:
            if c in done:
                continue
            if time.time() - t0 > a.time_budget:
                break
            rec = {"case": c}
            try:
                ann = pd.read_csv(Path(a.ann) / "Annotation_Files" / f"Annotation_file_{c}.csv")["time_second"].to_numpy(float)
                case = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital")
                e = case.tracks[ECG]
                te = sample_times(e)
                zero = case.origin
                first_track = min(((float(t.record_times.min()), n_) for n_, t in case.tracks.items() if t.record_times.size))[1]
                rel = te - te[0]
                sel = np.flatnonzero((rel >= ann.min() - SEARCH - 10) & (rel <= ann.max() + SEARCH + 10))
                x = e.physical()[sel].astype(np.float64)
                x = np.where(np.isfinite(x), x, 0.0)
                rp = r_peaks(x, FS)
                idx = np.asarray(ecg_subsample(x, rp, FS), float)          # sub-sample index within the window
                det = np.interp(idx, np.arange(sel.size), te[sel]) - te[0]  # by the stamps, from the ECG's first sample
                det = np.sort(det)
                rec |= {"ecg_start_delay_s": round(float(te[0] - zero), 5), "file_zero_track": first_track,
                        "ecg_gaps_in_window": int(np.sum(np.abs(np.diff(te[sel]) - 1 / FS) > 0.01 / FS)),
                        "detected": int(det.size)} | align(ann, det)
                if "offset_s" in rec:
                    rec["offset_plus_delay_ms"] = round((rec["offset_s"] + rec["ecg_start_delay_s"]) * 1e3, 2)
            except Exception as ex:  # noqa: BLE001
                rec["error"] = f"{type(ex).__name__}: {ex}"[:200]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
    print(json.dumps({"part": a.part, "new": n, "seconds": round(time.time() - t0, 1)}))


def q(v, ps=(5, 25, 50, 75, 95)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {f"p{p}": round(float(np.percentile(v, p)), 4) for p in ps} | {"n": int(v.size)} if v.size else None


def agg(a) -> None:
    rows = {}
    for f in a.cases:
        for x in Path(f).read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                rows[r["case"]] = r
    rows = list(rows.values())
    ok = [r for r in rows if "offset_s" in r]
    d = np.array([r["offset_plus_delay_ms"] for r in ok])
    S = {"cases": len(rows), "errors": sum(1 for r in rows if "error" in r), "aligned": len(ok),
         "offset_s": q([r["offset_s"] for r in ok]),
         "file_zero_tracks": pd.Series([r["file_zero_track"] for r in ok]).value_counts().to_dict(),
         "matched_share": q([r["matched_share"] for r in ok]),
         "drift_ppm": q([r["drift_ppm"] for r in ok]),
         "offset_plus_delay_ms": q(d)}
    for tol in (5, 10, 50):
        w = np.abs(d) <= tol
        S[f"within_{tol}_ms"] = {"n": int(w.sum()), "mean_ms": round(float(d[w].mean()), 2), "sd_ms": round(float(d[w].std(ddof=1)), 2),
                                 "mean_abs_ms": round(float(np.abs(d[w]).mean()), 2), "sd_abs_ms": round(float(np.abs(d[w]).std(ddof=1)), 2)}
    S["outside_10_ms"] = sorted(([r["case"], r["offset_plus_delay_ms"], r["matched_share"]] for r in ok
                                 if abs(r["offset_plus_delay_ms"]) > 10), key=lambda x: -abs(x[1]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: v for k, v in S.items() if k != "outside_10_ms"}, indent=1))
    print("outside 10 ms:", S["outside_10_ms"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--root", required=True)
    p.add_argument("--ann", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--time-budget", type=float, default=150)
    p = sub.add_parser("agg")
    p.add_argument("--cases", required=True, nargs="+")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": run, "agg": agg}[a.cmd](a)


if __name__ == "__main__":
    main()
