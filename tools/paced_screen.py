# SPDX-License-Identifier: MIT
"""Screen every case for paced (fixed-interval) ECG rhythm. Per case, this machine only.

    python tools/paced_screen.py --root R --out screen.jsonl --shard 0 --shards 2 --time-budget 165

Pacing spikes do not survive the monitor's ECG output at 500 Hz, so pacing is found by its rhythm: runs of
beats whose every R-R interval lies within ``--tol-ms`` of the run's median. Beats are peaks of the
5-25 Hz band-passed |ECG| (threshold per 10-min block), timed by sample count (count time, which the
stamp sawtooth does not touch). Each lead (SNUADC/ECG_II, SNUADC/ECG_V5) is screened on its own. One JSON line
per case: beats, the runs of at least ``--min-run`` beats (start s, beats, median interval ms), and how
much of the case they cover. Resumable; cases are split into shards by position for parallel workers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import signal
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

LEADS = ("SNUADC/ECG_II", "SNUADC/ECG_V5")
FS_EXPECTED = 500.0


def beats(x: np.ndarray, fs: float) -> np.ndarray:
    """Sample indices of beats: peaks of the band-passed |ECG|, threshold set per 10-min block."""
    x = np.nan_to_num(x.astype(np.float64), nan=0.0)
    b, a = signal.butter(2, [5 / (fs / 2), 25 / (fs / 2)], "band")
    y = np.abs(signal.filtfilt(b, a, x))
    blk = int(600 * fs)
    out = []
    for s in range(0, y.size, blk):
        seg = y[s:s + blk]
        if seg.size < fs * 5:
            continue
        h = np.percentile(seg, 98) * 0.4
        if h <= 0:
            continue
        pk, _ = signal.find_peaks(seg, distance=int(0.25 * fs), height=h)
        out.append(pk + s)
    return np.concatenate(out) if out else np.empty(0, np.int64)


def runs(rr_ms: np.ndarray, tol: float, min_run: int) -> list[tuple[int, int, float]]:
    """Runs of consecutive intervals all within ``tol`` of the run's median: (first interval, count, median)."""
    out, i, n = [], 0, rr_ms.size
    while i < n:
        j, med = i + 1, rr_ms[i]
        while j < n and abs(rr_ms[j] - med) <= tol:
            j += 1
            if (j - i) % 8 == 0:
                med = float(np.median(rr_ms[i:j]))
        if j - i >= min_run:
            out.append((i, j - i, float(np.median(rr_ms[i:j]))))
            i = j
        else:
            i += 1
    return out


def screen(path: Path, tol: float, min_run: int) -> dict:
    """Screen one .vital file."""
    v = read_vital(path, track_names=set(LEADS))
    res: dict = {}
    for lead in LEADS:
        tr = v.tracks.get(lead)
        if tr is None or not tr.record_times.size:
            continue
        fs = float(tr.srate)
        pk = beats(tr.physical(), fs)
        if pk.size < 50:
            res[lead] = {"beats": int(pk.size)}
            continue
        rr = np.diff(pk) / fs * 1000.0
        rs = runs(rr, tol, min_run)
        covered = int(sum(n for _, n, _ in rs))
        meds = np.array([m for _, _, m in rs]) if rs else np.empty(0)
        res[lead] = {"beats": int(pk.size), "hours": round(pk[-1] / fs / 3600, 2), "rr_median_ms": round(float(np.median(rr)), 1),
                     "frac_within_tol": round(float(np.mean(np.abs(rr - np.median(rr)) <= tol)), 4),
                     "run_beats": covered, "run_frac": round(covered / rr.size, 4), "n_runs": len(rs),
                     "run_period_ms_median": round(float(np.median(meds)), 1) if meds.size else None,
                     "run_period_ms_iqr": round(float(np.subtract(*np.percentile(meds, [75, 25]))), 1) if meds.size else None,
                     "runs": [[round(pk[i] / fs, 2), n, round(m, 2)] for i, n, m in rs[:200]]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--tol-ms", type=float, default=3.0)
    ap.add_argument("--min-run", type=int, default=30)
    ap.add_argument("--time-budget", type=float, default=165)
    a = ap.parse_args()
    files = sorted((Path(a.root) / "vital_files").glob("*.vital"))
    files = [f for i, f in enumerate(files) if i % a.shards == a.shard]
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t0, slowest, n = time.time(), 0.0, 0
    with out.open("a") as fh:
        for f in files:
            c = int(f.stem)
            if c in done:
                continue
            if time.time() - t0 + max(2 * slowest, 5) > a.time_budget:
                break
            t1 = time.time()
            try:
                r = {"case": c, **screen(f, a.tol_ms, a.min_run)}
            except Exception as ex:  # noqa: BLE001
                r = {"case": c, "error": f"{type(ex).__name__}: {ex}"[:300]}
            r["seconds"] = round(time.time() - t1, 2)
            slowest = max(slowest, r["seconds"])
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
    left = len(files) - len(done) - n
    print(json.dumps({"shard": a.shard, "done_now": n, "left": left, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
