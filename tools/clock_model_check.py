# SPDX-License-Identifier: MIT
"""Check the clock model as a reader sees it: ECG~EEG bursts on times read back through CorrectedStore.

    python tools/clock_model_check.py --build timenet_db/main-nominal-integer --out check.jsonl

For each case of the build with both SNUADC/ECG_II and BIS/EEG1_WAV clocks, the two tracks are read three
ways: the stored (nominal) times, corrected with the leave-one-out population constant, and corrected with
the case's own bursts. The burst probe of tools/shared_artifacts.py then measures the EEG's lag behind the
ECG in 10-min windows. After correction the lag is the remaining timing error between the two devices, so
it should sit near zero, and inside the 0.99 bounds the store reports about 99% of the time (the bounds of
the population variant are held out: the case's own bursts did not set its constant). Lags are kept unfiltered (the stored-times variant has lags far outside
the corrected range, so summaries should filter it as shared_artifacts does). Resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from shared_artifacts import BAND, DT, burst_probe, envelope

from vitaldb_bench.clock_model import GOOD_MARGIN, GOOD_PEAK, TimeNetSource, load_clock_model
from vitaldb_bench.correct import CorrectedStore

TRACKS = {"ecg": "SNUADC/ECG_II", "eeg1": "BIS/EEG1_WAV"}


def lags(cs: CorrectedStore, case: int) -> tuple[list, dict]:
    """Good burst windows ``[t, lag_s]`` and the EEG track's error summary."""
    raw, err = {}, {}
    for k, name in TRACKS.items():
        w = cs.window(case, name, -1e7, 1e7)
        if w is None or not w.values.size:
            return [], {}
        x = np.nan_to_num(w.values.astype(np.float32), nan=float(np.nanmedian(w.values)))
        raw[k] = (w.t, x, 1.0 / float(np.median(np.diff(w.t_nominal[:1000]))))
        if k == "eeg1":
            err = {"corrected": w.corrected, **w.error, "notes": w.notes}
    t0 = float(np.floor(min(t[0] for t, _, _ in raw.values())))
    n = int((max(t[-1] for t, _, _ in raw.values()) - t0) / DT) + 1
    env = {k: envelope(t, x, fs, BAND[k], t0, n) for k, (t, x, fs) in raw.items()}
    a = np.asarray(burst_probe(env, 600, 300, 20).get("ecg~eeg1", []), dtype=float).reshape(-1, 5)
    g = a[(a[:, 2] >= GOOD_PEAK) & (a[:, 2] - a[:, 3] >= GOOD_MARGIN)]
    return [[float(r[0]), float(r[1]) / 1000.0] for r in g], err


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=0)
    a = ap.parse_args()
    src = TimeNetSource(a.build)
    models = {v: load_clock_model(a.build, corrections=v) for v in ("none", "population", "per_case")}
    margin = 1.0 + max(abs(c.ref_shift_s) for c in models["population"].clocks.values())
    stores = {v: CorrectedStore(src, m, margin_s=margin) for v, m in models.items()}
    cases = sorted({c for c, t in models["population"].clocks if t == TRACKS["eeg1"]}
                   & {c for c, t in models["population"].clocks if t == TRACKS["ecg"]})
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    start = time.time()
    with out.open("a") as fh:
        for c in [c for c in cases if c not in done]:
            if a.time_budget and time.time() - start > a.time_budget:
                print("time budget reached")
                break
            t1 = time.time()
            res = {"case": c}
            for v, cs in stores.items():
                res[v], res[f"{v}_error"] = lags(cs, c)
            res["seconds"] = round(time.time() - t1, 1)
            fh.write(json.dumps(res) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
