# SPDX-License-Identifier: MIT
"""Look for recorder-PC clock events: moments when every PC-stamped device jumps together. This machine only.

    python tools/pc_clock_steps.py --root R --out steps.jsonl --shard 0 --shards 2 --time-budget 160

Solar8000 and Primus numerics have no clock of their own: the recorder stamps them on arrival. A jump in the PC
clock, or a stall of the recorder, therefore shows in both at the same moment as an update interval longer (or
shorter) than usual. Waveforms are stamped from each device's own sample count and re-anchored to the PC only now
and then (SNUADC on a drift threshold, BIS on a timer), so whether they jump at the same moment tells the cases
apart: a recording pause makes every track jump together; a PC clock step or a recorder stall moves only the
PC-stamped numerics at once.

Per case: every interval in Solar8000/HR and Primus/ETCO2 longer than the track's median by more than ``--min-s``,
or negative; events where both devices have one within ``--window-s``; and at each event the stamp jump of
SNUADC/ECG_II and BIS/EEG1_WAV records within the window. One JSON line per case; resumable, sharded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

NUM = ("Solar8000/HR", "Primus/ETCO2")
WAVE = ("SNUADC/ECG_II", "BIS/EEG1_WAV")


def odd_intervals(t: np.ndarray, min_s: float) -> list[tuple[float, float]]:
    """(time after the interval, excess over the median interval) for unusual intervals, in raw record order."""
    if t.size < 20:
        return []
    d = np.diff(t)
    m = float(np.median(d))
    i = np.flatnonzero((d - m > min_s) | (d < 0))
    return [(float(t[k + 1]), float(d[k] - m)) for k in i]


def wave_jump(tr, t_at: float, window: float) -> float | None:
    """Sum of record stamp jumps (stamp step minus block length) within the window, s; None when no records there."""
    t, sz = tr.record_times, tr.record_sizes[: tr.record_times.size]
    k = np.flatnonzero(np.abs(t[1:] - t_at) <= window)
    if not k.size:
        return None
    dev = (t[1:] - t[:-1] - sz[:-1] / float(tr.srate))[k]
    return float(np.sum(dev[np.abs(dev) > 1e-3]))


def one(path: Path, a) -> dict:
    v = read_vital(path, track_names=set(NUM) | set(WAVE))
    origin = v.origin
    odd = {n: odd_intervals(v.tracks[n].record_times, a.min_s) for n in NUM if n in v.tracks}
    res: dict = {"have": sorted(v.tracks), "odd_counts": {n: len(x) for n, x in odd.items()}}
    if len(odd) < 2:
        return res
    sol, pri = odd[NUM[0]], odd[NUM[1]]
    events = []
    for ts, es in sol:
        near = [(tp, ep) for tp, ep in pri if abs(tp - ts) <= a.window_s]
        if not near:
            continue
        tp, ep = min(near, key=lambda x: abs(x[0] - ts))
        ev = {"t": round(ts - origin, 1), "solar_excess_s": round(es, 3), "primus_excess_s": round(ep, 3)}
        for n in WAVE:
            if n in v.tracks and v.tracks[n].record_times.size > 10:
                j = wave_jump(v.tracks[n], ts, a.window_s)
                ev[n.split("/")[0].lower() + "_jump_s"] = None if j is None else round(j, 3)
        events.append(ev)
    res["events"] = events
    res["hours"] = round(float((v.tracks[NUM[0]].record_times[-1] - v.tracks[NUM[0]].record_times[0]) / 3600), 2)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--min-s", type=float, default=0.5)
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--time-budget", type=float, default=160)
    a = ap.parse_args()
    files = [f for i, f in enumerate(sorted((Path(a.root) / "vital_files").glob("*.vital"))) if i % a.shards == a.shard]
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t0, n = time.time(), 0
    with out.open("a") as fh:
        for f in files:
            c = int(f.stem)
            if c in done:
                continue
            if time.time() - t0 > a.time_budget:
                break
            try:
                r = {"case": c, **one(f, a)}
            except Exception as ex:  # noqa: BLE001
                r = {"case": c, "error": f"{type(ex).__name__}: {ex}"[:200]}
            fh.write(json.dumps(r) + "\n")
            n += 1
    print(json.dumps({"shard": a.shard, "done_now": n, "left": len(files) - len(done) - n}))


if __name__ == "__main__":
    main()
