# SPDX-License-Identifier: MIT
"""Stamp corrections in every waveform track (physiosync.stamp_clock, direction of corrections estimated per
track). Per case, this machine.

    python tools/sa_stamp_scan.py --root R --cases 1,2 --out out.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "physiosync"))
from physiosync import stamp_clock
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

WAVES = ["SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/PLETH", "SNUADC/ART", "SNUADC/CVP", "SNUADC/FEM",
         "BIS/EEG1_WAV", "BIS/EEG2_WAV", "Primus/AWP", "Primus/CO2", "CardioQ/ABP", "CardioQ/FLOW"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in Path(a.out).read_text().splitlines() if l.strip()}
    with open(a.out, "a") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            t1 = time.time()
            res: dict = {"case": c, "tracks": {}}
            try:
                v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names=set(WAVES))
                for n in WAVES:
                    tr = v.tracks.get(n)
                    if tr is None or tr.record_times.size < 10:
                        continue
                    rt = tr.record_times
                    st = tr.record_offsets[: rt.size]
                    k = stamp_clock(rt, st, float(tr.srate))
                    d = np.diff(rt - rt[0] - st / float(tr.srate))
                    res["tracks"][n] = {"fs": float(tr.srate), "hours": round(float(rt[-1] - rt[0]) / 3600, 2),
                                        "rec_ms": round(float(np.median(np.diff(rt))) * 1000, 1),
                                        "rate_ppm": round(k["rate_ppm"], 2), "source": k["rate_source"], "teeth": k["teeth"],
                                        "events": k["corrections"]["events"],
                                        "event_ms": None if k["corrections"]["median_event_s"] is None else round(k["corrections"]["median_event_s"] * 1000, 1),
                                        "event_t": [round(x, 1) for x in k["corrections"]["event_times_s"]],
                                        "gaps": k["gaps"], "steps_gt2ms": int(np.sum(np.abs(d) > 0.002)),
                                        "p2p_ms": round(k["sawtooth_p2p_s"] * 1000, 1)}
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            res["seconds"] = round(time.time() - t1, 1)
            fh.write(json.dumps(res) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
