# SPDX-License-Identifier: MIT
"""Every gap and every continuous segment of every waveform track, as durations (for histograms). This machine only.

    python tools/gap_durations.py scan --root R --cases 1,2,3 --out part.jsonl [--time-budget S]

Records of a track are joined into one continuous segment when the next record starts within 1.5 sample periods of
the end of the previous one (as tools/signal_survey.py). Per case and track: the gap durations and segment durations
in seconds (3 decimals), and the first-to-last span. Only record times and sizes are read, not sample values.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

WAVES = {"SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/PLETH", "SNUADC/ART", "SNUADC/CVP", "SNUADC/FEM", "Primus/CO2",
         "Primus/AWP", "BIS/EEG1_WAV", "BIS/EEG2_WAV", "CardioQ/FLOW", "CardioQ/ABP"}


def pieces(rt, sizes, fs):
    o = np.argsort(rt, kind="stable")
    rt, sizes = rt[o], sizes[o]
    end = rt + sizes / fs
    run_end = np.maximum.accumulate(end)
    step = rt[1:] - run_end[:-1]
    brk = step > 1.5 / fs
    seg = np.r_[run_end[:-1][brk], run_end[-1]] - np.r_[rt[0], rt[1:][brk]]
    return step[brk], seg, float(run_end[-1] - rt[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=1e9)
    a = ap.parse_args()
    from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital  # noqa: PLC0415

    out = Path(a.out)
    done = {json.loads(l)["case"] for l in open(out)} if out.exists() else set()
    t0 = time.time()
    for c in [int(x) for x in a.cases.split(",") if x]:
        if c in done:
            continue
        if time.time() - t0 > a.time_budget:
            break
        r = {"case": c, "tracks": {}}
        try:
            v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names=WAVES)
            for n, tr in v.tracks.items():
                if n in WAVES and tr.record_times.size:
                    g, s, span = pieces(np.asarray(tr.record_times, float), np.asarray(tr.record_sizes, float), float(tr.srate))
                    r["tracks"][n] = {"span": round(span, 1), "gaps": np.round(g, 3).tolist(), "segs": np.round(s, 3).tolist()}
        except Exception as e:  # noqa: BLE001
            r["error"] = repr(e)[:200]
        with open(out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
