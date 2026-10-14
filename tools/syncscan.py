# SPDX-License-Identifier: MIT
"""Pass 1 of the clock re-synchronisation study: every run boundary of every waveform track, from the
original ``.vital`` files (no database involved).

    python tools/syncscan.py --root R --cases cases.json --out DIR [--time-budget 150] [--snip-s 6]

For each waveform track the shared reader's runs (records that join to within half a sample) are
listed. At the end of each run the *jump* is measured: the next run's first time stamp minus where this
run would end at the nominal rate (positive = a gap, negative = an overlap). With the run's duration
this gives the rate error the jump would imply if it were a clock correction, in ppm.

Short runs often come in bursts: single records stamped a little early or late, each its own run. So
the scan also lists *macro boundaries* between consecutive runs of at least ``--long-run-s`` seconds,
counting any short runs between them as part of the boundary; the macro jump is the next long run's
first time stamp minus where the previous long run's start plus all samples since would put it.

For SNUADC/ECG_II, ART and PLETH, BIS/EEG1_WAV and Primus/CO2 (``--snip-tracks``) the scan keeps the stored values
from ``--snip-s`` seconds before each macro boundary (``|jump| <= --snip-max-jump-s``), through any
samples of the short runs, to ``--snip-s`` seconds after it; and the same around up to ``--controls``
random record boundaries inside long runs (known-continuous data, jump 0). Those snippets let later passes test whether the signal is continuous across a boundary
without reading the files again.

Output, all local to DIR (resumable; a case is done once its line is in boundaries.jsonl):
    boundaries.jsonl   one line per case: devices, tracks, runs, boundaries (times and counts only)
    snips/<case>.npz   the snippets (stored values in the file's own dtype, with gain/offset)
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import random
import time

import numpy as np

from timenet_connectors.datasets.vitaldb.vitaldb.timing import nominal_rate, runs
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import TRACK_WAVE, read_vital

SNIP_TRACKS = ("SNUADC/ECG_II", "SNUADC/ART", "SNUADC/PLETH", "BIS/EEG1_WAV", "Primus/CO2")


def scan_case(caseid: int, path: Path, a, rng: random.Random) -> tuple[dict, dict]:
    case = read_vital(path)
    origin = case.origin
    out = {"case": caseid, "devices": {n: {"type": d.type_name, "port": d.port} for n, d in case.devices.items()},
           "tracks": {}}
    snips: dict[str, np.ndarray] = {}
    for name, tr in case.tracks.items():
        if tr.track_type != TRACK_WAVE or not tr.record_times.size:
            continue
        rate = nominal_rate(tr)
        period = float(1 / rate)
        rel = tr.record_times - origin
        offs = tr.record_offsets
        rl = runs(tr)
        sizes = tr.record_sizes
        t = {"device": tr.device, "srate": float(tr.srate), "rate": str(rate), "gain": tr.gain, "offset": tr.offset,
             "fmt": tr.fmt, "n_records": int(rel.size), "n_samples": int(offs[-1]),
             "rec_samples_median": float(np.median(sizes)) if sizes.size else 0.0,
             "start_s": float(rel[0]), "runs": [], "boundaries": [], "controls": []}
        for i, (ra, rb) in enumerate(rl):
            n = int(offs[rb] - offs[ra])
            t["runs"].append({"start_s": float(rel[ra]), "samples": n, "records": rb - ra, "dur_s": n * period})
        for i in range(len(rl) - 1):
            ra, rb = rl[i]
            nb = rl[i + 1][0]
            n = int(offs[rb] - offs[ra])
            end = float(Fraction(float(rel[ra])) + n * (1 / rate))
            jump = float(rel[nb]) - end
            dur = n * period
            b = {"run": i, "end_s": end, "jump_s": jump, "dur_before_s": dur,
                 "dur_after_s": t["runs"][i + 1]["dur_s"], "ppm": (jump / dur * 1e6) if dur > 0 else None,
                 "last_rec_samples": int(sizes[rb - 1]), "next_rec_samples": int(sizes[nb]),
                 "step_back": bool(rel[nb] < rel[rb - 1])}
            t["boundaries"].append(b)
        # macro boundaries: between consecutive LONG runs, with any short runs between them (bursts of
        # single records stamped irregularly) treated as part of the boundary
        longs = [i for i, r in enumerate(t["runs"]) if r["dur_s"] >= a.long_run_s]
        t["macro"] = []
        for i, j in zip(longs, longs[1:]):
            (ra, rb), (sa, _) = rl[i], rl[j]
            n_span = int(offs[sa] - offs[ra])  # samples from the start of run i to the start of run j
            jump = float(rel[sa]) - float(Fraction(float(rel[ra])) + n_span * (1 / rate))
            span = n_span * period
            m = {"from_run": i, "to_run": j, "short_runs_between": j - i - 1,
                 "samples_between": int(offs[sa] - offs[rb]), "end_s": float(rel[ra]) + t["runs"][i]["dur_s"],
                 "jump_s": jump, "span_s": span, "dur_before_s": t["runs"][i]["dur_s"], "dur_after_s": t["runs"][j]["dur_s"],
                 "ppm": jump / span * 1e6 if span > 0 else None,
                 "step_back": bool(any(rel[rl[k][0]] < rel[rl[k - 1][1] - 1] for k in range(i + 1, j + 1)))}
            if name in a.snip_tracks and abs(jump) <= a.snip_max_jump_s:
                w = round(a.snip_s * float(rate))
                if offs[rb] - offs[ra] >= w and offs[rl[j][1]] - offs[sa] >= w and m["samples_between"] <= 60 * float(rate):
                    key = f"{name}|m{i}"
                    snips[key] = np.asarray(tr.values[int(offs[rb]) - w : int(offs[sa]) + w])
                    m["snip"] = key
                    m["snip_pre"] = w
            t["macro"].append(m)
        if name in a.snip_tracks:
            # controls: record boundaries inside long runs, where the data is continuous by construction
            w = round(a.snip_s * float(rate))
            cands = []
            for i, (ra, rb) in enumerate(rl):
                for r in range(ra + 1, rb):
                    if offs[r] - offs[ra] >= w and offs[rb] - offs[r] >= w:
                        cands.append((i, r))
            for i, r in rng.sample(cands, min(a.controls, len(cands))):
                key = f"{name}|c{r}"
                s = int(offs[r])
                snips[key] = np.asarray(tr.values[s - w : s + w])
                t["controls"].append({"run": i, "record": int(r), "at_s": float(rel[r]), "snip": key, "snip_pre": w})
        out["tracks"][name] = t
    return out, snips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=150)
    ap.add_argument("--snip-s", type=float, default=6.0)
    ap.add_argument("--snip-max-jump-s", type=float, default=10.0)
    ap.add_argument("--snip-tracks", nargs="+", default=list(SNIP_TRACKS))
    ap.add_argument("--controls", type=int, default=10)
    ap.add_argument("--long-run-s", type=float, default=5.0,
                    help="runs at least this long anchor macro boundaries; shorter runs between them belong to the boundary")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "snips").mkdir(parents=True, exist_ok=True)
    log = out / "boundaries.jsonl"
    done = {json.loads(x)["case"] for x in log.read_text().splitlines() if x.strip()} if log.exists() else set()
    cases = [c for c in json.loads(Path(a.cases).read_text())["cases"] if c not in done]
    t0 = time.time()
    slowest = 0.0
    n_done, mb = 0, 0.0
    for c in cases:
        if time.time() - t0 + max(2 * slowest, 5) > a.time_budget:
            break
        path = Path(a.root) / "vital_files" / f"{c:04d}.vital"
        t1 = time.time()
        try:
            res, snips = scan_case(c, path, a, random.Random(a.seed * 100003 + c))
        except Exception as e:  # noqa: BLE001 - a broken file is a finding, not a stop
            res, snips = {"case": c, "error": f"{type(e).__name__}: {e}"}, {}
        res["seconds"] = time.time() - t1
        res["file_mb"] = path.stat().st_size / 1e6
        if snips:
            np.savez_compressed(out / "snips" / f"{c:04d}.npz", **snips)
        with log.open("a") as fh:
            fh.write(json.dumps(res) + "\n")
        slowest = max(slowest, time.time() - t1)
        n_done += 1
        mb += res["file_mb"]
    left = len(cases) - n_done
    print(json.dumps({"cases_done": n_done, "mb": round(mb, 1), "seconds": round(time.time() - t0, 1), "cases_left": left}))


if __name__ == "__main__":
    main()
