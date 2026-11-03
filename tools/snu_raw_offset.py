# SPDX-License-Identifier: MIT
"""How far the raw SNUADC record stamps sit from the stamp model: the bias of the snu-raw timing form of the
SNUADC waveforms (metrology-atlas variables snuadc-ecg-ii, snuadc-art, snuadc-pleth).

    python tools/snu_raw_offset.py scan --root $R --units $W/results/sync_full/units.csv \
        --step-rates $W/results/pat/snuadc_step_rates.json --every 10 --out $W/snu_raw_offset/part.jsonl
    python tools/snu_raw_offset.py agg $W/snu_raw_offset/part.jsonl --out $W/snu_raw_offset

``scan``: every --every-th recording (case ids 1, 1 + every, ...), SNUADC/ECG_II with its reset clusters in sampling
order (vitaldb_reader.resets), the stamp clock as tools/clock_model_fit.py fits it (same fallback rules, same
signal loss at each step-mode correction); per record, its stamp minus the model's time for it. Histograms per
recording in 5 ms bins (records weigh equally: each holds 0.2 s), with the recording's mode (step: model rate below
-20 ppm, as clock_model_fit). Resumable; stops after --time-budget seconds.

``agg``: pooled over recordings, per mode and overall, the 5th, 50th and 95th percentiles of stamp minus model
(seconds; positive = the stamp is later than the model time), and the same per recording (median of medians).
The model's offset is centred in the sawtooth; the snu-raw form instead takes a stamp just after a step-mode
correction as on time, so ``anchored`` also gives the quantiles with each recording's histogram shifted to start at
its own 0.5th percentile (the bottom of its teeth), and the sawtooth's full height (99.5th minus 0.5th percentile).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from physiosync import corrected_times

sys.path.insert(0, str(Path(__file__).parent))
from clock_model_fit import fit_track  # noqa: E402
from vitaldb_reader.resets import reorder_case  # noqa: E402
from vitaldb_reader.vital_file import read_vital  # noqa: E402

ECG = "SNUADC/ECG_II"
EDGES = np.arange(-1.0, 1.0001, 0.005)


def scan(a) -> None:
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    step_rates = json.loads(Path(a.step_rates).read_text())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    n_files = len(list((Path(a.root) / "vital_files").glob("*.vital")))
    t_start = time.time()
    with out.open("a") as fh:
        for c in range(1, n_files + 1, a.every):
            if c in done:
                continue
            if a.time_budget and time.time() - t_start > a.time_budget:
                break
            r: dict = {"case": c}
            try:
                v, _ = reorder_case(read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={ECG}))
                tr = v.tracks.get(ECG)
                if tr is None or tr.record_times.size < 3 or not tr.srate:
                    r["skip"] = "no ECG_II"
                else:
                    info, clk = fit_track(tr, "SNUADC", U.get(c, {}), step_rates)
                    st = tr.record_offsets[: tr.record_times.size]
                    d = tr.record_times - corrected_times(clk, st)
                    h, _ = np.histogram(np.clip(d, EDGES[0], EDGES[-1] - 1e-9), EDGES)
                    nz = np.flatnonzero(h)
                    r.update({"mode": "step" if info["rate_ppm"] < -20 else "small", "rate_ppm": info["rate_ppm"],
                              "rate_tier": info["rate_tier"], "records": int(d.size),
                              "q_s": [round(float(x), 4) for x in np.quantile(d, [0.05, 0.5, 0.95])],
                              "hist_first_bin": int(nz[0]) if nz.size else 0,
                              "hist": h[nz[0]:nz[-1] + 1].tolist() if nz.size else []})
            except Exception as ex:  # noqa: BLE001
                r["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print(json.dumps({"done": len(out.read_text().splitlines())}))


def agg(a) -> None:
    rows = [json.loads(x) for f in a.parts for x in open(f) if x.strip()]
    rows = list({r["case"]: r for r in rows}.values())
    ok = [r for r in rows if "hist" in r]
    mids = (EDGES[:-1] + EDGES[1:]) / 2
    S = {"recordings_read": len(rows), "recordings": len(ok), "skipped": sum("skip" in r for r in rows),
         "errors": sum("error" in r for r in rows)}
    for mode in ("step", "small", "all"):
        sel = [r for r in ok if mode == "all" or r["mode"] == mode]
        if not sel:
            continue
        h = np.zeros(EDGES.size - 1)
        for r in sel:
            h[r["hist_first_bin"]:r["hist_first_bin"] + len(r["hist"])] += r["hist"]
        cdf = np.cumsum(h) / h.sum()
        q = [float(mids[np.searchsorted(cdf, p)]) for p in (0.05, 0.5, 0.95)]
        S[mode] = {"recordings": len(sel), "records": int(h.sum()),
                   "pooled_p5_p50_p95_s": [round(x, 3) for x in q],
                   "per_recording_median_of_q_s": [round(float(np.median([r["q_s"][i] for r in sel])), 3) for i in range(3)],
                   "rate_tiers": {t: sum(r["rate_tier"] == t for r in sel) for t in sorted({r["rate_tier"] for r in sel})}}
        # anchored: each recording shifted so its 0.5th percentile bin is zero
        ha = np.zeros(EDGES.size - 1)
        heights = []
        for r in sel:
            hh = np.asarray(r["hist"], float)
            c = np.cumsum(hh) / hh.sum()
            lo, hi = int(np.searchsorted(c, 0.005)), int(np.searchsorted(c, 0.995))
            ha[: hh.size - lo] += hh[lo:]
            heights.append((hi - lo) * 0.005)
        ca = np.cumsum(ha) / ha.sum()
        S[mode]["anchored_p5_p50_p95_s"] = [round(float(np.searchsorted(ca, p) * 0.005 + 0.0025), 3) for p in (0.05, 0.5, 0.95)]
        S[mode]["height_s_p10_p50_p90"] = [round(float(x), 3) for x in np.quantile(heights, [0.1, 0.5, 0.9])]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--root", required=True)
    p.add_argument("--units", required=True)
    p.add_argument("--step-rates", required=True)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--time-budget", type=float, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("agg")
    p.add_argument("parts", nargs="+")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"scan": scan, "agg": agg}[a.cmd](a)


if __name__ == "__main__":
    main()
