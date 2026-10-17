# SPDX-License-Identifier: MIT
"""Fit the stamp clock of every SNUADC and BIS waveform track, and measure what is left with the bursts.

    python tools/clock_model_fit.py --root R --cases-json cases.json --units units.csv \
        --step-rates snuadc_step_rates.json --out clock_fits.jsonl

Per case, one JSON line. ``tracks`` holds each track's clock line (physiosync.stamp_clock, with the adopted
VitalDB rules of tools/sa_corrected.py variant "model": SNUADC corrects downwards and BIS upwards; where
SNUADC has no whole tooth and no step-mode correction its stamps are trusted (rate 0), with step-mode
corrections but no tooth the port's typical rate; where BIS has no tooth, its unit's mean rate, else the
case's sync_full rate), with stretch breaks as sample counts so a store can apply it. ``model`` holds the
ECG~EEG burst probe on those corrected times. SNUADC step-mode corrections add back the signal time
the device drops at each one (SNUADC_EVENT_LOSS_S) (the same shape as sa_corrected output), which
:mod:`vitaldb_bench.clock_model` turns into the reference shift and the error distributions. Resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from physiosync import corrected_times, stamp_clock

sys.path.insert(0, str(Path(__file__).parent))
from shared_artifacts import BAND, DT, burst_probe, envelope
from timenet_connectors.datasets.vitaldb.vitaldb.resets import RULE_VERSION as RESET_RULE, reorder_case
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

DEVICES = {"SNUADC": -1, "BIS": 1}           # device -> correction_sign
PROBE = {"ecg": "SNUADC/ECG_II", "eeg1": "BIS/EEG1_WAV", "eeg2": "BIS/EEG2_WAV"}
STEP_EVENT_S = -0.4                          # a median SNUADC correction event this large means step mode
# Signal time SNUADC loses at each step-mode resynchronisation (about 9 samples at 500 Hz): measured on fixed-rate
# paced ECG (tools/paced_breaks.py; -17.2 and -18.2 ms phase steps on the sample count at two corrections).
SNUADC_EVENT_LOSS_S = 0.0177


def _num(x):
    return float(x) if x not in (None, "") else None


def fit_track(tr, dev: str, u: dict, step_rates: dict) -> dict:
    """Stamp clock of one track with the VitalDB fallback rules; breaks as sample counts."""
    st = tr.record_offsets[: tr.record_times.size]
    fs, sign = float(tr.srate), DEVICES[dev]
    loss = SNUADC_EVENT_LOSS_S if dev == "SNUADC" else 0.0
    clk = stamp_clock(tr.record_times, st, fs, correction_sign=sign, event_loss_s=loss)
    tier = clk["rate_source"]
    if clk["rate_source"] == "fit":
        if dev == "SNUADC":
            ev = clk["corrections"]["median_event_s"]
            if ev is not None and ev <= STEP_EVENT_S:
                fall, tier = step_rates.get(u.get("snuadc_port"), step_rates.get("*", 0.0)), "port_step_rate"
            else:
                fall, tier = 0.0, "trusted_stamps"
        else:
            unit, known = _num(u.get("bis_unit_mean_ppm")), _num(u.get("bis_drift_ppm"))
            fall, tier = (unit, "unit_mean") if unit is not None else (known, "sync_full")
        if fall is not None:
            clk = stamp_clock(tr.record_times, st, fs, rate_ppm=fall, correction_sign=sign, event_loss_s=loss)
        else:
            tier = "fit"
    rec = corrected_times(clk, st)
    # one offset per piece: stretches between gaps, split again at each lossy correction (offset + loss)
    seg = np.zeros(st.size, dtype=np.int64)
    seg[clk["breaks"][1:]] = 1
    off = clk["offsets_s"][np.cumsum(seg)].astype(float)
    for i in clk["loss_records"]:
        off[i:] += clk["event_loss_s"]
    starts_rec = np.r_[0, np.flatnonzero(np.diff(off) != 0) + 1]
    k_breaks = np.asarray(st, dtype=np.int64)[starts_rec]
    return {"device": dev, "fs": fs, "t0_s": clk["t0"], "rate_ppm": clk["rate_ppm"], "rate_source": clk["rate_source"],
            "rate_tier": tier, "teeth": clk["teeth"], "k_breaks": k_breaks.tolist(),
            "offsets_s": [float(x) for x in off[starts_rec]], "loss_events": len(clk["loss_records"]),
            "event_loss_s": clk["event_loss_s"], "n_samples": int(np.sum(tr.record_sizes)),
            "n_records": int(st.size), "max_shift_s": float(np.max(np.abs(rec - tr.record_times))),
            "median_event_s": clk["corrections"]["median_event_s"], "events": clk["corrections"]["events"],
            "gaps": clk["gaps"], "sawtooth_p2p_s": clk["sawtooth_p2p_s"]}, clk


def one_case(path: Path, u: dict, step_rates: dict) -> dict:
    """Fits and burst probe for one .vital file."""
    # SNUADC reset clusters in sampling order, as the TimeNet builds store them, so sample counts agree
    v, _ = reorder_case(read_vital(path))
    res: dict = {"origin_s": v.origin, "reset_order": RESET_RULE, "tracks": {}}
    times = {}
    for name, tr in v.tracks.items():
        dev = name.split("/", 1)[0]
        if dev not in DEVICES or tr.track_type != 1 or tr.record_times.size < 3 or not tr.srate:
            continue
        info, clk = fit_track(tr, dev, u, step_rates)
        res["tracks"][name] = info
        if name in PROBE.values():
            st = tr.record_offsets[: tr.record_times.size]
            times[name] = corrected_times(clk, st, tr.record_sizes[: st.size])
    ecg = res["tracks"].get(PROBE["ecg"])
    res["snuadc_mode"] = None if ecg is None else ("step" if ecg["rate_ppm"] < -20 else "small")
    raw = {}
    for k, name in PROBE.items():
        if name in times:
            tr = v.tracks[name]
            x = tr.physical().astype(np.float32)
            x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
            n = min(x.size, times[name].size)
            raw[k] = (times[name][:n], x[:n], float(tr.srate))
    if {"ecg", "eeg1"} <= set(raw):
        t0 = float(np.floor(min(t[0] for t, _, _ in raw.values())))
        n = int((max(t[-1] for t, _, _ in raw.values()) - t0) / DT) + 1
        env = {k: envelope(t, x, fs, BAND[k], t0, n) for k, (t, x, fs) in raw.items()}
        res["model"] = {"t0": t0, "burst": burst_probe(env, 600, 300, 20),
                        "clocks": {"ecg": {"rate_ppm": ecg["rate_ppm"]}}}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="VitalDB release folder (holds vital_files/)")
    ap.add_argument("--cases", default="", help="comma-separated case ids")
    ap.add_argument("--cases-json", action="append", default=[], help="JSON list of case ids, or {\"cases\": [...]} (repeatable)")
    ap.add_argument("--units", required=True, help="sync_full units.csv")
    ap.add_argument("--step-rates", required=True, help="JSON port -> SNUADC step-mode rate (ppm)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=0, help="stop after this many seconds (0: no limit)")
    a = ap.parse_args()
    cases = [int(x) for x in a.cases.split(",") if x]
    for f in a.cases_json:
        doc = json.loads(Path(f).read_text())
        cases += [int(c) for c in (doc["cases"] if isinstance(doc, dict) else doc)]
    step_rates = json.loads(Path(a.step_rates).read_text())
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    out = Path(a.out)
    done = {json.loads(line)["case"] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    start = time.time()
    with out.open("a") as fh:
        for c in sorted(set(cases) - done):
            if a.time_budget and time.time() - start > a.time_budget:
                print("time budget reached")
                break
            t1 = time.time()
            try:
                res = {"case": c, **one_case(Path(a.root) / "vital_files" / f"{c:04d}.vital", U.get(c, {}), step_rates)}
            except Exception as ex:  # noqa: BLE001
                res = {"case": c, "error": f"{type(ex).__name__}: {ex}"[:300]}
            res["seconds"] = round(time.time() - t1, 1)
            fh.write(json.dumps(res) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
