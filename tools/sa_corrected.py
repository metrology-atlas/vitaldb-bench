# SPDX-License-Identifier: MIT
"""Validate the stamp sawtooth model (physiosync.stamp_clock) with the shared bursts. Per case, this machine.

    python tools/sa_corrected.py --root R --cases 1,2 --units units.csv --out out.jsonl

Each track's record stamps are refitted (one rate per recording, one offset per stretch between gaps) and every
sample placed on the fitted line; the ECG~EEG burst cross-correlation of shared_artifacts is then redone on
those times. VitalDB-specific choices: SNUADC corrects downwards (correction_sign -1) and BIS upwards (+1);
variant "teeth": where a track has no whole tooth, its rate is the sync_full rate of that case ("given");
variant "model" (the adopted one): where SNUADC has no whole tooth and no correction its stamps are trusted
(rate 0: frequent-small-correction mode); with step-mode corrections (an event of -0.4 s or more) but no whole tooth, the port's typical rate
(--step-rates, from the cases with teeth); and where BIS has none, its unit's mean rate (else the
case's sync_full rate) is used. Variant "syncfull" uses the sync_full rate for both devices throughout.
Variant "model2" (19 Oct) is "model" except that a SNUADC track with no correction at all takes its port's typical
step-mode rate unless the recording is in frequent-small-correction mode (--modes label "small"): the whole-release
burst check showed such recordings drift like step mode (they end before their first reset). Ports come from
--ports (every recording) when given, else from --units.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from physiosync import corrected_times, stamp_clock
from shared_artifacts import BAND, DT, burst_probe, envelope
from timenet_connectors.datasets.vitaldb.vitaldb.vital_file import read_vital

TR = {"ecg": ("SNUADC/ECG_II", "snuadc", -1), "eeg1": ("BIS/EEG1_WAV", "bis", 1), "eeg2": ("BIS/EEG2_WAV", "bis", 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="teeth,syncfull")
    ap.add_argument("--step-rates", default=None, help="JSON port -> typical SNUADC step-mode rate (ppm), '*' for any")
    ap.add_argument("--modes", default=None, help="JSON case -> SNUADC correction mode label (step / small / none)")
    ap.add_argument("--ports", default=None, help="JSON case -> {port: SNUADC port, ...} for every recording")
    a = ap.parse_args()
    step_rates = json.loads(Path(a.step_rates).read_text()) if a.step_rates else {}
    modes = {int(k): v for k, v in json.loads(Path(a.modes).read_text()).items()} if a.modes else {}
    ports = {int(k): (v or {}).get("port") for k, v in json.loads(Path(a.ports).read_text()).items()} if a.ports else {}
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["case"] for l in Path(a.out).read_text().splitlines() if l.strip()}
    with open(a.out, "a") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            t1 = time.time()
            res: dict = {"case": c}
            try:
                v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={n for n, _, _ in TR.values()})
                u = U.get(c, {})
                raw = {}
                for k, (n, dev, sign) in TR.items():
                    tr = v.tracks.get(n)
                    if tr is None or not tr.record_times.size:
                        continue
                    x = tr.physical().astype(np.float32)
                    x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
                    raw[k] = (tr, x, float(tr.srate), dev, sign)
                if not {"ecg", "eeg1"} <= set(raw):
                    raise ValueError("no ECG or EEG")
                for variant in a.variants.split(","):
                    times, clocks = {}, {}
                    for k, (tr, x, fs, dev, sign) in raw.items():
                        st = tr.record_offsets[: tr.record_times.size]
                        known = u.get(f"{dev}_drift_ppm")
                        known = float(known) if known not in (None, "") else None
                        unit = u.get(f"{dev}_unit_mean_ppm")
                        unit = float(unit) if unit not in (None, "") else None
                        if variant in ("model", "model2"):
                            clk = stamp_clock(tr.record_times, st, fs, correction_sign=sign)
                            if clk["rate_source"] == "fit":
                                if dev == "snuadc":
                                    # no step-mode correction: stamps trusted; step-mode corrections but no whole
                                    # tooth: the port's typical step-mode rate
                                    ev = clk["corrections"]["median_event_s"]
                                    port = ports.get(c) or u.get("snuadc_port")
                                    port_rate = step_rates.get(port, step_rates.get("*", 0.0))
                                    if ev is not None and ev <= -0.4:
                                        fall = port_rate
                                    elif variant == "model2" and ev is None and modes.get(c) != "small":
                                        fall = port_rate      # no correction at all: step mode before its first reset
                                    else:
                                        fall = 0.0
                                else:
                                    fall = unit if unit is not None else known
                                if fall is not None:
                                    clk = stamp_clock(tr.record_times, st, fs, rate_ppm=fall, correction_sign=sign)
                        elif variant == "syncfull" and known is not None:
                            clk = stamp_clock(tr.record_times, st, fs, rate_ppm=known, correction_sign=sign)
                        else:
                            clk = stamp_clock(tr.record_times, st, fs, correction_sign=sign)
                            if clk["rate_source"] == "fit" and known is not None:
                                clk = stamp_clock(tr.record_times, st, fs, rate_ppm=known, correction_sign=sign)
                        times[k] = corrected_times(clk, st, tr.record_sizes[: st.size])[: x.size]
                        clocks[k] = {"rate_ppm": round(clk["rate_ppm"], 2), "source": clk["rate_source"], "teeth": clk["teeth"],
                                     "events": clk["corrections"]["events"], "gaps": clk["gaps"],
                                     "p2p_ms": round(clk["sawtooth_p2p_s"] * 1000, 1)}
                    t0 = float(np.floor(min(t[0] for t in times.values())))
                    n = int((max(t[-1] for t in times.values()) - t0) / DT) + 1
                    env = {k: envelope(times[k], raw[k][1][: times[k].size], raw[k][2], BAND[k], t0, n) for k in times}
                    res[variant] = {"clocks": clocks, "t0": t0, "burst": burst_probe(env, 600, 300, 20)}
                    del env
                res["sync_full"] = {"snuadc": u.get("snuadc_drift_ppm"), "bis": u.get("bis_drift_ppm")}
            except Exception as ex:  # noqa: BLE001
                res["error"] = f"{type(ex).__name__}: {ex}"[:300]
            res["seconds"] = round(time.time() - t1, 1)
            fh.write(json.dumps(res) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
