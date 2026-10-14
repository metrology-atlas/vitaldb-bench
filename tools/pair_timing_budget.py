# SPDX-License-Identifier: MIT
"""Timing budget for combining two or more VitalDB time sources (aggregates only; no data read).

    python tools/pair_timing_budget.py snu_model pleth            # ECG + PPG from SNUADC
    python tools/pair_timing_budget.py snu_model hr_lab --resolve-ms 5
    python tools/pair_timing_budget.py --list
    python tools/pair_timing_budget.py --selftest

Reads the ellipses of Figure 28 (ENTRIES in time_source_accuracy_figure.py): x = delay from the end of the
interval a value describes to its stamp, [x_lo, x_hi]; y = width of that interval, [y_lo, y_hi] (s).

First-order model (a screening tool, not a fit):
  * Each source's delay is corrected to the middle of its x range; what is left is treated as uniform,
    sigma_delay = (x_hi - x_lo) / sqrt(12). An uncorrected delay is a bias, reported separately.
  * Where in its interval a value's time lies is uniform over the interval, sigma_res = y_hi / sqrt(12).
  * Sources are taken as independent; sources on one clock (CLOCK) share drift, which then cancels.
  sigma_rel        = sqrt(sum of both terms over all sources)
  coherent band    = 0.1325 / sigma_rel      (-3 dB of a Gaussian timing blur)
  finest grid      = sqrt(12) * sigma_rel   (a shared grid finer than this adds no cross-channel information)
  latency floor    = max x_hi               (availability rule: a joint vector at t can only hold windows that
                                             ended at least this long before t. For *_lab and *_sync entries x is
                                             the residual after correction, so add the corrected delay itself)
  oldest data      = max(x_hi + y_hi)       (age of the oldest sample feeding the joint vector)
  beats to average = (sigma_rel / resolve)^2 (random part only; bias does not average away)
  longest window   = sigma_rel / |drift_a - drift_b|  (only between clocks with known drift and no resync)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from time_source_accuracy_figure import ENTRIES  # noqa: E402

# Clock domain of each entry. Entries on the reference (SNUADC stamp-model time, or synchronised to it) share
# the "ref" domain. Raw stamps keep their device clock. Unlisted entries are treated as their own clock.
CLOCK = {
    "snu_model": "ref", "pleth": "ref", "bis_sync": "ref", "primus_sync": "ref", "cardioq_sync": "ref",
    "ev1000_sync": "ref",
    # a per-case delay label removes a fixed offset only, so drift on the device clock remains
    "solar_lab": "solar", "hr_lab": "solar", "primus_lab": "primus",
    "snu_raw": "snuadc_raw", "bis_raw": "bis", "primus_raw": "primus", "primus_num": "primus",
    "cardioq_raw": "cardioq", "ev1000": "ev1000", "solar_win": "solar", "hr_early": "solar",
    "hr_late": "solar", "pleth_hr": "solar", "spo2": "solar", "utc": "utc",
}
# Drift against the reference between corrections (ppm). None = not known. Synchronised entries are 0 by
# construction. BIS from its stamp model (measured); Primus 115 ppm is ASSUMED (see the measured-or-assumed table).
DRIFT_PPM = {"ref": 0.0, "bis": 23.0, "primus": 115.0}


def budget(keys, resolve_s=None):
    e = [ENTRIES[k] for k in keys]
    sd = [(x[3] - x[2]) / math.sqrt(12) for x in e]
    sr = [x[5] / math.sqrt(12) for x in e]
    s = math.sqrt(sum(v * v for v in sd + sr))
    out = {
        "sigma_rel_s": s,
        "sigma_delay_s": dict(zip(keys, sd)),
        "sigma_res_s": dict(zip(keys, sr)),
        "bias_if_uncorrected_s": {k: (x[2] + x[3]) / 2 for k, x in zip(keys, e)},
        "coherent_band_hz": 0.1325 / s if s > 0 else math.inf,
        "finest_grid_s": math.sqrt(12) * s,
        "latency_floor_s": max(x[3] for x in e),
        "oldest_data_s": max(x[3] + x[5] for x in e),
        "estimated_inputs": [k for k, x in zip(keys, e) if not x[6]],
    }
    if resolve_s:
        out["beats_to_average"] = math.ceil((s / resolve_s) ** 2)
    doms = {k: CLOCK.get(k, k) for k in keys}
    if len(set(doms.values())) == 1:
        out["longest_window_s"] = math.inf
        out["drift_note"] = "one clock: drift cancels"
    else:
        r = [DRIFT_PPM.get(d) for d in doms.values()]
        if None in r:
            out["longest_window_s"] = None
            out["drift_note"] = "drift unknown for " + ", ".join(k for k, d in doms.items() if DRIFT_PPM.get(d) is None)
        else:
            dr = (max(r) - min(r)) * 1e-6
            out["longest_window_s"] = s / dr if dr else math.inf
            out["drift_note"] = f"relative drift {max(r) - min(r):g} ppm, no resync within the window"
    return out


def fmt(v):
    if v is None:
        return "unknown"
    if v == math.inf:
        return "unbounded"
    return f"{v * 1e3:.1f} ms" if v < 1 else f"{v:.2f} s"


def report(keys, resolve_s=None):
    b = budget(keys, resolve_s)
    print("Sources: " + ", ".join(f"{k} ({ENTRIES[k][1].replace(chr(10), ' ')})" for k in keys))
    for k in keys:
        print(f"  {k:14s} delay spread {fmt(b['sigma_delay_s'][k]):>10s}   interval {fmt(b['sigma_res_s'][k]):>10s}"
              f"   bias if uncorrected {fmt(b['bias_if_uncorrected_s'][k])}")
    print(f"Relative timing sigma        {fmt(b['sigma_rel_s'])}")
    print(f"Coherent cross-channel band  {b['coherent_band_hz']:.3g} Hz")
    print(f"Finest useful shared grid    {fmt(b['finest_grid_s'])}")
    print(f"Latency floor (avail. rule)  {fmt(b['latency_floor_s'])}")
    print(f"Oldest data in the vector    {fmt(b['oldest_data_s'])}")
    if "beats_to_average" in b:
        print(f"Beats to average to resolve  {b['beats_to_average']}  (random part only)")
    print(f"Longest window before drift  {fmt(b['longest_window_s'])}  ({b['drift_note']})")
    if b["estimated_inputs"]:
        print("Caution: estimated, not measured: " + ", ".join(b["estimated_inputs"]))


def selftest():
    b = budget(["snu_model", "pleth"])
    # PLETH: delay spread 30 ms -> 8.7 ms; 30 Hz frame 33 ms -> 9.5 ms; SNUADC 10 ms interval -> 2.9 ms
    assert 0.012 < b["sigma_rel_s"] < 0.015, b["sigma_rel_s"]
    assert b["longest_window_s"] == math.inf
    assert abs(b["latency_floor_s"] - 0.40) < 1e-9
    b = budget(["bis_raw", "primus_raw"])
    assert b["longest_window_s"] is not None and b["longest_window_s"] < math.inf
    assert budget(["snu_model", "vigilance"])["longest_window_s"] is None
    assert budget(["snu_model", "pleth"], 0.005)["beats_to_average"] >= 6
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("keys", nargs="*", help="two or more ENTRIES keys")
    ap.add_argument("--resolve-ms", type=float, help="timing difference to resolve, e.g. a PAT change (ms)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.list or len(a.keys) < 2:
        for k, e in ENTRIES.items():
            print(f"{k:14s} {CLOCK.get(k, k):11s} {e[1].replace(chr(10), ' ')}")
        return
    bad = [k for k in a.keys if k not in ENTRIES]
    if bad:
        sys.exit("unknown keys: " + ", ".join(bad) + " (see --list)")
    report(a.keys, a.resolve_ms / 1e3 if a.resolve_ms else None)


if __name__ == "__main__":
    main()
