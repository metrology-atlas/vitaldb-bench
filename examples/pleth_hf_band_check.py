"""Why did the cautery count with a PLETH high-frequency rise halve between the 28-29 Oct and 10 Nov release runs?

spo2_event_pilot.py's envelope() has capped the band's upper edge at 0.45 x fs since 56ca1b2 (29 Oct, for 100 Hz
files). At 500 Hz that also narrows the PLETH band from 30-240 Hz to 30-225 Hz. This runs run_case on given
recordings with the tool as it is (cap 0.45 fs) and with the cap at 0.48 fs (240 Hz at 500 Hz, the 28 Oct band),
and counts cautery events with pleth_hf_rise. Everything else is the tool's own code (10 Nov 2026).
After the fix (the cap applies only when the band reaches Nyquist), "tool_0.45fs" is the fixed tool and should
equal "repl_0.48fs" at 500 Hz.

  python examples/pleth_hf_band_check.py --root R --units U --step-rates S --cases 4150,6368 --out OUT.jsonl
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import spo2_event_pilot as sp  # noqa: E402
from fallback_rate import load_step_rates  # noqa: E402

ORIG = sp.envelope


def envelope_cap(cap):
    def env(t, x, fs, band, t0, n):
        ok = np.isfinite(x)
        x = np.where(ok, x, 0.0)
        band = (band[0], min(band[1], cap * fs))
        sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
        y = signal.sosfiltfilt(sos, x) ** 2
        b = np.floor((t - t0) / sp.DT).astype(np.int64)
        k = (b >= 0) & (b < n) & ok
        s = np.bincount(b[k], y[k], minlength=n)
        c = np.bincount(b[k], minlength=n)
        m = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        e = np.log10(m + 1e-12)
        e[~np.isfinite(m)] = np.nanmedian(e)
        e = ndimage.median_filter(e, 11)
        return e - ndimage.median_filter(e, 201)
    return env


def summary(res):
    ev = res.get("cautery") or []
    pl = res.get("pleth_only") or []
    return {"cautery": len(ev), "pleth_hf_rise": sum(bool(e.get("pleth_hf_rise")) for e in ev),
            "pleth_only": len(pl) if isinstance(pl, list) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--step-rates", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    rates = load_step_rates(a.step_rates)
    with open(a.out, "a") as f:
        for c in [int(x) for x in a.cases.split(",")]:
            row = {"case": c}
            t1 = time.time()
            # check first that the replacement reproduces the tool at its own cap
            for label, env in (("tool_0.45fs", ORIG), ("repl_0.45fs", envelope_cap(0.45)), ("repl_0.48fs", envelope_cap(0.48))):
                sp.envelope = env
                row[label] = summary(sp.run_case(a.root, c, U.get(c, {}), rates.for_case(c), a.seed))
            sp.envelope = ORIG
            row["secs"] = round(time.time() - t1, 1)
            f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
