# SPDX-License-Identifier: MIT
"""SNUADC sample clock against the recorder PC, measured through the PC-stamped Solar8000 HR. This machine only.

    python tools/numeric_clock_check.py --root R --cases-json cases.json --units units.csv \
        --step-rates rates.json --out out.jsonl --time-budget 160

Solar8000/HR is stamped by the recorder when it arrives, so it is on the PC clock with no device clock of its own.
Its delay behind the ECG-derived HR (tools/numeric_delay_variability.py, 30-min blocks, fixed algorithm) is
measured with the ECG placed on two timelines:

- count: first record stamp + sample index / nominal rate (the SNUADC sample clock alone);
- line: the stamp model of tools/clock_model_fit.py (sawtooth line, with the SNUADC signal lost at resets).

On count time the delay ramps at the SNUADC-vs-PC rate (plus 17.7 ms at each reset); on the line it should be
flat. The slope of block delay against elapsed time, in ppm, is a second measurement of the SNUADC rate that
does not use the SNUADC stamps. One JSON line per case; resumable.
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
import clock_model_fit as cmf  # noqa: E402
import numeric_delay_variability as ndv  # noqa: E402
from physiosync import corrected_times  # noqa: E402

ndp = ndv.ndp


def slope_ppm(blocks: list[dict]) -> tuple[float | None, int]:
    """Weighted slope of block delay (s) against block time (h), in ppm; blocks at the search edge are dropped."""
    b = [x for x in blocks if not x.get("edge") and x.get("ci")]
    if len(b) < 4:
        return None, len(b)
    h = np.array([x["h"] for x in b]); d = np.array([x["d"] for x in b])
    w = 1 / np.maximum(np.array([(x["ci"][1] - x["ci"][0]) / 3.92 for x in b]), 0.02) ** 2
    A = np.vstack([np.ones_like(h), h]).T * np.sqrt(w)[:, None]
    coef = np.linalg.lstsq(A, d * np.sqrt(w), rcond=None)[0]
    return float(coef[1] / 3600 * 1e6), len(b)


def one(path: Path, u: dict, step_rates: dict) -> dict:
    v = ndp.read_vital(path, track_names={"SNUADC/ECG_II", ndp.SOLAR["HR"]})
    tr, hr = v.tracks.get("SNUADC/ECG_II"), v.tracks.get(ndp.SOLAR["HR"])
    if tr is None or hr is None or not tr.record_times.size or not hr.record_times.size:
        return {"skip": "no ECG or HR"}
    fs = float(tr.srate)
    x = tr.physical().astype(np.float64)
    st = tr.record_offsets[: tr.record_times.size]
    info, clk = cmf.fit_track(tr, "SNUADC", u, step_rates)
    k = np.arange(x.size, dtype=np.float64)
    lines = {"count": tr.record_times[0] + k / fs,
             "line": corrected_times(clk, st, tr.record_sizes[: st.size])[: x.size]}
    ot, ov, og = ndp.numeric_obs(hr, *ndp.VALID["HR"])
    out = {"stamp_rate_ppm": round(info["rate_ppm"], 2), "rate_tier": info["rate_tier"], "events": info["events"],
           "loss_events": info["loss_events"], "hours": round(x.size / fs / 3600, 2)}
    algo = ndv.ALGO["HR"][0]
    for name, te in lines.items():
        bs = ndp.beat_series(te, x, fs, None, None)
        ser = bs.get("rr")
        if ser is None or ser[0].size < 200:
            return {**out, "skip": "few beats"}
        m = (ot > ser[0][0] + 60) & (ot < ser[0][-1])
        if m.sum() < 300:
            return {**out, "skip": "few HR records"}
        tc, y = ndv.series(algo, ser[0], ser[1])
        r = ndv.track_delays("HR", ot[m], ov[m], og[m], tc, y, bs["rr"])
        if not r:
            return {**out, "skip": "no fit"}
        s, nb = slope_ppm(r["blocks"])
        out[name] = {"d": r["d"], "slope_ppm": None if s is None else round(s, 1), "blocks_used": nb,
                     "blocks": [[b["h"], b["d"], b.get("edge")] for b in r["blocks"]]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases-json", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--step-rates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=160)
    a = ap.parse_args()
    cases = json.loads(Path(a.cases_json).read_text())
    step_rates = json.loads(Path(a.step_rates).read_text())
    U = {int(r["case"]): r for r in csv.DictReader(open(a.units))}
    out = Path(a.out)
    done = {json.loads(x)["case"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    t0, slowest, n = time.time(), 0.0, 0
    with out.open("a") as fh:
        for c in cases:
            if c in done:
                continue
            if time.time() - t0 + max(1.3 * slowest, 10) > a.time_budget:
                break
            t1 = time.time()
            try:
                r = {"case": c, **one(Path(a.root) / "vital_files" / f"{c:04d}.vital", U.get(c, {}), step_rates)}
            except Exception as ex:  # noqa: BLE001
                r = {"case": c, "error": f"{type(ex).__name__}: {ex}"[:200]}
            r["seconds"] = round(time.time() - t1, 1)
            slowest = max(slowest, r["seconds"])
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
    print(json.dumps({"done_now": n, "left": len(cases) - len(done) - n}))


if __name__ == "__main__":
    main()
