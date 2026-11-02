# SPDX-License-Identifier: MIT
"""Which breath is which: the release-wide Primus breath matching with the breath-identity evidence kept.

    bash tools/run_tool_scan.sh primus_breath_id --modes $W/results/sync_full/snuadc_mode_label.json \
        --ports $W/results/sync_full/snuadc_mode.json --step-rates $W/results/pat/snuadc_step_rates.json
    (writes $W/primus_breath_id/part_*.jsonl; rerunning resumes)

Why. tools/primus_sync_accuracy.py finds each 10-minute window's AWP-to-reference lag within half a breath of a
coarse envelope estimate and starts each window from the previous one. Ventilation is regular, so a stretch that
starts on the wrong breath stays on it; about a quarter to a third of its stretches sit a whole breath off
(atlas study primus-awp-anchoring). This scan repeats that lag measurement unchanged and adds the evidence needed
to choose the breath afterwards, without another rerun:

  - peaks: per window, the local maxima of the shape correlation between the breath-triggered averages of AWP and
    the reference over the whole lag range (-3 to +8 s), each as [lag_s, r], strongest first, and the window's
    breath period T (median onset interval);
  - match: per 30-minute block of each stretch, the pilot's breath-identity check (tools/vent_sync_pilot.py
    breath_match): the breath-to-breath size of the reference's response, placed by the tracked lag, against the
    peak airway pressure of breath i + k, k = -3..3; if the tracking is k breaths off the correlation peaks at k.
    Rows: [mid_s, stretch, k, c_best, c_second, n_breaths, shift_ms] (times from the first AWP sample).

Arms: CVP and the arterial mean pressure (ART low-passed over 2 s), SNUADC on its stamp model, as in
primus_sync_accuracy. Lag = reference time minus AWP stamp of the same breath (bias = -lag).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppv_by_breath as P  # noqa: E402
from pat import sample_times  # noqa: E402
from primus_sync_accuracy import lowpass_mean  # noqa: E402
from vent_sync_pilot import G, binned, breath_match, trig_avg  # noqa: E402

TAUS = np.arange(-4.0, 12.0, G)
CORE = np.flatnonzero((TAUS >= -1.0) & (TAUS <= 4.0))


def profile_peaks(aa, cc, top=5):
    """Shape correlation of the AWP average against the reference average shifted by L, over lags -3..+8 s, and its
    local maxima [lag_s, r] (strongest first)."""
    a = ndimage.uniform_filter1d(np.nan_to_num(aa), 10)
    c = ndimage.uniform_filter1d(np.nan_to_num(cc), 10)
    Ls = np.arange(int(-3.0 / G), int(8.0 / G) + 1)
    rs = np.full(Ls.size, np.nan)
    x = a[CORE] - a[CORE].mean()
    sx = np.sqrt((x ** 2).sum())
    for i, L in enumerate(Ls):
        ic = CORE + L
        if ic[0] < 0 or ic[-1] >= c.size or sx == 0:
            continue
        y = c[ic] - c[ic].mean()
        sy = np.sqrt((y ** 2).sum())
        if sy > 0:
            rs[i] = float((x * y).sum() / (sx * sy))
    pk = [i for i in range(1, Ls.size - 1) if np.isfinite(rs[i]) and rs[i] >= np.nanmax(rs[i - 1:i + 2]) and rs[i] > 0.3]
    pk = sorted(pk, key=lambda i: -rs[i])[:top]
    return [[round(float(Ls[i] * G), 2), round(float(rs[i]), 3)] for i in pk]


def scan_case(c, a, modes, ports, step_rates):
    from vitaldb_reader.vital_file import read_vital
    v = read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital", track_names={"Primus/AWP", "SNUADC/CVP", "SNUADC/ART"})
    aw = v.tracks.get("Primus/AWP")
    refs = {k: v.tracks.get(f"SNUADC/{k.upper()}") for k in ("cvp", "art")}
    refs = {k: tr for k, tr in refs.items() if tr is not None and tr.record_times.size}
    if aw is None or not aw.record_times.size or not refs:
        return {"case": c, "skip": "no AWP, or neither CVP nor ART"}
    ta = sample_times(aw)
    xa = np.nan_to_num(aw.physical().astype(np.float64))[: ta.size]
    on = P.breath_onsets(ta, xa, float(aw.srate))
    if on.size < 200:
        return {"case": c, "skip": "fewer than 200 mechanical breaths"}
    z = float(ta[0])
    g = np.diff(ta)
    pauses = ta[1:][g > 5]
    res = {"case": c, "z": z, "pauses": (pauses - z).round(1).tolist(), "arms": {}}
    for k, tr in refs.items():
        tc = P.model_times(tr, ports.get(c), modes.get(c), step_rates)
        xc = tr.physical().astype(np.float64)[: tc.size]
        if k == "art":
            xc = lowpass_mean(np.nan_to_num(xc, nan=float(np.nanmedian(xc))), float(tr.srate))
        tl, ll = P.cvp_lag(ta, xa, tc, xc, on)
        arm = {"t": (tl - z).round(1).tolist(), "lag": ll.round(4).tolist(), "T": [], "peaks": [], "match": []}
        if tl.size:
            t0 = float(np.floor(min(ta[0], tc[0])))
            n50 = int((max(ta[-1], tc[-1]) - t0) / G) + 1
            okc = np.isfinite(xc)
            awp50, ref50 = binned(ta, xa, t0, G, n50), binned(tc[okc], xc[okc], t0, G, n50)
            for mid in tl:
                o = on[(on >= mid - 300) & (on < mid + 300)]
                arm["T"].append(round(float(np.median(np.diff(o))), 3) if o.size > 2 else None)
                arm["peaks"].append(profile_peaks(trig_avg(awp50, t0, o, TAUS), trig_avg(ref50, t0, o, TAUS)))
            rows = [[float(t - t0), float(l * 1000.0), 1.0, 0.0] for t, l in zip(tl, ll)]
            sst = np.sort(pauses - t0)
            for r in breath_match(on, awp50, ref50, t0, rows, sst):
                r[0] = round(float(r[0] + t0 - z), 1)
                arm["match"].append(r)
        res["arms"][k] = arm
    return res


def scan(a):
    modes = {int(k): v for k, v in json.loads(Path(a.modes).read_text()).items()}
    ports = {int(k): (v or {}).get("port") for k, v in json.loads(Path(a.ports).read_text()).items()}
    step_rates = json.loads(Path(a.step_rates).read_text())
    out = Path(a.out)
    done = {json.loads(l)["case"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t_start = time.time()
    with open(out, "a", encoding="utf-8") as fh:
        for c in [int(x) for x in a.cases.split(",") if x]:
            if c in done:
                continue
            if a.time_budget and time.time() - t_start > a.time_budget:
                break
            t1 = time.time()
            try:
                r = scan_case(c, a, modes, ports, step_rates)
            except Exception as err:  # noqa: BLE001
                r = {"case": c, "error": f"{type(err).__name__}: {err}"[:300]}
            r["secs"] = round(time.time() - t1, 1)
            fh.write(json.dumps(r) + "\n")
            fh.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    for k in ("root", "cases", "modes", "ports", "step_rates", "out"):
        s.add_argument("--" + k.replace("_", "-"), required=True)
    s.add_argument("--time-budget", type=float, default=0)
    a = ap.parse_args()
    scan(a)


if __name__ == "__main__":
    main()
