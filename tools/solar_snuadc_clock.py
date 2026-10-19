# SPDX-License-Identifier: MIT
"""The SNUADC against the recorder PC, through the Solar8000 numerics the PC stamps on arrival.

    python tools/solar_snuadc_clock.py scan --root R --cases 1,2 --modes M --ports P --step-rates S --out part.jsonl
    python tools/solar_snuadc_clock.py agg part*.jsonl [--resets CSV] [--out DIR]

The Solar8000's heart rate and mean arterial pressure are computed by the monitor from the same ECG and arterial
pressure that SNUADC samples from its analogue outputs, and the recorder stamps each value on arrival with the PC
clock. Their delay behind the waveform is fixed within a case (numeric-delay-variability), so if the waveform is
placed on its own sample count, the delay ramps at the SNUADC's drift against the PC, measured without the SNUADC's
stamps or resets. The 66-case pilot (numeric-clock-check) gave -65.3 ppm against -66.7 from the stamp model.

Per recording, two timelines for the SNUADC samples:
  counts  the sample count (physiosync.stamp_clock with the rate held at 0: one offset per stretch between gaps,
          the 17.7 ms lost at each reset added back), so the delay's slope is the SNUADC crystal against the PC;
  model   the adopted stamp model (rules of model2: a track without a whole tooth takes its port's step-mode rate,
          unless the recording is in frequent-correction mode), so the slope is what the model leaves.
For HR (60 / median R-R over 6 s, truncated) and ART_MBP (beat-mean EWMA 1/4), the delay is fitted for the whole
case and per 30-minute block (tools/numeric_delay_variability.py, same likelihood). agg fits each recording's
block delays with a robust line (Theil-Sen, since the Solar's own delay occasionally steps; blocks whose profile
interval touches the search edge left out; at least 4 blocks over 1.5 h), and reports the rate by SNUADC mode, its agreement with the reset intervals (--resets, the per-recording
drift of signal_survey/continuity/snuadc_reset_drift_per_recording.csv), and the model's leftover slope.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import numeric_delay_variability as nd  # noqa: E402
import numeric_derivation_pilot as ndp  # noqa: E402

warnings.filterwarnings("ignore")
LOSS_S = 0.0177
TRACKS = ("HR", "ART_MBP")


def timelines(tr, port, mode, step_rates):
    from physiosync import corrected_times, stamp_clock
    st = tr.record_offsets[: tr.record_times.size]
    fs = float(tr.srate)
    clk = stamp_clock(tr.record_times, st, fs, correction_sign=-1, event_loss_s=LOSS_S)
    counts = corrected_times(dict(clk, rate_ppm=0.0), st, tr.record_sizes[: st.size])
    info = {"rate_ppm": round(clk["rate_ppm"], 2), "source": clk["rate_source"], "teeth": clk["teeth"],
            "events": clk["corrections"]["events"], "median_event_s": clk["corrections"]["median_event_s"]}
    if clk["rate_source"] == "fit":
        ev = clk["corrections"]["median_event_s"]
        pr = step_rates.get(port, step_rates.get("*", 0.0))
        fall = pr if (ev is not None and ev <= -0.4) or (ev is None and mode != "small") else 0.0
        clk = stamp_clock(tr.record_times, st, fs, rate_ppm=fall, correction_sign=-1, event_loss_s=LOSS_S)
        info["model_rate_ppm"] = round(fall, 2)
    model = corrected_times(clk, st, tr.record_sizes[: st.size])
    return {"counts": counts, "model": model}, info


def scan_case(c, a, modes, ports, step_rates):
    v = ndp.read_vital(Path(a.root) / "vital_files" / f"{c:04d}.vital",
                       track_names={"SNUADC/ECG_II", "SNUADC/ART", *(ndp.SOLAR[k] for k in TRACKS)})
    ecg, art = v.tracks.get("SNUADC/ECG_II"), v.tracks.get("SNUADC/ART")
    if ecg is None or not ecg.record_times.size or float(ecg.srate) < 400:
        return {"case": c, "skip": "no 500 Hz ECG"}
    if not any(v.tracks.get(ndp.SOLAR[k]) is not None for k in TRACKS):
        return {"case": c, "skip": "no Solar HR or ART_MBP"}
    te_all, info = timelines(ecg, ports.get(c), modes.get(c), step_rates)
    ta_all = timelines(art, ports.get(c), modes.get(c), step_rates)[0] if art is not None and art.record_times.size else None
    xe = ecg.physical().astype(np.float64)
    xa = art.physical().astype(np.float64) if ta_all else None
    res = {"case": c, "snuadc": info, "mode": modes.get(c), "timelines": {}}
    for tl in ("counts", "model"):
        te = te_all[tl]
        bs = ndp.beat_series(te, xe[: te.size], float(ecg.srate),
                             (ta_all[tl], xa[: ta_all[tl].size]) if ta_all else None, None)
        out = {}
        for name in TRACKS:
            algo, _, _, _, key = nd.ALGO[name]
            ser = bs.get(key)
            tr = v.tracks.get(ndp.SOLAR[name])
            if tr is None or ser is None or ser[0].size < 200:
                continue
            ot, ov, og = ndp.numeric_obs(tr, *ndp.VALID[name])
            m = (ot > ser[0][0] + 60) & (ot < ser[0][-1])
            ot, ov, og = ot[m], ov[m], og[m]
            if ot.size < 300:
                continue
            tc, y = nd.series(algo, ser[0], ser[1])
            r = nd.track_delays(name, ot, ov, og, tc, y, bs["rr"])
            if r:
                out[name] = {"d": r["d"], "blocks": [[b["h"], b["d"], b["ci"][0], b["ci"][1], b["edge"]] for b in r["blocks"]]}
        res["timelines"][tl] = out
        del bs
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


def slope_ppm(blocks):
    """Theil-Sen slope of the block delays (robust to the occasional step in the Solar's own delay), in ppm,
    with a standard error from its 68% interval."""
    from scipy.stats import theilslopes
    B = np.array([b for b in blocks if not b[4]], float)
    if len(B) < 4 or np.ptp(B[:, 0]) < 1.5:
        return None
    s, _, lo, hi = theilslopes(B[:, 1], B[:, 0] * 3600, alpha=0.68)
    return float(s * 1e6), float(max((hi - lo) / 2 * 1e6, 0.5))


def agg(a):
    import pandas as pd
    R = [json.loads(l) for p in a.parts for l in open(p, encoding="utf-8") if l.strip()]
    rows = []
    for r in R:
        for tl, tracks in (r.get("timelines") or {}).items():
            for name, t in tracks.items():
                s = slope_ppm(t["blocks"])
                if s:
                    rows.append(dict(case=r["case"], timeline=tl, track=name, ppm=s[0], se=s[1], mode=r.get("mode"),
                                     source=r["snuadc"]["source"], events=r["snuadc"]["events"],
                                     stamp_rate=r["snuadc"]["rate_ppm"]))
    D = pd.DataFrame(rows)
    pct = lambda s: [round(float(x), 1) for x in np.percentile(s, [10, 25, 50, 75, 90])] + [int(len(s))] if len(s) else None
    out = {"recordings_read": len(R), "errors": sum(1 for r in R if "error" in r),
           "skipped": {k: sum(1 for r in R if r.get("skip") == k) for k in {r.get("skip") for r in R if r.get("skip")}}}
    if len(D):
        for tl in ("counts", "model"):
            for name in TRACKS:
                d = D[(D.timeline == tl) & (D.track == name)]
                out[f"{tl}_{name}"] = {"ppm_by_mode": {m: pct(g.ppm) for m, g in d.groupby(d["mode"].fillna("unknown"))},
                                       "se_ppm_median": round(float(d.se.median()), 1) if len(d) else None}
        # combine HR and ART_MBP per recording (inverse-variance) on the counts timeline
        c = D[D.timeline == "counts"].copy()
        c["w"] = 1 / c.se ** 2
        comb = c.groupby("case").apply(lambda g: pd.Series({"ppm": float(np.sum(g.w * g.ppm) / g.w.sum()),
                                                            "se": float(1 / np.sqrt(g.w.sum())), "mode": g["mode"].iloc[0]}))
        out["counts_combined_by_mode"] = {m: pct(g.ppm) for m, g in comb.groupby(comb["mode"].fillna("unknown"))}
        if a.resets:
            s = pd.read_csv(a.resets)[["case", "ppm"]].rename(columns={"ppm": "reset_ppm"}).set_index("case")
            j = comb.join(s, how="inner")
            j["diff"] = j.ppm - j.reset_ppm
            out["closure_vs_reset_intervals"] = {"recordings": int(len(j)), "solar_minus_resets_ppm": pct(j["diff"]),
                                                 "z_abs_median": round(float((j["diff"].abs() / j.se).median()), 2)}
        Path(a.out or ".").mkdir(parents=True, exist_ok=True)
        if a.out:
            D.to_csv(Path(a.out) / "slopes.csv", index=False)
            comb.to_csv(Path(a.out) / "counts_combined.csv")
    print(json.dumps(out, indent=1))
    if a.out:
        (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    for k in ("root", "cases", "modes", "ports", "step_rates", "out"):
        s.add_argument("--" + k.replace("_", "-"), required=True)
    s.add_argument("--time-budget", type=float, default=0)
    g = sp.add_parser("agg"); g.add_argument("parts", nargs="+"); g.add_argument("--resets"); g.add_argument("--out")
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else agg(a)


if __name__ == "__main__":
    main()
