# SPDX-License-Identifier: MIT
"""SNUADC rate from the mains hum, for recordings whose stamps carry no correction. Per case, this machine only;
the summary leaves.

    python tools/hum_snuadc_rate.py --table per_recording.csv --bursts burst_per_recording.csv \
        --rates hum_rates.csv --summary summary.json [--check-out check.csv]

Without stamp corrections the SNUADC samples are put on the recorder PC's timeline with an assumed rate (the port's
typical rate), and drift on it. The hum gives the SNUADC rate against the grid (line = f_grid - f_SNUADC, from
tools/mains_hum.py agg); adding the PC's rate against the grid, g = f_PC - f_grid, gives the rate against the PC:

    rate_ppm = line + g_setup + k

g_setup is the median g of the recordings with corrections in the same setup (PC key; else the same SNUADC port;
else all; at least 10 recordings). k is one constant, calibrated on recordings with corrections whose BIS-against-ECG
burst drift is known (tools/burst_validation_agg.py, both rates from their own corrections): which recordings carry
corrections is selected by their rate, so the pool's g is not representative of the recordings the rate is applied
to. rate_ppm is in physiosync's convention (r = f_PC - f_SNUADC; negative = SNUADC fast) and goes to
physiosync.stamp_clock(..., rate_ppm=rate_ppm) in place of the assumed rate.

Check: on recordings without SNUADC corrections whose BIS rate comes from its own corrections (none of them used
for g or k), the burst drift left by the assumed rate against that left by the hum rate. Whole release, 29 Oct 2026:
224 recordings; zero-centred robust SD 31 ppm with the assumed rate, 13 ppm with the hum rate (median 76 against
31 ms/h); 1,248 recordings without corrections have hum.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def z(v):
    v = np.asarray(v, float)
    return round(float(1.4826 * np.median(np.abs(v))), 1)


def g_setup(row, pool, exclude=None):
    for col in ("key", "snu_port"):
        p = pool[(pool[col] == row[col]) & (pool.case != exclude)].g
        if len(p) >= 10:
            return float(np.median(p)), col
    return float(np.median(pool[pool.case != exclude].g)), "all"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", required=True, help="per_recording.csv from tools/mains_hum.py agg")
    ap.add_argument("--bursts", required=True, help="per_recording.csv from tools/burst_validation_agg.py")
    ap.add_argument("--rates", required=True); ap.add_argument("--summary", required=True)
    ap.add_argument("--check-out", help="per-recording errors of the check, for tools/hum_snuadc_plot.py (stays local)")
    a = ap.parse_args()
    m = pd.read_csv(a.table); b = pd.read_csv(a.bursts)
    pool = m[(m.snu_events >= 1) & m.snu_line.notna()].copy(); pool["g"] = pool.snu_rate - pool.snu_line

    def with_hum_rate(t):
        gs = [g_setup(r, pool, exclude=r.case) for _, r in t.iterrows()]
        t = t.copy(); t["g_ppm"] = [g for g, _ in gs]; t["g_source"] = [s for _, s in gs]
        t["x"] = t.snu_line + t.g_ppm - t.ecg_rate          # hum rate minus the rate the stamp model used
        t["y"] = t.slope_ms_h / 3.6                         # burst drift left by that rate, ppm
        return t

    j = b.merge(m, on="case")
    j = j[j.snu_line.notna() & (j.eeg_source == "teeth")]
    cal = with_hum_rate(j[(j.ecg_source == "teeth")])
    k = float(np.median(cal.y - cal.x))
    test = with_hum_rate(j[(j.ecg_source == "given") & j["mode"].isin(["step", "none"])])
    e_given, e_hum = test.y, test.y - test.x - k
    if a.check_out:
        test.assign(err_assumed_ppm=e_given, err_hum_ppm=e_hum, hum_minus_assumed_ppm=test.x + k)[
            ["case", "mode", "err_assumed_ppm", "err_hum_ppm", "hum_minus_assumed_ppm"]].round(2).to_csv(a.check_out, index=False)

    out = m[(m.snu_events == 0) & m.snu_line.notna()].copy()
    gs = [g_setup(r, pool) for _, r in out.iterrows()]
    out["g_ppm"] = [round(g, 2) for g, _ in gs]; out["g_source"] = [s for _, s in gs]
    out["constant_ppm"] = round(k, 2)
    out["rate_ppm"] = (out.snu_line + out.g_ppm + k).round(2)
    out[["case", "snu_line", "g_ppm", "g_source", "constant_ppm", "rate_ppm"]].rename(
        columns={"snu_line": "line_ppm"}).to_csv(a.rates, index=False)

    def stats(e):
        e = np.asarray(e, float)
        return {"n": int(e.size), "median_ppm": round(float(np.median(e)), 1), "zero_robust_sd_ppm": z(e),
                "median_abs_ms_per_h": round(float(3.6 * np.median(np.abs(e)))),
                "share_over_0.1_s_per_h": round(float(np.mean(np.abs(e) > 100 / 3.6)), 3)}
    summ = {"pool_with_corrections": int(len(pool)), "pool_g_ppm_median": round(float(pool.g.median()), 1),
            "constant_ppm": round(k, 1), "calibration_recordings": int(len(cal)),
            "test": {"assumed_rate": stats(e_given), "hum_rate": stats(e_hum),
                     "by_mode": {md: {"assumed_rate": stats(s.y), "hum_rate": stats(s.y - s.x - k)}
                                 for md, s in test.groupby("mode")}},
            "rates_written": int(len(out)), "recordings_without_corrections": int((m.snu_events == 0).sum())}
    json.dump(summ, open(a.summary, "w"), indent=1)
    print(json.dumps(summ["test"]["assumed_rate"]), json.dumps(summ["test"]["hum_rate"]), "k", summ["constant_ppm"], "rates", len(out))


if __name__ == "__main__":
    main()
