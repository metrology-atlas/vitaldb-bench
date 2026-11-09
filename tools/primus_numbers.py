# SPDX-License-Identifier: MIT
"""The numbers of the Primus AWP and CO2 timing forms, from saved outputs, with each stretch on its right breath.

    python tools/primus_numbers.py --work $W --out $W/primus_numbers [--sync DIR] [--uncorrected]

Replaces work/reframe_stage3/primus_numbers.py (a script outside any repository; --uncorrected reproduces its
numbers). Inputs: the per-window AWP-to-CVP lags of tools/primus_sync_accuracy.py ($W/primus_sync_accuracy/
part_*.jsonl; CVP arm, recordings with at least 6 windows) and the breath of each stretch from
tools/primus_breath_rule.py ($W/primus_breath_rule/stretch_k.json and summary.json).

Corrected lag = window lag - k T for the window's stretch (stretches between AWP pauses, as primus_sync_accuracy).
Lag = CVP breath time minus AWP stamp, so bias = -lag.

  - sync form (after breath matching): leave-one-window-out errors of a line per stretch, windows within 5 min of a
    pause left out (primus_stretch_models), as before: a whole-breath shift of a stretch does not change them, so
    they describe a stretch on its right breath. Plus the breath error: the probability that a stretch is still a
    whole breath off, by the step of the rule that set its breath (own pauses: taken as 0; carried and main mode:
    1 - their leave-one-out share right), averaged over the windows; its standard deviation as a mixture,
    sqrt(p) x the median breath period, and the 95% interval of the mixture (the leave-one-out errors, plus a
    whole breath period either way with probability p / 2 each, the period drawn from the windows' own); the
    breath error holds for a whole stretch, whose median length is reported.
  - raw and shift forms: the per-recording median of the corrected window lags, its 5th and 95th percentiles and
    robust SD between recordings (per_recording.json: each recording's median).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from primus_stretch_models import STRADDLE_S, predict, stretch_index  # noqa: E402

rsd = lambda x: float(1.4826 * np.median(np.abs(x)))  # noqa: E731
rsd_c = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))  # noqa: E731


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sync", default=None, help="folder of primus_sync_accuracy parts (default WORK/primus_sync_accuracy)")
    ap.add_argument("--uncorrected", action="store_true", help="k = 0 for every stretch (the 1 November numbers)")
    a = ap.parse_args()
    W = Path(a.work)
    K = {}
    rule = json.loads((W / "primus_breath_rule" / "summary.json").read_text())
    for x in json.loads((W / "primus_breath_rule" / "stretch_k.json").read_text()):
        if x["arm"] == "cvp":
            K[(x["case"], x["stretch"])] = (0 if a.uncorrected else x["k"], x["T"], x["step"])
    loo = rule["leave_one_out"]
    p_step = {1: 0.0, 2: 1 - loo["step2_right_share"], 3: 1 - loo["step3_right_share"]}
    E, C, RAW, PW, TW, steps, SL, RC = [], [], [], [], [], {1: 0, 2: 0, 3: 0}, [], []
    missing = 0
    for p in sorted(glob.glob(str(Path(a.sync) if a.sync else W / "primus_sync_accuracy") + "/part_*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            A = (r.get("arms") or {}).get("cvp")
            if not A or len(A["t"]) < 6:
                continue
            t, y, pa = np.array(A["t"], float), np.array(A["lag"], float), np.array(r["pauses"], float)
            st = stretch_index(t, pa)
            edges = np.concatenate([[t[0]], pa[(pa > t[0]) & (pa < t[-1])], [t[-1]]])
            SL += list(np.diff(edges))
            ks = []
            for s in st:
                k, T, step = K.get((r["case"], int(s)), (0, 0.0, None))
                if step is None:
                    missing += 1
                ks.append((k, T, step))
            yc = y - np.array([k * T for k, T, _ in ks])
            near = np.array([np.min(np.abs(x - pa)) < STRADDLE_S if pa.size else False for x in t])
            e = np.array([predict("line_each", t, yc, st, i) - yc[i] for i in range(t.size)])
            m = ~near & np.isfinite(e)
            E += list(e[m]); C += [r["case"]] * int(m.sum()); RAW.append(float(np.median(yc))); RC.append(r["case"])
            for (k, T, step), keep in zip(ks, m):
                if keep and step is not None:
                    PW.append(p_step[step]); TW.append(T); steps[step] += 1
    E, C, RAW = np.array(E), np.array(C), np.array(RAW)
    cm = pd.Series(E).groupby(C).median()
    share = 0.249  # primus_stretch_models summary, two_references line_each (458 recordings), as before
    p = float(np.mean(PW)) if PW else 0.0
    Tm = float(np.median(TW)) if TW else float("nan")
    rng = np.random.default_rng(0)
    n = 200000
    core = rng.choice(E, n)
    Td = rng.choice(np.array(TW), n) if TW else np.full(n, Tm)
    u = rng.random(n)
    mix = core + np.where(u < p / 2, -Td, np.where(u < p, Td, 0.0))
    mix95 = [round(float(v), 3) for v in np.percentile(mix, [2.5, 97.5])]
    out = {"corrected": not a.uncorrected, "recordings": int(np.unique(C).size), "windows": int(E.size),
           "windows_without_a_rule_entry": missing,
           "u_headline_rsd_s": round(rsd(E), 4), "interval95_s": [round(float(v), 4) for v in np.percentile(E, [2.5, 97.5])],
           "median_abs_s": round(float(np.median(np.abs(E))), 4),
           "common_to_recording_rsd_s": round(rsd(cm.values), 4),
           "within_recording_rsd_s": round(rsd(E - pd.Series(C).map(cm).values), 4),
           "likely_u_s": round(rsd(E) * np.sqrt(share), 4), "likely_share_common": share,
           "breath_error": {"p_by_step": {str(k): round(v, 3) for k, v in p_step.items()},
                            "windows_by_step": {str(k): v for k, v in steps.items()},
                            "p_window_weighted": round(p, 3), "median_T_s": round(Tm, 3),
                            "mixture_sd_s": round(float(np.sqrt(p) * Tm), 3), "mixture_interval95_s": mix95,
                            "stretch_length_s_median": round(float(np.median(SL)), 1) if SL else None},
           "raw_lag_per_recording_s": {"median": round(float(np.median(RAW)), 3), "p5": round(float(np.percentile(RAW, 5)), 3),
                                       "p95": round(float(np.percentile(RAW, 95)), 3), "n": int(RAW.size), "rsd": round(rsd_c(RAW), 3)}}
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    tag = "" if not a.uncorrected else "_uncorrected"
    (o / f"summary{tag}.json").write_text(json.dumps(out, indent=1))
    (o / f"per_recording{tag}.json").write_text(json.dumps({"case": RC, "lag_median_s": [round(x, 4) for x in RAW]}))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
