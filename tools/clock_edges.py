# SPDX-License-Identifier: MIT
"""The VitalDB clock-link edge table: every measured link between two clocks, per recording, from existing outputs.

    python tools/clock_edges.py --work W --out W/clock_network/edges.csv

One row per measured link: recording, clocks a and b, value = x[a] - x[b], its standard error, the method, the data
it came from (sources, ';'-separated), and a label. Here x is a clock's drift rate against the recorder PC (ppm),
defined as the atlas does: how much the PC gains on the clock (BIS about +25, SNUADC step mode about -63, Primus
about +104). The PC is the reference (x = 0). The adjustment itself is dataset-neutral (physiosync.network);
everything VitalDB-specific (which outputs, which signs, which standard errors) is here.

Clocks (nodes):
  pc             the recorder PC
  snuadc_stamps  the SNUADC's record stamps (sample count plus the recorder's stamp arithmetic)
  snuadc         the SNUADC's own sample count (its crystal)
  bis            the BIS's own sample count
  primus         the Primus waveform's stamps (AWP)

Edges (methods) and their inputs under W:
  resets        snuadc_stamps-pc  signal_survey/continuity/snuadc_reset_drift_per_recording.csv (reset size / interval);
                with --reset-loss, scaled by 0.498 / 0.480: the stamps show 0.480 s of each 0.498 s correction, the
                rest being signal never sampled (paced-ecg), so size / interval understates the drift by 3.7%;
                with --drop-missed, recordings whose interval rate is under 1/1.5 of the rate timed from the first
                reset (snuadc_drift_per_recording.csv ppm_first) are left out: their interval most likely spans a
                reset that was not seen (in a gap), which halves the apparent rate
  arithmetic    snuadc_stamps-snuadc  +0.954 ppm in every recording (stamps fall behind the count by 0.954 ppm,
                so the PC gains 0.954 ppm more on the stamps than on the count); se 0.05
  bis_steps     bis-pc            signal_survey/continuity/bis_drift_per_recording.csv (step / interval)
  bursts        bis-snuadc        direct_rate_intervals/agg/direct_rate_per_interval.csv (ECG-EEG lag slope on both
                                  devices' counts; the lag falls when x[bis] > x[snuadc], so value = -slope)
  breath        primus-pc         results/primus_drift/probe_*.jsonl (AWP-to-CVP lag slope, SNUADC on its stamp model,
                                  one offset per stretch between AWP pauses); depends on the SNUADC stamps
  pip_numeric   primus-pc         primus_numeric_clock/part_*.jsonl (PIP delay slope within stretches; PIP stamped by
                                  the PC)

Standard errors. Per recording, from each method's own scatter where it is measured (bursts, breath, pip_numeric:
the slope's standard error); for the correction-step rates, from the documented within-recording variation between
intervals (SNUADC 1.2 ppm, BIS 3.5 ppm; device-correction-schedules) and stamp quantisation (0.3 and 1.0 ppm),
divided by the number of intervals. The reconciliation reports each method's variance factor, which says whether
these are right.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARITH_PPM, ARITH_SE = 0.954, 0.05
RESET_TRUE_S, RESET_STAMPS_S = 0.498, 0.4804
MISSED_RATIO = 1.5
SNU_SWING, SNU_QUANT = 1.2, 0.3
BIS_SWING, BIS_QUANT = 3.5, 1.0
FIELDS = ["recording", "a", "b", "value", "se", "method", "sources", "label"]


def stretch_slope(t_s, y, pauses_s):
    """Common slope (per second) and its standard error, one offset per stretch between pauses."""
    t_s, y = np.asarray(t_s, float), np.asarray(y, float)
    st = np.searchsorted(np.asarray(pauses_s, float), t_s)
    u = np.unique(st)
    X = np.column_stack([t_s] + [(st == s).astype(float) for s in u])
    if len(y) <= X.shape[1] + 1:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = float(res @ res) / (len(y) - X.shape[1])
    cov = s2 * np.linalg.pinv(X.T @ X)
    return float(beta[0]), float(np.sqrt(cov[0, 0]))


def edges(W: Path, reset_loss: bool = False, drop_missed: bool = False) -> pd.DataFrame:
    rows = []
    cont = W / "signal_survey" / "continuity"
    s = pd.read_csv(cont / "snuadc_reset_drift_per_recording.csv")
    if drop_missed:
        f = pd.read_csv(cont / "snuadc_drift_per_recording.csv")[["case", "ppm_first"]]
        s = s.merge(f, on="case", how="left")
        s = s[~(s.ppm_first / s.ppm > MISSED_RATIO)]
    for r in s.itertuples():
        n = max(int(r.n), 1)
        se = float(np.hypot(SNU_SWING, SNU_QUANT) / np.sqrt(n))
        v = r.ppm * (RESET_TRUE_S / RESET_STAMPS_S if reset_loss else 1.0)
        rows.append([r.case, "snuadc_stamps", "pc", v, se, "resets", "snuadc stamps", f"{n} intervals"])
    b = pd.read_csv(cont / "bis_drift_per_recording.csv")
    for r in b.itertuples():
        n = max(int(r.n), 1)
        se = float(np.hypot(BIS_SWING, BIS_QUANT) / np.sqrt(n))
        rows.append([r.case, "bis", "pc", r.ppm, se, "bis_steps", "bis stamps", f"{n} intervals"])
    I = W / "direct_rate_intervals" / "agg" / "direct_rate_per_interval.csv"
    if I.exists():
        d = pd.read_csv(I)
        for case, g in d.groupby("case"):
            w = 1 / np.maximum(g.se_ppm.to_numpy(), 0.5) ** 2
            v = float(np.sum(w * g.ppm) / np.sum(w))
            rows.append([case, "bis", "snuadc", -v, float(1 / np.sqrt(w.sum())), "bursts", "ecg;eeg",
                         f"{len(g)} intervals"])
    for p in sorted(glob.glob(str(W / "results" / "primus_drift" / "probe_*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            f = stretch_slope(r["tl"], r["ll"], r["p"])
            if f:
                rows.append([r["case"], "primus", "pc", f[0] * 1e6, f[1] * 1e6, "breath", "awp;cvp;snuadc stamps",
                             f"{len(r['tl'])} windows"])
    for p in sorted(glob.glob(str(W / "primus_numeric_clock" / "part_*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            B = np.array([x for x in r.get("blocks", []) if x[2] >= 0.5], float)
            if len(B) < 4:
                continue
            f = stretch_slope(B[:, 0], B[:, 1], r.get("pauses") or [])
            if f:
                rows.append([r["case"], "primus", "pc", f[0] * 1e6, f[1] * 1e6, "pip_numeric", "awp;pip numerics",
                             f"{len(B)} blocks"])
    E = pd.DataFrame(rows, columns=FIELDS)
    E = E[np.isfinite(E.value) & np.isfinite(E.se) & (E.se > 0)]
    # the stamp arithmetic joins the SNUADC stamps to its count wherever both SNUADC nodes are used
    need = set(E.recording[E.method == "resets"]) & set(E.recording[E.method == "bursts"])
    E = pd.concat([E, pd.DataFrame([[c, "snuadc_stamps", "snuadc", ARITH_PPM, ARITH_SE, "arithmetic", "snuadc stamps",
                                     "constant"] for c in sorted(need)], columns=FIELDS)], ignore_index=True)
    return E.sort_values(["recording", "method"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reset-loss", action="store_true", help="scale reset-based SNUADC rates by 0.498 / 0.480")
    ap.add_argument("--drop-missed", action="store_true", help="leave out reset intervals that likely span a missed reset")
    a = ap.parse_args()
    E = edges(Path(a.work), a.reset_loss, a.drop_missed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    E.to_csv(a.out, index=False)
    print(E.groupby("method").agg(edges=("value", "size"), median=("value", "median"), se_median=("se", "median")).round(2))


if __name__ == "__main__":
    main()
